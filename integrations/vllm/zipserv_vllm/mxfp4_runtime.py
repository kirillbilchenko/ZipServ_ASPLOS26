# Copyright MxWave contributors.
# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
# Adapted for ZipServ; see NOTICE and mxfp4-provenance.json.
"""Bounded component interface for the separately compiled grouped MoE kernels.

This is not a vLLM engine plugin. Callers own every input and scratch tensor
until CUDA completion, including during graph replay and on non-default streams.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class PackedScales:
    """Exact records for a contiguous [experts, groups, output] E8M0 tensor."""

    shape: tuple[int, ...]
    records: torch.Tensor
    exceptions: torch.Tensor
    descriptor: torch.Tensor
    encoding: str = "records"
    affine_palette: bool = False
    stage_metadata_verified: bool = False
    aligned_nibble_base: int | None = None

    @property
    def retained_bytes(self) -> int:
        """Include all record, exception and descriptor tensor payloads."""
        return sum(t.numel() * t.element_size()
                   for t in (self.records, self.exceptions, self.descriptor))


def pack_scales(scales: torch.Tensor) -> PackedScales:
    """Pack verified E8M0 bytes without quantization, with bounded CPU scratch."""
    if (scales.device.type != "cuda" or scales.ndim != 3 or not scales.is_contiguous()
            or scales.dtype != torch.float8_e8m0fnu or not scales.numel()
            or scales.numel() % 16 or scales.numel() >= 2**31):
        raise ValueError("Require contiguous 3D CUDA E8M0 scales within the native int32 bound")
    import numpy as np

    from .mxfp4_codec import pack_sparse, unpack_sparse

    raw = scales.view(torch.uint8).cpu().numpy().reshape(-1)
    if np.any(raw == 255):
        raise ValueError("Reserved E8M0 NaN scale cannot qualify for the finite MXFP4 consumer")
    packed = pack_sparse(raw)
    if not np.array_equal(unpack_sparse(packed), raw):
        raise RuntimeError("Independent sparse-scale CPU round-trip changed bytes")
    records = torch.from_numpy(packed.selectors.view(np.int32)).to(scales.device)
    exceptions = torch.from_numpy(packed.exceptions).to(scales.device)
    descriptor = torch.tensor([records.data_ptr(), exceptions.data_ptr(), packed.descriptor],
                              dtype=torch.int64, device=scales.device)
    return PackedScales(tuple(scales.shape), records, exceptions, descriptor, "sparse_palette",
                        bool(packed.descriptor & (1 << 24)))


@dataclass
class Buffers:
    """Output, FP32 partial reduction and zero-initialized managed lock storage."""

    output: torch.Tensor
    temporary: torch.Tensor
    locks: torch.Tensor


def allocate_buffers(
    m: int, n: int, topk: int, block: int, sorted_capacity: int, device: torch.device
) -> Buffers:
    """Match stock's grouped GEMM temporary and lock bounds, allocating once."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    count = min(n * sorted_capacity, sms * 4 * block * 256)
    if block == 8:
        count *= 2
    return Buffers(
        torch.empty((m * topk, n), dtype=torch.bfloat16, device=device),
        torch.empty(count, dtype=torch.float32, device=device),
        torch.zeros(sms * 4, dtype=torch.int32, device=device),
    )


class NativeMoe:
    """Launch grouped BF16/MXFP4 GEMMs on the caller's current CUDA stream."""

    def __init__(self, path: Path) -> None:
        """Load one raw or compressed-scale variant from its isolated library."""
        self.library = ctypes.CDLL(str(path.resolve()))
        if not hasattr(self.library, "zipserv_mxfp4_abi"):
            raise ValueError("Require a native library built for ZipServ MXFP4")
        self.library.zipserv_mxfp4_abi.restype = ctypes.c_int
        self.library.zipserv_mxfp4_features.restype = ctypes.c_int
        if self.library.zipserv_mxfp4_abi() != 1 or self.library.zipserv_mxfp4_features() != 7:
            raise ValueError("Incompatible MXFP4 ABI or missing generic-palette/reuse-barrier feature")
        self.library.mxwave_moe_compressed.restype = ctypes.c_int
        self.compressed = bool(self.library.mxwave_moe_compressed())
        self.affine_only = False
        if hasattr(self.library, "mxwave_moe_affine_only"):
            self.library.mxwave_moe_affine_only.restype = ctypes.c_int
            self.affine_only = bool(self.library.mxwave_moe_affine_only())
        self.shared_stage = False
        if hasattr(self.library, "mxwave_moe_shared_stage"):
            self.library.mxwave_moe_shared_stage.restype = ctypes.c_int
            self.shared_stage = bool(self.library.mxwave_moe_shared_stage())
        self.nibble_stage = False
        if hasattr(self.library, "mxwave_moe_nibble_stage"):
            self.library.mxwave_moe_nibble_stage.restype = ctypes.c_int
            self.nibble_stage = bool(self.library.mxwave_moe_nibble_stage())
        self.static_nibble_stage = False
        if hasattr(self.library, "mxwave_moe_static_nibble_stage"):
            self.library.mxwave_moe_static_nibble_stage.restype = ctypes.c_int
            self.static_nibble_stage = bool(self.library.mxwave_moe_static_nibble_stage())
        self.u8_fallback: NativeMoe | None = None
        self.encoding = "records"
        if hasattr(self.library, "mxwave_moe_encoding"):
            self.library.mxwave_moe_encoding.restype = ctypes.c_int
            encodings = {0: "records", 1: "nibbles", 2: "aligned_nibbles", 3: "sparse_palette"}
            self.encoding = encodings[self.library.mxwave_moe_encoding()]
        if (not self.compressed or self.encoding != "sparse_palette" or self.affine_only
                or not self.shared_stage or self.nibble_stage):
            raise ValueError("Require the generic sparse-palette shared-U8 consumer")
        self.library.mxwave_moe_last_error.restype = ctypes.c_char_p
        self.library.mxwave_moe_plan.argtypes = [ctypes.c_int] * 7 + [ctypes.POINTER(ctypes.c_int)]
        self.library.mxwave_moe_plan.restype = ctypes.c_int
        self.library.mxwave_moe_run.argtypes = (
            [ctypes.c_void_p] * 10 + [ctypes.c_int] * 11 + [ctypes.c_size_t]
        )
        self.library.mxwave_moe_run.restype = ctypes.c_int
        self.library.mxwave_moe_decode.argtypes = (
            [ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_size_t]
        )
        self.library.mxwave_moe_decode.restype = ctypes.c_int
        self.has_bf16_decoder = hasattr(self.library, "mxwave_moe_decode_bf16")
        if self.has_bf16_decoder:
            self.library.mxwave_moe_decode_bf16.argtypes = self.library.mxwave_moe_decode.argtypes
            self.library.mxwave_moe_decode_bf16.restype = ctypes.c_int

    def plan(
        self, m: int, n: int, k: int, block: int, topk: int, experts: int, device: int
    ) -> tuple[int, int, int]:
        """Choose stock's tile once; use this same plan for both compared variants."""
        values = (ctypes.c_int * 3)()
        if self.library.mxwave_moe_plan(m, n, k, block, topk, experts, device, values):
            raise RuntimeError("No supported grouped GEMM plan")
        return values[0], values[1], values[2]

    def set_u8_fallback(self, fallback: NativeMoe) -> None:
        """Bind an exact U8-stage consumer outside capture and all timed launches."""
        if (not self.static_nibble_stage or fallback is self or fallback.nibble_stage
                or not fallback.shared_stage or fallback.encoding != self.encoding):
            raise ValueError("Static nibble staging requires a separate exact U8-stage fallback")
        self.u8_fallback = fallback

    def scale_consumer(self, scales: PackedScales) -> NativeMoe:
        """Select a verified tensor layout once, rejecting a missing exact fallback."""
        if self.nibble_stage and not scales.stage_metadata_verified:
            raise ValueError("Nibble staging requires verified complete source ranges")
        if self.static_nibble_stage and scales.aligned_nibble_base is None:
            if self.u8_fallback is None:
                raise ValueError("Unsupported range requires an explicitly bound exact U8 fallback")
            return self.u8_fallback
        return self

    def decode(self, scales: PackedScales) -> torch.Tensor:
        """Check the GPU record decoder independently of routing or GEMM arithmetic."""
        if scales.encoding != self.encoding:
            raise ValueError("Native library and scale encoding differ")
        if self.nibble_stage and not scales.stage_metadata_verified:
            raise ValueError("Nibble staging requires verified complete source ranges")
        output = torch.empty(scales.shape, dtype=torch.uint8, device=scales.descriptor.device)
        if self.library.mxwave_moe_decode(
            scales.descriptor.data_ptr(), output.data_ptr(), output.numel(),
            torch.cuda.current_stream(output.device).cuda_stream,
        ):
            raise RuntimeError("Grouped scale decoding launch failed")
        return output

    def decode_bf16(self, scales: PackedScales) -> torch.Tensor:
        """Expose the exact stock BF16 fragment bits for independent decoder checks."""
        if not self.has_bf16_decoder:
            raise ValueError("This native library has no BF16 fragment decoder")
        if scales.encoding != self.encoding:
            raise ValueError("Native library and scale encoding differ")
        if self.nibble_stage and not scales.stage_metadata_verified:
            raise ValueError("Nibble staging requires verified complete source ranges")
        shape = (*scales.shape[:-1], scales.shape[-1] * 2)
        output = torch.empty(shape, dtype=torch.uint8, device=scales.descriptor.device)
        if self.library.mxwave_moe_decode_bf16(
            scales.descriptor.data_ptr(), output.data_ptr(), output.numel() // 2,
            torch.cuda.current_stream(output.device).cuda_stream,
        ):
            raise RuntimeError("Grouped BF16 fragment decoding launch failed")
        return output

    def bind(
        self, x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor | PackedScales,
        buffers: Buffers, routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        routing_weights: torch.Tensor, topk: int, block: int, multiply: bool,
        plan: tuple[int, int, int],
    ) -> Any:
        """Validate all static buffers once and return an allocation-free launch closure."""
        shape = tuple(scales.shape)
        if len(shape) != 3:
            raise ValueError("Expected [experts, groups, output] scale storage")
        e, groups, n = shape
        k = groups * 32
        m = x.shape[0]
        if min(e, m, n, k, topk) < 1 or n % 64 or k % 64 or block not in (8, 16, 32, 48, 64):
            raise ValueError("Unsupported grouped GEMM shape or block size")
        if self.compressed != isinstance(scales, PackedScales):
            raise ValueError("Native variant and scale representation differ")
        if isinstance(scales, PackedScales) and scales.encoding != self.encoding:
            raise ValueError("Native library and packed scale encoding differ")
        if self.affine_only and (not isinstance(scales, PackedScales) or not scales.affine_palette):
            raise ValueError("Affine kernel requires a verified consecutive scale palette")
        if self.nibble_stage and (not isinstance(scales, PackedScales)
                                  or not scales.stage_metadata_verified):
            raise ValueError("Nibble staging requires verified complete source ranges")
        if isinstance(scales, PackedScales):
            consumer = self.scale_consumer(scales)
            if consumer is not self:
                return consumer.bind(x, weight, scales, buffers, routing, routing_weights,
                                     topk, block, multiply, plan)
        if isinstance(scales, torch.Tensor) and scales.dtype != torch.float8_e8m0fnu:
            raise ValueError("Raw scales must retain the runtime E8M0 dtype")
        storage = scales.descriptor if isinstance(scales, PackedScales) else scales
        if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != k:
            raise ValueError("Expected compatible BF16 input")
        if weight.dtype != torch.int32 or tuple(weight.shape) != (e, k // 16, n * 2):
            raise ValueError("Expected grouped stock-Marlin-repacked int32 weights")
        if tuple(buffers.output.shape) != (m * topk, n):
            raise ValueError("Native GEMM output shape is wrong")
        if (buffers.output.dtype != torch.bfloat16 or buffers.temporary.dtype != torch.float32
                or buffers.locks.dtype != torch.int32 or routing_weights.dtype != torch.float32
                or any(t.dtype != torch.int32 for t in routing)):
            raise ValueError("Native buffer dtype mismatch")
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        needed = min(n * routing[0].numel(), sms * 4 * block * 256) * (2 if block == 8 else 1)
        if (buffers.temporary.numel() < needed or buffers.locks.numel() < sms * 4
                or routing_weights.numel() < m * topk or routing[2].numel() != 1):
            raise ValueError("Insufficient grouped GEMM scratch or routing metadata")
        tensors = (x, weight, storage, buffers.output, buffers.temporary, buffers.locks,
                   *routing, routing_weights)
        if x.device.type != "cuda" or any(t.device != x.device or not t.is_contiguous()
                                         for t in tensors):
            raise ValueError("All buffers must be contiguous on the same CUDA device")
        owners = (tensors, scales, buffers)

        def launch() -> torch.Tensor:
            _ = owners  # Keep every referenced allocation alive through the caller's CUDA work.
            status = self.library.mxwave_moe_run(
                *(t.data_ptr() for t in tensors), m, n, k, e, topk, block, int(multiply),
                x.device.index, *plan, torch.cuda.current_stream(x.device).cuda_stream,
            )
            if status:
                raise RuntimeError(self.library.mxwave_moe_last_error().decode())
            return buffers.output

        return launch
