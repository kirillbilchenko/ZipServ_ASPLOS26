# Copyright MxWave contributors.
# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Exact post-Marlin NVFP4 scale storage and a versioned native interface."""
from __future__ import annotations

import ctypes
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .nvfp4_codec import decode, encode


@dataclass
class PackedScales:
    shape: tuple[int, int, int]
    descriptor: torch.Tensor
    buffers: list[torch.Tensor]
    records: list[dict]
    source_sha256: str

    @property
    def retained_bytes(self) -> int:
        return sum(t.nbytes for t in self.buffers)

    @property
    def allocated_bytes(self) -> int:
        # CUDA's small allocation granularity matters for per-expert metadata.
        return sum((t.nbytes + 511) // 512 * 512 for t in self.buffers)


def layout_rejection(shape: tuple[int, ...], parts: int) -> str | None:
    if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
        return "Require positive [experts, groups, output] scale dimensions"
    e, groups, n = shape
    if parts not in (1, 2) or n < 64 or n & (n - 1) or n % (parts * 32):
        return "Five-bit consumer requires power-of-two output columns and complete high-plane words"
    if (groups * 16) % 64 or e * groups * n >= 2**31:
        return "NVFP4 scales exceed the aligned group-16/int32 contract"
    return None


def pack_scales(scales: torch.Tensor, parts: int) -> PackedScales:
    """Pack one expert at a time; preserve every post-Marlin scale byte."""
    reason = layout_rejection(tuple(scales.shape), parts)
    if reason:
        raise ValueError(reason)
    if (scales.dtype != torch.float8_e4m3fn or scales.device.type != "cuda"
            or not scales.is_contiguous()):
        raise ValueError("Require contiguous CUDA post-Marlin S0E5M3 scale storage")
    e, _, n = scales.shape
    retained, pointers, records = [], [], []
    source_hash = hashlib.sha256()
    for expert in range(e):
        # Bound host staging to one expert, not an entire model shard.
        source = scales[expert].detach().view(torch.uint8).cpu().numpy()
        source_hash.update(source.tobytes())
        for part in range(parts):
            raw = source[:, part * (n // parts):(part + 1) * (n // parts)]
            encoded = encode(raw)
            if not np.array_equal(decode(encoded), raw.reshape(-1)):
                raise RuntimeError("Independent NVFP4 scale CPU reconstruction changed bytes")
            arrays = (encoded.low, encoded.high, encoded.palette, encoded.raw)
            buffers = [torch.from_numpy(a).to(scales.device) if a.size else None for a in arrays]
            pointers.append([b.data_ptr() if b is not None else 0 for b in buffers])
            retained.extend(b for b in buffers if b is not None)
            records.append({"expert": expert, "part": part, "source_bytes": raw.size,
                            "encoded_bytes": encoded.nbytes, "raw_fallback": bool(encoded.raw.size)})
    table = torch.tensor(pointers, dtype=torch.int64, device=scales.device)
    row_shift, part_shift = (n // 16).bit_length() - 1, (n // parts // 16).bit_length() - 1
    # Keep an int64 descriptor distinguishable from real scale tensors, including
    # aliases/copies. The CUDA ABI is NvScales {pointer, four int32 values}.
    raw_descriptor = struct.pack("<Qiiii", table.data_ptr(), row_shift, part_shift, parts, 0)
    descriptor = torch.frombuffer(bytearray(raw_descriptor), dtype=torch.int64).to(scales.device)
    retained.extend((table, descriptor))
    return PackedScales(tuple(scales.shape), descriptor, retained, records, source_hash.hexdigest())


class NativeMoe:
    """Raw control or five-bit consumer, using standard math and FP32 reduction."""

    def __init__(self, path: Path) -> None:
        self.library = ctypes.CDLL(str(path.resolve()))
        for name in ("zipserv_nvfp4_abi", "zipserv_nvfp4_features", "zipserv_nvfp4_encoding"):
            if not hasattr(self.library, name):
                raise ValueError("Require a native library built for ZipServ NVFP4")
            getattr(self.library, name).restype = ctypes.c_int
        if (self.library.zipserv_nvfp4_abi() != 1
                or self.library.zipserv_nvfp4_features() != 7):
            raise ValueError("Incompatible NVFP4 ABI or missing standard-math/reuse-barrier features")
        encoding = self.library.zipserv_nvfp4_encoding()
        if encoding not in (0, 3):
            raise ValueError("Only raw and exact five-bit offset consumers are qualified for this interface")
        self.compressed = encoding == 3
        self.library.nv_last_error.restype = ctypes.c_char_p
        self.library.nv_plan.argtypes = [ctypes.c_int] * 7 + [ctypes.POINTER(ctypes.c_int)]
        self.library.nv_plan.restype = ctypes.c_int
        self.library.nv_run.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 10 + [ctypes.c_size_t]
        self.library.nv_run.restype = ctypes.c_int
        self.library.nv_decode.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2 + [ctypes.c_size_t]
        self.library.nv_decode.restype = ctypes.c_int

    def _check(self, status: int) -> None:
        if status:
            error = self.library.nv_last_error()
            raise RuntimeError(error.decode() if error else "NVFP4 native operation failed")

    def plan(self, m: int, n: int, k: int, block: int, topk: int, experts: int,
             device: int) -> tuple[int, int, int]:
        plan = (ctypes.c_int * 3)()
        self._check(self.library.nv_plan(m, n, k, block, topk, experts, device, plan))
        return tuple(plan)

    def decode_into(self, packed: PackedScales, output: torch.Tensor) -> None:
        if not self.compressed:
            raise ValueError("Packed scales require the delta5 consumer")
        if (tuple(output.shape) != packed.shape or output.dtype != torch.uint8
                or output.device != packed.descriptor.device or not output.is_contiguous()):
            raise ValueError("Decoder output does not match the packed scale layout")
        _, groups, n = packed.shape
        with torch.cuda.device(output.device):
            self._check(self.library.nv_decode(
                packed.descriptor.data_ptr(), output.data_ptr(), groups * n // 16,
                output.numel() // 16, torch.cuda.current_stream(output.device).cuda_stream))

    def decode(self, packed: PackedScales) -> torch.Tensor:
        output = torch.empty(packed.shape, dtype=torch.uint8, device=packed.descriptor.device)
        self.decode_into(packed, output)
        return output

    def launch(self, tensors: tuple[torch.Tensor, ...], geometry: tuple[int, ...]) -> None:
        """Internal launch after the adapter validates geometry and buffer lifetimes."""
        x = tensors[0]
        with torch.cuda.device(x.device):
            self._check(self.library.nv_run(
                *(t.data_ptr() for t in tensors), *geometry,
                torch.cuda.current_stream(x.device).cuda_stream))
