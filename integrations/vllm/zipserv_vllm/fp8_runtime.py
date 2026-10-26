# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Experimental ZFB1 Marlin consumer with exact decode for large prefills.

Compression is lossless storage. Fused GEMM still needs matched speed and quality
validation; a storage threshold is not a claim of lower latency.
"""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import torch

_SCRATCH: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


@lru_cache(maxsize=1)
def library() -> ctypes.CDLL:
    path = Path(os.environ["ZIPSERV_FP8_LIBRARY"]).resolve()
    lib = ctypes.CDLL(str(path))
    for name in ("abi", "compressed", "codec", "compact_only"):
        function = getattr(lib, "zipserv_marlin_fp8_" + name)
        function.argtypes, function.restype = [], ctypes.c_int
    if (lib.zipserv_marlin_fp8_abi() != 1 or lib.zipserv_marlin_fp8_compressed() != 1
            or lib.zipserv_marlin_fp8_codec() != 3 or lib.zipserv_marlin_fp8_compact_only() != 0):
        raise ValueError("Require the mixed-tile, byteplane FP8 native library (ABI 1, codec 3)")
    lib.zipserv_marlin_fp8_last_error.argtypes = []
    lib.zipserv_marlin_fp8_last_error.restype = ctypes.c_char_p
    lib.zipserv_marlin_fp8_run.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    lib.zipserv_marlin_fp8_run.restype = ctypes.c_int
    lib.zipserv_marlin_fp8_decode.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_void_p]
    lib.zipserv_marlin_fp8_decode.restype = ctypes.c_int
    return lib


def _launch(x: torch.Tensor, data: torch.Tensor, scales: torch.Tensor,
            output: torch.Tensor, temporary: torch.Tensor, locks: torch.Tensor,
            n: int, k: int, atomic: bool) -> None:
    m = x.shape[0]
    if (x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] != k
            or not 1 <= m <= 128 or min(n, k) < 64 or n % 64 or k % 64
            or (n % 128 and k % 128) or n * k >= 2**31):
        raise ValueError("Unsupported BF16/MXFP8 Marlin GEMM shape")
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    specs = [(data, torch.uint8, None), (scales, torch.float8_e8m0fnu, (k // 32, n)),
             (output, torch.bfloat16, (m, n)), (temporary, torch.float32, (sms * 64 * 256,)),
             (locks, torch.int32, (sms,))]
    if x.device.type != "cuda" or not x.is_contiguous() or x.data_ptr() % 16:
        raise ValueError("Require aligned contiguous CUDA inputs")
    for tensor, dtype, shape in specs:
        if (tensor.device != x.device or tensor.dtype != dtype or not tensor.is_contiguous()
                or tensor.data_ptr() % 16 or (shape is not None and tuple(tensor.shape) != shape)):
            raise ValueError("Invalid FP8 buffer contract")
    inputs = {t.untyped_storage().data_ptr() for t in (x, data, scales)}
    outputs = [t.untyped_storage().data_ptr() for t in (output, temporary, locks)]
    if inputs.intersection(outputs) or len(set(outputs)) != 3:
        raise ValueError("FP8 output and scratch must not alias inputs or each other")
    with torch.cuda.device(x.device):
        lib = library()
        status = lib.zipserv_marlin_fp8_run(
            *(t.data_ptr() for t in (x, data, scales, output, temporary, locks)),
            m, n, k, x.device.index, int(atomic), torch.cuda.current_stream(x.device).cuda_stream)
        if status:
            raise RuntimeError(lib.zipserv_marlin_fp8_last_error().decode(errors="replace"))


_DISPATCH = torch.library.Library("zipserv_fp8", "DEF")
_DISPATCH.define("gemm" + torch.library.infer_schema(_launch, mutates_args=("output", "temporary", "locks")))
_DISPATCH.impl("gemm", _launch, "CUDA")


@torch.library.register_fake("zipserv_fp8::gemm")
def _fake_gemm(*args, **kwargs) -> None:
    return None


@torch.library.custom_op("zipserv_fp8::decode", mutates_args=(), device_types="cuda")
def decode(data: torch.Tensor, n: int, k: int) -> torch.Tensor:
    if (data.dtype != torch.uint8 or data.device.type != "cuda" or not data.is_contiguous()
            or data.data_ptr() % 16 or n < 64 or k < 64 or n % 64 or k % 64 or n * k >= 2**31):
        raise ValueError("Invalid encoded FP8 buffer or shape")
    output = torch.empty((k // 16, n * 4), dtype=torch.int32, device=data.device)
    with torch.cuda.device(data.device):
        status = library().zipserv_marlin_fp8_decode(data.data_ptr(), output.data_ptr(), n * k,
            torch.cuda.current_stream(data.device).cuda_stream)
        if status:
            raise RuntimeError(f"FP8 decode failed: CUDA status {status}")
    return output


@decode.register_fake
def _fake_decode(data: torch.Tensor, n: int, k: int) -> torch.Tensor:
    return torch.empty((k // 16, n * 4), dtype=torch.int32, device=data.device)


def apply(layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        get_marlin_workspace,
        marlin_pad_dim,
        marlin_unpad_output,
        should_use_atomic_add_reduce,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_mxfp8_marlin_linear,
    )
    from vllm.v1.worker.workspace import is_workspace_manager_initialized

    n, k = layer.zipserv_fp8_shape
    logical_n, logical_k = layer.output_size_per_partition, layer.input_size_per_partition
    if x.dtype != torch.bfloat16 or x.shape[-1] != logical_k:
        raise ValueError("The FP8 trial requires BF16 activations with the original input width")
    m = x.numel() // logical_k
    if m == 0:
        return x.new_empty((*x.shape[:-1], logical_n))
    if is_workspace_manager_initialized():
        # Small warmup batches can take only our fused path on an auxiliary
        # shared-expert stream. Reserve stock fallback locks in that same
        # stream/ubatch/lane before vLLM freezes its workspace. Larger mixed
        # prefill/decode batches may need stock Marlin there during serving.
        get_marlin_workspace(x.device)
    if m > 128 or bias is not None:
        # Per-call exact reconstruction is released after stock GEMM; there is
        # no persistent raw-weight cache. Report its peak separately.
        layer.zipserv_fp8_calls["decode_then_stock"] += 1
        restored = decode(layer.zipserv_fp8_data, n, k)
        return apply_mxfp8_marlin_linear(x, restored, layer.weight_scale, None,
                                         logical_n, logical_k, bias)
    layer.zipserv_fp8_calls["fused"] += 1
    flat = marlin_pad_dim(x.reshape(-1, logical_k), logical_k, k).contiguous()
    stream = torch.cuda.current_stream(x.device).cuda_stream
    key = (x.device.index, stream)
    if key not in _SCRATCH:
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        _SCRATCH[key] = (torch.empty(sms * 64 * 256, device=x.device, dtype=torch.float32),
                         torch.zeros(sms, device=x.device, dtype=torch.int32))
    temporary, locks = _SCRATCH[key]
    output = x.new_empty((m, n))
    atomic = should_use_atomic_add_reduce(m=m, n=n, k=k, device=x.device, dtype=x.dtype)
    torch.ops.zipserv_fp8.gemm.default(flat, layer.zipserv_fp8_data, layer.weight_scale,
                                     output, temporary, locks, n, k, atomic)
    return marlin_unpad_output(output, logical_n, n).reshape(*x.shape[:-1], logical_n)


def scratch_bytes() -> int:
    return sum(t.nbytes for buffers in _SCRATCH.values() for t in buffers)
