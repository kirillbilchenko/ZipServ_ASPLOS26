# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
# Adapted from MxWave; modified for this checkout. See NOTICE.
"""Lossless BF16 packing and stream-aware CUDA calls to the external ZipServ bridge."""

from __future__ import annotations

import ctypes
import os
import weakref
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

BUFFER_NAMES = ("sm", "rare", "bitmap1", "bitmap2", "bitmap3", "medium", "global")


class _NativePacked(ctypes.Structure):
    _fields_ = [
        ("buffers", ctypes.c_void_p * 7),
        ("lengths", ctypes.c_uint64 * 7),
        ("m", ctypes.c_int32),
        ("k", ctypes.c_int32),
        ("max_common", ctypes.c_int32),
        ("max_rare", ctypes.c_int32),
        ("start_exp", ctypes.c_int32),
    ]


@lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    path = os.environ.get("ZIPSERV_LIBRARY")
    if not path or not Path(path).is_file():
        raise RuntimeError(
            "Build the ZipServ bridge and set ZIPSERV_LIBRARY to its shared library"
        )
    lib = ctypes.CDLL(path)
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.zipserv_abi_version.restype = integer
    if lib.zipserv_abi_version() != 2:
        raise RuntimeError("Unsupported ZipServ bridge ABI")
    lib.zipserv_error.restype = ctypes.c_char_p
    lib.zipserv_compress.argtypes = [
        ptr,
        integer,
        integer,
        ctypes.POINTER(_NativePacked),
    ]
    lib.zipserv_compress.restype = integer
    lib.zipserv_free.argtypes = [ctypes.POINTER(_NativePacked)]
    lib.zipserv_free.restype = None
    if hasattr(lib, "zipserv_mm") and hasattr(lib, "zipserv_decode"):
        lib.zipserv_mm.argtypes = (
            [ptr] * 8 + [integer] * 3 + [ptr] * 2 + [integer] * 3 + [ptr, integer]
        )
        lib.zipserv_mm.restype = integer
        lib.zipserv_decode.argtypes = [ptr] * 8 + [integer] * 3 + [ptr] + [integer] * 2
        lib.zipserv_decode.restype = integer
    return lib


def _cuda_library() -> ctypes.CDLL:
    lib = _library()
    if not hasattr(lib, "zipserv_mm") or not hasattr(lib, "zipserv_decode"):
        raise RuntimeError(
            "This ZipServ library has no CUDA support; rebuild with --arch for GPU execution"
        )
    return lib


def _free_packed(lib: ctypes.CDLL, native: _NativePacked) -> None:
    lib.zipserv_free(ctypes.byref(native))


class _NativeBufferOwner:
    """Release native allocations after the last tensor storage view is gone."""

    __slots__ = ("__weakref__", "_finalizer")

    def __init__(self, lib: ctypes.CDLL, native: _NativePacked) -> None:
        # Keep both the descriptor and its library alive independently of the
        # library cache. No callback argument holds a reference to this owner.
        self._finalizer = weakref.finalize(self, _free_packed, lib, native)
        # Do not free live tensor storage early during interpreter shutdown.
        # The OS reclaims allocations still reachable when the process exits.
        self._finalizer.atexit = False


def _check(status: int) -> None:
    if status:
        message = _library().zipserv_error().decode("utf-8", errors="replace")
        raise RuntimeError(f"ZipServ native call failed ({status}): {message}")


@dataclass(frozen=True)
class PackedWeight:
    """Seven owned byte buffers and the native BF16 decoding metadata."""

    buffers: tuple[Tensor, ...]
    m: int
    k: int
    max_common: int
    max_rare: int
    start_exp: int

    @property
    def storage_bytes(self) -> int:
        """Return logical payload and index size, excluding allocator padding."""
        return sum(buffer.numel() for buffer in self.buffers)

    def to(self, device: torch.device | str) -> PackedWeight:
        """Transfer packed bytes without changing their representation."""
        return PackedWeight(
            tuple(buffer.to(device) for buffer in self.buffers),
            self.m,
            self.k,
            self.max_common,
            self.max_rare,
            self.start_exp,
        )


def compress(weight: Tensor) -> PackedWeight:
    """Pack CPU BF16 bits and share native buffers with tensors without copying."""
    if (
        weight.device.type != "cpu"
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
    ):
        raise ValueError(
            "ZipServ compression requires a two-dimensional CPU BF16 tensor"
        )
    m, k = weight.shape
    if m <= 0 or k <= 0 or m % 64 or k % 64 or m * k > 2**31 - 4097:
        raise ValueError(
            "ZipServ requires positive, 64-aligned dimensions and int32 offsets"
        )
    weight = weight.contiguous()
    native = _NativePacked()
    lib = _library()
    _check(lib.zipserv_compress(weight.data_ptr(), m, k, ctypes.byref(native)))
    try:
        owner = _NativeBufferOwner(lib, native)
    except BaseException:
        lib.zipserv_free(ctypes.byref(native))
        raise
    buffers = []
    for pointer, length in zip(native.buffers, native.lengths, strict=True):
        if not length:
            buffers.append(torch.empty(0, dtype=torch.uint8))
            continue
        view = (ctypes.c_uint8 * length).from_address(pointer)
        # frombuffer retains this Python buffer object for the tensor storage's
        # lifetime, including slices and detached views of the resulting tensor.
        view._zipserv_owner = owner
        buffers.append(torch.frombuffer(view, dtype=torch.uint8))
    return PackedWeight(
        tuple(buffers),
        native.m,
        native.k,
        native.max_common,
        native.max_rare,
        native.start_exp,
    )


def _validate_buffers(buffers: list[Tensor], device: torch.device) -> None:
    if len(buffers) != 7 or any(
        tensor.device != device
        or tensor.dtype != torch.uint8
        or not tensor.is_contiguous()
        for tensor in buffers
    ):
        raise ValueError(
            "All seven ZipServ byte buffers must be contiguous on the input GPU"
        )


@torch.library.custom_op("zipserv_vllm::linear", mutates_args=(), device_types="cuda")
def _linear(
    x: Tensor,
    buffers: list[Tensor],
    m: int,
    k: int,
    max_common: int,
    max_rare: int,
    start_exp: int,
    split_k: int,
) -> Tensor:
    if m <= 0 or k <= 0 or m % 64 or k % 64:
        raise ValueError(
            "ZipServ linear requires positive, 64-aligned weight dimensions"
        )
    if x.dtype != torch.bfloat16 or x.ndim < 1 or x.shape[-1] != k:
        raise ValueError(
            "ZipServ linear requires BF16 activations with the matching input width"
        )
    if split_k < 1 or (k // 64) % split_k:
        raise ValueError("Split-K must divide the number of 64-column weight tiles")
    _validate_buffers(buffers, x.device)
    tokens = x.numel() // k
    output_shape = (*x.shape[:-1], m)
    if tokens == 0:
        return x.new_empty(output_shape)
    n = tokens
    if max(m * n * split_k, k * n, m * k) > 2**31 - 1 or (m // 64) * split_k > 65535:
        raise ValueError(
            "ZipServ output and Split-K workspace exceed native int32 indexing"
        )
    lib = _cuda_library()
    with torch.cuda.device(x.device):
        flat = x.reshape(tokens, k).contiguous()
        output = x.new_empty((n, m))
        # Native partial outputs have shape [split_k, n, m], not [n, m].
        workspace = x.new_empty((split_k, n, m)) if split_k > 1 else x.new_empty(0)
        stream = torch.cuda.current_stream(x.device).cuda_stream
        _check(
            lib.zipserv_mm(
                stream,
                *(buffer.data_ptr() for buffer in buffers),
                max_common,
                max_rare,
                start_exp,
                flat.data_ptr(),
                output.data_ptr(),
                m,
                n,
                k,
                workspace.data_ptr(),
                split_k,
            )
        )
        # This view uses a fresh internal allocation and aliases no operator input.
        # Cloning it would add a full output read/write to every inference call.
        return output[:tokens].reshape(output_shape)


@_linear.register_fake
def _linear_fake(
    x: Tensor,
    buffers: list[Tensor],
    m: int,
    k: int,
    max_common: int,
    max_rare: int,
    start_exp: int,
    split_k: int,
) -> Tensor:
    return x.new_empty((*x.shape[:-1], m))


def linear(x: Tensor, packed: PackedWeight, *, split_k: int = 1) -> Tensor:
    """Multiply with fused BF16 decoding; restored weights are exact, GEMM may round differently."""
    return _linear(
        x,
        list(packed.buffers),
        packed.m,
        packed.k,
        packed.max_common,
        packed.max_rare,
        packed.start_exp,
        split_k,
    )


def decompress(packed: PackedWeight) -> Tensor:
    """Decode on the current CUDA stream, primarily for bitwise verification."""
    device = packed.buffers[0].device
    if device.type != "cuda":
        raise ValueError("ZipServ decompression requires packed CUDA buffers")
    _validate_buffers(list(packed.buffers), device)
    lib = _cuda_library()
    with torch.cuda.device(device):
        output = torch.empty((packed.m, packed.k), dtype=torch.bfloat16, device=device)
        _check(
            lib.zipserv_decode(
                torch.cuda.current_stream(device).cuda_stream,
                *(buffer.data_ptr() for buffer in packed.buffers),
                packed.max_common,
                packed.max_rare,
                packed.start_exp,
                output.data_ptr(),
                packed.m,
                packed.k,
            )
        )
    return output
