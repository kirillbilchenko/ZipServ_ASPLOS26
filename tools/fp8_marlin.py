"""Explicit, experimental native-Marlin binding; no vLLM hooks or default changes.

Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
Use already repacked vLLM 0.32 BF16/MXFP8 weights and scales. A bound call owns
its buffers; retain it until all CUDA work completes. Calls sharing a bound
workspace must be serialized, including graph replays on different streams.
"""
from __future__ import annotations

import ctypes
import copy
from pathlib import Path
import time

import numpy as np
import torch

import fp8_range
import fp8_async
import fp8_byteplane

CODEC_IDS = {"range16": 1, "async": 2, "byteplane": 3}


def load_library(path: Path, *, compressed: bool, codec="range16"):
    if codec not in CODEC_IDS:
        raise ValueError("Choose range16, async or byteplane codec")
    library = ctypes.CDLL(str(path.resolve()))
    for name in ("zipserv_marlin_fp8_abi", "zipserv_marlin_fp8_compressed"):
        getattr(library, name).argtypes = []
        getattr(library, name).restype = ctypes.c_int
    abi = library.zipserv_marlin_fp8_abi()
    if abi not in (1, 2) or library.zipserv_marlin_fp8_compressed() != int(compressed):
        raise ValueError("Wrong Marlin experiment library ABI or variant")
    compact_only = 0
    if hasattr(library, "zipserv_marlin_fp8_compact_only"):
        library.zipserv_marlin_fp8_compact_only.argtypes = []
        library.zipserv_marlin_fp8_compact_only.restype = ctypes.c_int
        compact_only = library.zipserv_marlin_fp8_compact_only()
    if (compact_only not in (0, 1) or (abi == 2) != bool(compact_only)
            or (compact_only and (not compressed or codec == "range16"))):
        raise ValueError("Invalid compact-only ABI contract")
    library.zipserv_compact_only = bool(compact_only)
    expected = CODEC_IDS[codec] if compressed else 0
    if hasattr(library, "zipserv_marlin_fp8_codec"):
        library.zipserv_marlin_fp8_codec.argtypes = []
        library.zipserv_marlin_fp8_codec.restype = ctypes.c_int
        if library.zipserv_marlin_fp8_codec() != expected:
            raise ValueError("Library uses a different compressed format")
    elif codec in ("async", "byteplane"):
        raise ValueError("Async codec requires a format-aware library")
    library.zipserv_codec_id = expected
    library.zipserv_marlin_fp8_last_error.argtypes = []
    library.zipserv_marlin_fp8_last_error.restype = ctypes.c_char_p
    library.zipserv_marlin_fp8_run.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    library.zipserv_marlin_fp8_run.restype = ctypes.c_int
    if compressed:
        library.zipserv_marlin_fp8_decode.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_void_p]
        library.zipserv_marlin_fp8_decode.restype = ctypes.c_int
    return library


def check_representation(library, compressed, codec, accounting):
    """Check the accepted domain before upload or rebinding immutable weights."""
    if library.zipserv_marlin_fp8_compressed() != int(compressed):
        raise ValueError("Library does not match selected weight representation")
    expected = CODEC_IDS[codec] if compressed else 0
    if getattr(library, "zipserv_codec_id", None) != expected:
        raise ValueError("Bound library format does not match weight encoding")
    if getattr(library, "zipserv_compact_only", False) and accounting.get("raw_fallback_tiles") != 0:
        raise ValueError("Compact-only kernel requires zero raw fallback tiles")


def device_buffer(tensor, device, dtype, shape):
    if (tensor.device != device or device.type != "cuda" or tensor.dtype != dtype
            or tuple(tensor.shape) != tuple(shape) or not tensor.is_contiguous()
            or tensor.data_ptr() % 16):
        raise ValueError(f"Require aligned contiguous {dtype} CUDA buffer of shape {shape}")


class Weights:
    """Keep native packed bytes or encode ZFR1/ZFA1/ZFB1 according to explicit flags.

    on: force compression for measurements, including tensors that grow.
    off: unchanged bytes in the isolated stock Marlin build.
    auto: retain raw bytes unless the complete container clears min_saving_pct.
    This is a storage filter, not a latency qualification.
    This class does not retain a decoded GPU copy when using compression.
    """

    def __init__(self, weight, scales, n, k, *, raw_library, range_library,
                 compression="off", min_saving_pct=1.0, codec="range16", force_raw=False,
                 async_alignment=16):
        if codec not in CODEC_IDS or (force_raw and codec not in ("async", "byteplane")):
            raise ValueError("Choose range16/async/byteplane; force_raw is an async-format control only")
        if async_alignment not in (16, 32) or (codec == "range16" and async_alignment != 16):
            raise ValueError("Alignment 16/32 applies to the async codec only")
        if compression not in ("off", "on", "auto") or not 0 <= min_saving_pct <= 100:
            raise ValueError("Choose compression off/on/auto and a saving threshold in [0,100]")
        if (n < 64 or k < 64 or n % 64 or k % 64 or (n % 128 and k % 128)
                or n * k > 2**31 - 1):
            raise ValueError("Require a native Marlin tile family and N*K <= INT32_MAX")
        self.n, self.k, self.device = n, k, weight.device
        device_buffer(weight, self.device, torch.int32, (k // 16, n * 4))
        device_buffer(scales, self.device, torch.float8_e8m0fnu, (k // 32, n))
        self.scales = scales
        self.accounting = {"raw_bytes": n * k, "selected": "raw", "requested": compression,
                           "retained_weight_bytes": n * k, "decoded_weight_scratch_bytes": 0}
        self.compressed = False
        if compression != "off":
            raw = weight.view(torch.uint8).cpu().numpy().tobytes()
            start = time.perf_counter()
            encoder = {"async": fp8_async, "byteplane": fp8_byteplane, "range16": fp8_range}[codec]
            encoded, sizes = (encoder.encode(raw, force_raw=force_raw, alignment=async_alignment) if codec in ("async", "byteplane")
                              else encoder.encode(raw, 16, 2048))
            encode_seconds = time.perf_counter() - start
            start = time.perf_counter()
            if encoder.decode(encoded) != raw:
                raise RuntimeError("Independent CPU reconstruction changed native Marlin bytes")
            self.accounting.update(sizes, cpu_encode_seconds=encode_seconds,
                                   cpu_verify_seconds=time.perf_counter() - start)
            self.compressed = compression == "on" or sizes["saving_pct"] >= min_saving_pct
            if (self.compressed and compression == "auto"
                    and getattr(range_library, "zipserv_compact_only", False)
                    and sizes["raw_fallback_tiles"]):
                self.compressed = False
                self.accounting["fallback_reason"] = "compact-only kernel cannot consume raw tiles"
            if self.compressed:
                check_representation(range_library, True, codec, self.accounting)
                self.data = torch.from_numpy(np.frombuffer(encoded, dtype=np.uint8).copy()).to(self.device)
                self.accounting.update(selected=codec, retained_weight_bytes=len(encoded))
        if not self.compressed:
            self.data = weight
        self.library = range_library if self.compressed else raw_library
        check_representation(self.library, self.compressed, codec, self.accounting)
        self.accounting["kernel_tile_policy"] = ("compact-only" if
            getattr(self.library, "zipserv_compact_only", False) else "mixed" if self.compressed else "raw")
        self.accounting["retained_saving_pct"] = 100 * (1 - self.data.nbytes / (n * k))

    def with_library(self, library):
        """Share immutable bytes with a compatible kernel; bound workspaces stay separate."""
        check_representation(library, self.compressed, self.accounting["selected"], self.accounting)
        other = copy.copy(self)
        other.library = library
        other.accounting = {**self.accounting, "shared_immutable_payload": True,
            "kernel_tile_policy": "compact-only" if getattr(library, "zipserv_compact_only", False)
                                  else "mixed" if self.compressed else "raw"}
        return other

    def bind_decode(self, output):
        """Correctness/debug operation only; GEMM does not call a decode kernel."""
        if not self.compressed:
            raise ValueError("Decode requires a compressed owner")
        device_buffer(output, self.device, torch.uint8, (self.n * self.k,))
        def run():
            if torch.cuda.current_device() != self.device.index:
                raise ValueError("Current CUDA device changed")
            status = self.library.zipserv_marlin_fp8_decode(
                self.data.data_ptr(), output.data_ptr(), output.numel(),
                torch.cuda.current_stream().cuda_stream)
            if status:
                raise RuntimeError(f"Marlin shared-copy decode launch failed: {status}")
        return run

    def bind(self, x, *, atomic_add=False, output=None):
        return Bound(self, x, atomic_add=atomic_add, output=output)


class Bound:
    """Preallocated GEMM output and Marlin reduction workspace, safe for capture."""

    def __init__(self, weights, x, *, atomic_add=False, output=None):
        self.weights, self.x, self.atomic_add = weights, x, bool(atomic_add)
        self.m = x.shape[0]
        if not 1 <= self.m <= 128 or torch.cuda.current_device() != weights.device.index:
            raise ValueError("Require M in 1..128 and the current CUDA device")
        device_buffer(x, weights.device, torch.bfloat16, (self.m, weights.k))
        self.output = (torch.empty((self.m, weights.n), device=weights.device, dtype=x.dtype)
                       if output is None else output)
        device_buffer(self.output, weights.device, torch.bfloat16, (self.m, weights.n))
        if any(self.output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
               for t in (x, weights.data, weights.scales)):
            raise ValueError("Output must not alias an input buffer")
        sms = torch.cuda.get_device_properties(weights.device).multi_processor_count
        # Exactly the pinned vLLM FP32 reduction workspace sizing policy.
        self.temporary = torch.empty(sms * min((self.m + 15) // 16 * 16, 64) * 256,
                                     dtype=torch.float32, device=weights.device)
        self.locks = torch.zeros(sms, dtype=torch.int32, device=weights.device)

    @property
    def workspace_bytes(self):
        return self.temporary.nbytes + self.locks.nbytes

    def __call__(self):
        weights = self.weights
        if torch.cuda.current_device() != weights.device.index:
            raise ValueError("Current CUDA device changed")
        status = weights.library.zipserv_marlin_fp8_run(
            self.x.data_ptr(), weights.data.data_ptr(), weights.scales.data_ptr(),
            self.output.data_ptr(), self.temporary.data_ptr(), self.locks.data_ptr(),
            self.m, weights.n, weights.k, weights.device.index, int(self.atomic_add),
            torch.cuda.current_stream().cuda_stream)
        if status:
            message = weights.library.zipserv_marlin_fp8_last_error().decode(errors="replace")
            raise RuntimeError(f"Marlin FP8 launch failed: {message}")
        return self.output
