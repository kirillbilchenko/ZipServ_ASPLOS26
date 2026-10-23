# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Compare the old copying handoff and shared buffers in fresh CPU processes."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
from zipserv_vllm import runtime


def copy_packed(weight: torch.Tensor) -> runtime.PackedWeight:
    """The pre-change Python handoff, using the same current native encoder."""
    native = runtime._NativePacked()
    lib = runtime._library()
    runtime._check(
        lib.zipserv_compress(weight.data_ptr(), *weight.shape, ctypes.byref(native))
    )
    try:
        buffers = []
        for pointer, length in zip(native.buffers, native.lengths, strict=True):
            buffer = torch.empty(length, dtype=torch.uint8)
            if length:
                ctypes.memmove(buffer.data_ptr(), pointer, length)
            buffers.append(buffer)
        return runtime.PackedWeight(
            tuple(buffers),
            native.m,
            native.k,
            native.max_common,
            native.max_rare,
            native.start_exp,
        )
    finally:
        lib.zipserv_free(ctypes.byref(native))


def peak_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("copy", "share"), required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--cols", type=int, default=8192)
    args = parser.parse_args()
    if args.rows <= 0 or args.cols <= 0 or args.rows % 64 or args.cols % 64:
        parser.error("rows and cols must be positive multiples of 64")
    if args.rows * args.cols > 2**31 - 4097:
        parser.error("matrix exceeds the native int32 indexing limit")
    os.environ["ZIPSERV_LIBRARY"] = str(args.library.resolve())
    runtime._library()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    # Fill BF16 directly so a temporary FP32 input does not inflate peak RSS.
    weight = torch.empty((args.rows, args.cols), dtype=torch.bfloat16).normal_()
    initial_peak = peak_mib()
    start = time.perf_counter()
    packed = copy_packed(weight) if args.mode == "copy" else runtime.compress(weight)
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "mode": args.mode,
                "torch": torch.__version__,
                "m": args.rows,
                "k": args.cols,
                "packing_and_handoff_ms": elapsed * 1000,
                "initial_peak_rss_mib": initial_peak,
                "peak_rss_mib": peak_mib(),
                "input_bytes": weight.numel() * weight.element_size(),
                "packed_bytes": packed.storage_bytes,
                "checksum": sum(
                    int(buffer[0]) + int(buffer[-1])
                    for buffer in packed.buffers
                    if buffer.numel()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
