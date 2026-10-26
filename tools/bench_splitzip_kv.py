# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Qualify a separately supplied upstream SplitZip BF16 codec on captured KV.

Includes the public Python API's allocations and host synchronisation. This is
not the paper's kernel-only number, a network test, or fused attention timing.
The upstream source and its license stay outside this project's source tree.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from screen_kv_pages import full_pages


def load_phase(root: Path, phase: str) -> torch.Tensor:
    manifest = json.loads((root / "manifest.json").read_text())
    parts = []
    for run in manifest["runs"]:
        if run["request"]["phase"] != phase:
            continue
        request = json.loads((root / f"{run['request']['id']}.json").read_text())
        for record in request["records"]:
            with np.load(root / record["file"], allow_pickle=False) as arrays:
                for mask in full_pages(arrays, record):
                    for kind in ("key", "value"):
                        parts.append(arrays[kind][mask].reshape(-1))
    data = np.concatenate(parts).astype(np.int16, copy=False)
    return torch.from_numpy(data).view(torch.bfloat16).cuda()


def measure(call):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall = time.perf_counter()
    start.record()
    result = call()
    end.record()
    end.synchronize()
    elapsed = time.perf_counter() - wall
    return result, {"wall_ms": elapsed * 1000, "gpu_interval_ms": start.elapsed_time(end),
                    "peak_added_bytes": torch.cuda.max_memory_allocated() - before}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    import sys
    spec = importlib.util.spec_from_file_location("splitzip_reference", args.reference / "codec_gpu.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    codec = module.ChunkLocalSplitZipGPU(device="cuda", chunk_size=1024)
    calibration = load_phase(args.capture, "calibration")
    coverage = codec.calibrate(calibration)
    del calibration
    # All bit patterns, signed zero/NaN payloads, odd sizes and chunk tails.
    checked = []
    for size in (1, 3, 1023, 1025, 65536):
        sample = torch.arange(size, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        encoded = codec.encode(sample)
        decoded = codec.decode(encoded)
        if not torch.equal(sample.view(torch.int16), decoded.view(torch.int16)):
            raise AssertionError(f"Upstream bit-exact test failed, size {size}")
        checked.append(size)
    del sample, encoded, decoded
    sample = load_phase(args.capture, "evaluation")
    results = []
    for size in (32 << 10, 2 << 20, 16 << 20):
        if sample.numel() * 2 < size:
            continue
        raw = sample[:size // 2].clone()
        encoded = codec.encode(raw)
        decoded = codec.decode(encoded)
        assert torch.equal(raw.view(torch.int16), decoded.view(torch.int16))
        transmitted = encoded.compressed_bytes + codec.dec_lut.numel() + 32
        construction = sum(getattr(encoded, key).numel() * getattr(encoded, key).element_size()
                           for key in ("counts", "starts"))
        timings = {name: [] for name in ("raw_copy", "encode", "decode", "encode_decode")}
        actions = {"raw_copy": lambda: raw.clone(), "encode": lambda: codec.encode(raw),
                   "decode": lambda: codec.decode(encoded),
                   "encode_decode": lambda: codec.decode(codec.encode(raw))}
        for action in actions.values():
            for _ in range(3):
                warm = action()
                torch.cuda.synchronize()
                del warm
        names = list(actions)
        for i in range(args.rounds):
            order = names[i % len(names):] + names[:i % len(names)]
            for name in order:
                value, timing = measure(actions[name])
                timings[name].append(timing)
                del value
        medians = {name: {metric: statistics.median(t[metric] for t in samples)
                          for metric in samples[0]} for name, samples in timings.items()}
        # A diagnostic serial bound, NOT an observed network improvement.
        serial_codec_s = medians["encode_decode"]["wall_ms"] / 1000
        bound = max(0, size - transmitted) / serial_codec_s / 1e9
        results.append({"raw_bytes": size, "wire_bytes_with_table_and_header": transmitted,
                        "construction_scratch_bytes": construction, "saved_pct": 100 * (1 - transmitted / size),
                        "serial_break_even_link_GBps_estimate": bound,
                        "medians": medians, "samples": timings, "bit_exact": True})
        del raw, encoded, decoded
        gc.collect()
    report = {"schema": 1, "upstream_commit": (args.reference / "revision.txt").read_text().strip(),
              "upstream_source_sha256": hashlib.sha256((args.reference / "codec_gpu.py").read_bytes()).hexdigest(),
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "calibration_top16_coverage": coverage, "bit_pattern_test_sizes": checked,
              "scope": "Public allocating Python API, warmed paired rotating runs; no network or attention kernel",
              "cases": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "cases": [{k: v for k, v in row.items() if k != "samples"} for row in results]}, indent=2))


if __name__ == "__main__":
    main()
