"""Diagnose exact-scale dispatch changes without relaxing the stock benchmark gate.

Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
Checks compression against the equally tuned raw kernel, separately recording
any raw-vs-stock arithmetic differences. Does not measure latency or serving.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path

import torch

from bench_fp8_marlin import checks, equal_bytes, prepare, stock
from bench_fp8_palette import digest, source_fixture
from fp8_marlin import Weights, load_library


def comparison(actual, reference):
    torch.cuda.synchronize()
    if not bool(torch.isfinite(actual).all() & torch.isfinite(reference).all()):
        raise RuntimeError("Nonfinite diagnostic output")
    difference = actual.float() - reference.float()
    changed = (actual.view(torch.uint8).reshape(-1, 2)
               != reference.view(torch.uint8).reshape(-1, 2)).any(dim=1)
    return {"byte_exact": not bool(changed.any()),
            "different_bf16_elements": int(changed.sum()), "elements": actual.numel(),
            "maximum_absolute_difference": float(difference.abs().max()),
            "relative_l2": float(difference.norm() / reference.float().norm().clamp_min(1e-30))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--raw-library", type=Path, required=True)
    p.add_argument("--range-library", type=Path, required=True)
    p.add_argument("--async-library", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    if args.report.exists():
        p.error("Preserve existing diagnostic evidence")
    import vllm
    if vllm.__version__.split("+")[0] != "0.32.0":
        raise RuntimeError("Require pinned vLLM 0.32.0")
    pin = json.loads((Path(__file__).parent / "marlin_fp8_sources.json").read_text())
    package_root = Path(inspect.getfile(vllm)).parent.parent
    for relative, expected in pin["runtime_helpers_sha256"].items():
        if hashlib.sha256((package_root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Installed layout helper changed: {relative}")
    torch.cuda.set_per_process_memory_fraction(min(1.0, 1.5 * 2**30
        / torch.cuda.get_device_properties(0).total_memory))
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(181)
    raw_lib = load_library(args.raw_library, compressed=False)
    range_lib = load_library(args.range_library, compressed=True)
    async_lib = load_library(args.async_library, compressed=True, codec="async")
    # Reproduce the synthetic suite and RNG consumption of the original run.
    synthetic = checks(raw_lib, range_lib, async_lib, 16)
    result = {"scope": "dispatch diagnosis only; the original stock-exact gate is unchanged",
        "vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "synthetic_checks": len(synthetic),
        "libraries_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (args.raw_library, args.range_library, args.async_library)},
        "sources_sha256": {n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
            for n in ("check_fp8_dispatch.py", "bench_fp8_marlin.py", "bench_fp8_palette.py", "fp8_marlin.py")},
        "experiments": []}
    for layer_index, rank, family in ((0, 0, "gate_up"), (3, 0, "gate_up"), (44, 1, "down")):
        w, s, sources = source_fixture(args.source, layer_index, rank, family)
        n, k = w.shape
        layer = prepare(w, s)
        raw = Weights(layer.weight, layer.weight_scale, n, k, raw_library=raw_lib,
                      range_library=async_lib, compression="off")
        compressed = Weights(layer.weight, layer.weight_scale, n, k, raw_library=raw_lib,
                             range_library=async_lib, compression="on", codec="async")
        decoded = torch.empty(n * k, dtype=torch.uint8, device="cuda")
        compressed.bind_decode(decoded)()
        equal_bytes(decoded, layer.weight.view(torch.uint8).reshape(-1), "Exact encoded weight bytes")
        del decoded
        for m in (1, 8, 32, 128):
            x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * .125
            r, c = raw.bind(x), compressed.bind(x)
            reference = stock(x, layer, r.locks, n, k).clone()
            uncompressed, candidate = r(), c()
            equal_bytes(candidate, uncompressed, "Compressed vs equally tuned raw")
            result["experiments"].append({"case": [layer_index, rank, family],
                "shape_mnk": [m, n, k], "source": sources, "input_sha256": digest(x),
                "weight_sha256": digest(layer.weight), "scale_sha256": digest(layer.weight_scale),
                "exact_gpu_weight_bytes": True, "atomic_add": False,
                "raw_vs_stock": comparison(uncompressed, reference),
                "compressed_vs_matched_raw": comparison(candidate, uncompressed)})
            del x, r, c, reference, uncompressed, candidate
        del layer, raw, compressed, w, s
        gc.collect()
        torch.cuda.empty_cache()
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"report": str(args.report), "cases": len(result["experiments"]),
                      "stock_differences": [e for e in result["experiments"] if not e["raw_vs_stock"]["byte_exact"]]}))


if __name__ == "__main__":
    main()
