# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Compare FP8 off/on with unchanged BF16/FP4 compression and serving settings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from compare_mixed_model import read, require


def compare(off_root: Path, on_root: Path) -> dict:
    off, on = (read(root / "concurrent-fixed.json") for root in (off_root, on_root))
    require(off["status"] == on["status"] == "passed", "Both concurrent benchmarks must pass")
    require((off["fp8_weights"], on["fp8_weights"]) == ("off", "on"), "FP8 arms are reversed or identical")
    for key in ("model", "concurrency", "tokens_per_request", "repeats", "prompts", "source_sha256"):
        require(off[key] == on[key], f"Concurrent workload differs: {key}")
    require([p["tokens"] for p in off["rendered_prompts"]] ==
            [p["tokens"] for p in on["rendered_prompts"]], "Rendered prompt tokens differ")
    launches = [arm["server_receipt"] for arm in (off, on)]
    for key in ("image", "command", "package_source_sha256", "library_sha256", "boot_sha256"):
        require(launches[0][key] == launches[1][key], f"Serving configuration/source differs: {key}")
    environments = [{k: v for k, v in launch["environment"].items() if k != "ZIPSERV_MIXED_FP8"}
                    for launch in launches]
    require(environments[0] == environments[1], "Environment differs beyond the FP8 mode")
    mounts = [[m for m in launch["mounts"] if m["destination"] in ("/model", "/audit")]
              for launch in launches]
    require(mounts[0] == mounts[1], "Checkpoint or model fixture mounts differ")
    require(bool(off["batches"]) and len(off["batches"]) == off["repeats"], "Missing measured batches")
    pairs, token_matches = [], 0
    for raw, packed in zip(off["batches"], on["batches"], strict=True):
        require(raw["status"] == packed["status"] == "passed", "A measured batch failed")
        require((raw["repeat"], raw["completion_tokens"]) ==
                (packed["repeat"], packed["completion_tokens"]), "Measured batches differ")
        pairs.append(100 * (packed["completion_tokens_per_second"] / raw["completion_tokens_per_second"] - 1))
        for a, b in zip(raw["requests"], packed["requests"], strict=True):
            require((a["prompt_index"], a["usage"]) == (b["prompt_index"], b["usage"]),
                    "Per-request fixed workloads differ")
            token_matches += a["token_ids"] == b["token_ids"]
    workers = []
    for rank, (raw, packed, raw_scales, packed_scales) in enumerate(zip(
            off["final_workers"]["mixed"], on["final_workers"]["mixed"],
            off["final_workers"]["scales"], on["final_workers"]["scales"], strict=True)):
        require(raw["raw_owners_released"] and packed["raw_owners_released"], "An original weight owner remains")
        require(raw["per_type"]["fp8"]["compressed_layers"] == 0 and
                packed["per_type"]["fp8"]["compressed_layers"] > 0, "FP8 storage did not switch")
        def bf16_sources(worker):
            return [(r["name"], r["source_sha256"], r["raw_bytes"], r["retained_bytes"])
                    for r in worker["rows"] if r["compressed"] and r["kind"] == "bf16"]
        require(bool(bf16_sources(raw)) and bf16_sources(raw) == bf16_sources(packed),
                "BF16 compressed sources differ")
        def scale_sources(worker):
            return [(layer["ordinal"], s["name"], s["source_sha256"], s["retained_bytes"])
                    for layer in worker["layers"] for s in layer["scales"]]
        require(scale_sources(raw_scales) == scale_sources(packed_scales), "FP4 scale sources differ")
        for weights, scales in ((raw, raw_scales), (packed, packed_scales)):
            require(all(r["gpu_bytes_exact"] for r in weights["rows"] if r["compressed"]),
                    "Weight source verification failed")
            require(all(r["gpu_bytes_exact"] and r["raw_owner_released"] for r in scales["ownership"]),
                    "FP4 source verification failed")
        workers.append({"rank": rank,
            "fp8_on_additional_cuda_saving_bytes": raw_scales["cuda_allocated_bytes"] - packed_scales["cuda_allocated_bytes"],
            "fp8_on_extra_scratch_bytes": packed["fp8_shared_scratch_bytes"] - raw["fp8_shared_scratch_bytes"]})
    sequential = [read(root / "benchmark.json") for root in (off_root, on_root)]
    for key in ("status", "model", "source_sha256", "prompts"):
        require(sequential[0][key] == sequential[1][key], f"Sequential benchmark differs: {key}")
    require(sequential[0]["status"] == "passed", "Sequential benchmark failed")
    for a, b in zip(sequential[0]["requests"], sequential[1]["requests"], strict=True):
        require((a["repeat"], a["prompt_index"], a["usage"]) == (b["repeat"], b["prompt_index"], b["usage"]),
                "Sequential request workloads differ")
    speeds = [report["summary"]["median_decode_tokens_per_second"] for report in sequential]
    return {"status": "measured", "fp8_off": str(off_root), "fp8_on": str(on_root),
        "concurrent_fp8_off": off["summary"], "concurrent_fp8_on": on["summary"],
        "concurrent_on_throughput_change_percent": 100 * (
            on["summary"]["median_batch_completion_tokens_per_second"] /
            off["summary"]["median_batch_completion_tokens_per_second"] - 1),
        "paired_batch_on_throughput_change_percent": {"median": statistics.median(pairs),
                                                       "range": [min(pairs), max(pairs)]},
        "equal_output_pairs": token_matches,
        "sequential_fp8_off_decode_tokens_per_second": speeds[0],
        "sequential_fp8_on_decode_tokens_per_second": speeds[1],
        "sequential_on_throughput_change_percent": 100 * (speeds[1] / speeds[0] - 1),
        "workers": workers,
        "fp8_on_additional_cuda_saving_bytes": sum(w["fp8_on_additional_cuda_saving_bytes"] for w in workers),
        "limits": ["Both arms retain BF16 and FP4 scale compression; this isolates FP8 storage.",
                   "One server start per arm and three concurrent batches; small timing differences can be noise.",
                   "Four fixed prompts and output lengths; generated tokens and scheduling may differ.",
                   "Positive throughput changes favor FP8 on. This is not a quality evaluation."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp8-off", type=Path, required=True)
    parser.add_argument("--fp8-on", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists(), "Preserve existing comparison evidence")
    result = compare(args.fp8_off, args.fp8_on)
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
