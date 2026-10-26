# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Compare completed, matched mixed-compression serving runs and their inventories."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def read(path: Path):
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def serving_command(receipt: dict) -> list[str]:
    command = list(receipt["command"])
    i = command.index("--worker-extension-cls")
    del command[i:i + 2]
    return command


def evaluation_serving_metrics(root: Path) -> dict:
    summary = read(root / "summary.json")
    rows = [json.loads(line) for line in (root / "responses.jsonl").read_text().splitlines()]
    require(summary["status"] == "complete" and len(rows) == 1319,
            "Concurrent evaluation timing requires a complete run")
    usage = [row.get("response", {}).get("usage", {}) for row in rows]
    completion_tokens = sum(u.get("completion_tokens", 0) for u in usage)
    elapsed = summary["elapsed_seconds"]
    times = sorted(row["elapsed_seconds"] for row in rows)
    position = (len(times) - 1) * .95
    lower = int(position)
    return {
        "concurrency": read(root / "protocol.json")["concurrency"],
        "elapsed_seconds": elapsed,
        "api_errors": summary["api_errors"],
        "usage_records": sum(bool(u) for u in usage),
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / elapsed,
        "successful_requests_per_second": (len(rows) - summary["api_errors"]) / elapsed,
        "median_request_seconds": statistics.median(times),
        "p95_request_seconds": times[lower] + (times[lower + 1] - times[lower]) * (position - lower),
        "limits": "Natural answer lengths, routing and scheduling can differ; this is not fixed-output-length throughput.",
    }


def compare(control: Path, candidate: Path) -> dict:
    off, on = (read(p / "benchmark.json") for p in (control, candidate))
    launches = [read(p / "start-receipt.json") for p in (control, candidate)]
    require(off["status"] == on["status"] == "passed", "Both benchmarks must finish successfully")
    require(off["model"] == on["model"] and off["source_sha256"] == on["source_sha256"],
            "Model name or benchmark implementation differs")
    require(serving_command(launches[0]) == serving_command(launches[1]),
            "Serving arguments differ beyond the audit worker")
    require(launches[0]["image"] == launches[1]["image"], "Serving image differs")
    # Compression flags and the plugin list intentionally differ. All other
    # runtime environment settings must match, including collective transport.
    environments = [{k: v for k, v in launch["environment"].items()
                     if not k.startswith("ZIPSERV_") and k != "VLLM_PLUGINS"}
                    for launch in launches]
    require(environments[0] == environments[1], "Non-compression environment differs")
    model_mounts = [[m for m in launch["mounts"] if m["destination"] in ("/model", "/audit")]
                    for launch in launches]
    require(model_mounts[0] == model_mounts[1], "Checkpoint or model fixture mount differs")
    require(off["prompts"] == on["prompts"], "Prompt text differs")
    require([r["tokens"] for r in off["rendered_prompts"]]
            == [r["tokens"] for r in on["rendered_prompts"]], "Rendered input tokens differ")
    inventories = [read(p / "weight-inventory.json") for p in (control, candidate)]
    mixed = read(candidate / "mixed-after-benchmark.json")
    workers = list(zip(off["final_workers"], on["final_workers"],
                       inventories[0], inventories[1], mixed, strict=True))
    require(bool(workers), "No worker reports")
    per_type = {kind: Counter() for kind in ("fp4_scales", "bf16", "fp8")}
    fallbacks, ranks = Counter(), []
    for rank, (raw, packed, raw_inventory, packed_inventory, audit) in enumerate(workers):
        scale_keys = lambda w: [(layer["ordinal"], s["name"], s["shape"], s["source_sha256"])
                               for layer in w["layers"] for s in layer["scales"]]
        require(scale_keys(raw) == scale_keys(packed), "Checkpoint scale shards differ")
        require(audit["status"] == "prepared" and audit["raw_owners_released"],
                "Mixed preparation or source-owner release failed")
        baseline_weights = {module["prefix"]: t for module in raw_inventory["modules"]
                            for t in module["tensors"] if t["name"] == "weight"}
        rows = [r for r in audit["rows"] if r["compressed"]]
        require(all(r["gpu_bytes_exact"] for r in rows), "Live weight verification failed")
        require(all(sum(r["eager_or_capture_calls"].values()) > 0 for r in rows),
                "A compressed consumer was never executed")
        for row in rows:
            expected_dtype = "torch.bfloat16" if row["kind"] == "bf16" else "torch.int32"
            source = baseline_weights.get(row["name"], {})
            require(source.get("dtype") == expected_dtype,
                    "Compressed consumer does not match the baseline tensor type")
            require(source.get("shape") == row["shape"]
                    and source.get("logical_bytes") == row["raw_bytes"],
                    "Compressed consumer does not match the baseline tensor dimensions")
            per_type[row["kind"]].update(shards=1, raw_bytes=row["raw_bytes"],
                                         retained_bytes=row["retained_bytes"])
        for row in audit["rows"]:
            if not row["compressed"]:
                # Some unquantized methods own FP32 weights. Label them by
                # inventory dtype instead of incorrectly counting them as BF16.
                dtype = baseline_weights.get(row["name"], {}).get("dtype", "unknown")
                fallbacks[(dtype, row["fallback"])] += 1
        per_type["fp4_scales"].update(
            shards=packed["compressed_scale_groups"], raw_bytes=raw["scale_raw_bytes"],
            retained_bytes=packed["scale_retained_bytes"])
        ranks.append({"rank": rank,
            "control_cuda_allocated_bytes": raw["cuda_allocated_bytes"],
            "candidate_cuda_allocated_bytes": packed["cuda_allocated_bytes"],
            "net_cuda_saving_bytes": raw["cuda_allocated_bytes"] - packed["cuda_allocated_bytes"],
            "registered_storage_saving_bytes": raw_inventory["unique_cuda_storage_bytes"]
                - packed_inventory["unique_cuda_storage_bytes"],
            "fp4_scratch_bytes": packed["scratch_bytes"],
            "fp8_scratch_bytes": audit["fp8_shared_scratch_bytes"]})
    pairs = []
    for a, b in zip(off["requests"], on["requests"], strict=True):
        require((a["repeat"], a["prompt_index"], a["usage"])
                == (b["repeat"], b["prompt_index"], b["usage"]), "Request workload differs")
        pairs.append({"repeat": a["repeat"], "prompt_index": a["prompt_index"],
            "tokens_equal": a["token_ids"] == b["token_ids"],
            "decode_throughput_change_percent": 100 * (b["decode_tokens_per_second"]
                                                       / a["decode_tokens_per_second"] - 1),
            "ttft_change_percent": 100 * (b["ttft_seconds"] / a["ttft_seconds"] - 1)})
    require(bool(pairs), "No timing pairs")
    payloads = {kind: {**counts, "payload_saving_bytes": counts["raw_bytes"] - counts["retained_bytes"],
                      "saving_percent": 100 * (1 - counts["retained_bytes"] / counts["raw_bytes"])
                      if counts["raw_bytes"] else None}
                for kind, counts in per_type.items()}
    total_raw = sum(row["control_cuda_allocated_bytes"] for row in ranks)
    total_saved = sum(row["net_cuda_saving_bytes"] for row in ranks)
    result = {"status": "measured", "control": str(control), "candidate": str(candidate),
        "per_type_compressed_targets": payloads, "workers": ranks,
        "net_cuda_saving_bytes": total_saved,
        "net_cuda_saving_percent": 100 * total_saved / total_raw,
        "fallback_shards": [{"dtype": key[0], "reason": key[1], "count": count}
                            for key, count in sorted(fallbacks.items())],
        "control_speed": off["summary"], "candidate_speed": on["summary"],
        "paired_timing": {key: {"median": statistics.median(p[key] for p in pairs),
                                 "range": [min(p[key] for p in pairs), max(p[key] for p in pairs)]}
                          for key in ("decode_throughput_change_percent", "ttft_change_percent")},
        "equal_output_pairs": sum(p["tokens_equal"] for p in pairs), "timing_pairs": pairs,
        "limits": ["Per-type bytes cover compressed targets, not every tensor of that dtype.",
                   "FP4 values and FP8 scales are unchanged; FP4 savings are scale savings.",
                   "Net CUDA saving includes graph/workspace changes and allocator rounding.",
                   "Sequential short-request timings do not establish production throughput.",
                   "All-type savings are relative to this stock CT checkpoint recipe, not an older optimized CT deployment."]}
    summaries = [p / "gsm8k/summary.json" for p in (control, candidate)]
    if all(p.is_file() and read(p)["status"] == "complete" for p in summaries):
        from eval_ct_gsm8k import compare as compare_quality
        result["gsm8k"] = compare_quality(control / "gsm8k", candidate / "gsm8k")
        result["gsm8k"]["serving_metrics"] = {
            "control": evaluation_serving_metrics(control / "gsm8k"),
            "candidate": evaluation_serving_metrics(candidate / "gsm8k"),
        }
    else:
        result["gsm8k"] = {"status": "pending", "reason": "Two complete matched runs are required"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists(), "Preserve existing comparison evidence")
    result = compare(args.control, args.candidate)
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("timing_pairs", "fallback_shards")}, indent=2))


if __name__ == "__main__":
    main()
