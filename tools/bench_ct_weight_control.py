# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Measure an audited exact-CT arm with its production speculative decoder."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time

from bench_ct_bf16_model import LONG_PROMPT, PROMPTS, completion, serving_metrics
from bench_scale_cache_model import request, rpc


def validate_workers(workers, mode):
    if len(workers) != 2 or {row["rank"] for row in workers} != {0, 1}:
        raise RuntimeError("Both TP ranks must acknowledge the control")
    for row in workers:
        recipe = row["ct_recipe"]
        if (row["status"] != "verified" or row["mode"] != mode or not recipe["speculation"]
                or recipe["max_model_len"] != 500000
                or recipe["kv_cache_memory_bytes"] != 6 * 1024**3
                or not recipe["enable_prefix_caching"] or not recipe["physical_graph_token_counts"]):
            raise RuntimeError("The requested CT control did not load")
        if mode == "off" and (len(row["kda"]) != 68 or len(row["raw_scales"]) != 84
                              or row["optional_head_twins"] or row["mhc_bf16_twins"]):
            raise RuntimeError("An optional weight representation remains enabled")


def audit_stability(before, after, mode):
    validate_workers(before, mode)
    validate_workers(after, mode)
    left = {row["rank"]: row for row in before}
    for row in after:
        old = left[row["rank"]]
        keys = ("kda", "raw_scales", "embedding", "head") if mode == "off" else ()
        if any(old[key] != row[key] for key in keys):
            raise RuntimeError("Resident source bytes changed during serving")
        for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens",
                    "kv_cache_memory_bytes", "physical_graph_token_counts", "overlay_source_hashes"):
            if old["ct_recipe"][key] != row["ct_recipe"][key]:
                raise RuntimeError("CT configuration changed during serving: " + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18797")
    parser.add_argument("--mode", choices=("off", "ct"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.report.exists() or not 3 <= args.repeats <= 20:
        parser.error("Require a fresh report and 3..20 repetitions")
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference and reference["mode"] == args.mode:
        parser.error("The comparison reference must be the other arm")
    result = {"status": "running", "mode": args.mode,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "exact_CT_runtime_optional_weight_representations",
              "source_hashes": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("bench_ct_weight_control.py", "bench_ct_bf16_model.py",
                                             "bench_scale_cache_model.py")},
              "prompt_texts": [*PROMPTS, LONG_PROMPT], "requests": [], "cohorts": [],
              "limits": ["Checkpoint FP4/FP8 quantization and FP8 KV cache remain enabled.",
                         "Speculation, prefix caching and CT serving settings remain enabled.",
                         "The CT arm includes lossy KDA conversion; this is a whole-recipe comparison.",
                         "Short serving test; it does not establish GSM8K quality or 500K workload speed.",
                         "Streaming decode rates exclude the first speculative token burst.",
                         "Live CUDA allocations exclude driver, NCCL and allocator reservation."]}

    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(".incomplete")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.report)

    save()
    try:
        result["initial_workers"] = rpc(args.base, "ct_weight_control_report", [], 2)
        validate_workers(result["initial_workers"], args.mode)
        rendered = [request(args.base, "/tokenize", {"model": "glm-5.3-flash",
                    "messages": [{"role": "user", "content": prompt}], "add_generation_prompt": True})
                    for prompt in result["prompt_texts"]]
        prompts = [row["tokens"] for row in rendered]
        if any(not p or len(p) > 6919 or any(type(t) is not int for t in p) for p in prompts):
            raise RuntimeError("Invalid or oversized template-rendered prompt")
        result["rendered_prompts"] = rendered
        if reference and prompts != [row["tokens"] for row in reference["rendered_prompts"]]:
            raise RuntimeError("The matched arms rendered different prompts")
        result["serving_metrics_before"] = serving_metrics(args.base)
        result["warmup_requests"] = [completion(args.base, prompt, f"ct-weight-warm-{i}")
                                      for i, prompt in enumerate(prompts)]

        def cohort(salt):
            barrier = threading.Barrier(4)
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(completion, args.base, prompts[i % 2], f"{salt}-{i}", barrier)
                           for i in range(4)]
                rows = [future.result() for future in futures]
            return {"elapsed_seconds": time.perf_counter() - started, "requests": rows}

        result["warmup_cohorts"] = [cohort(f"ct-weight-warm-c4-{i}") for i in range(2)]
        save()
        for repeat in range(args.repeats):
            for workload, index in (("short_cold", 0), ("short_warm", 1), ("long_cold", 2)):
                salt = (f"ct-weight-warm-{index}" if workload == "short_warm"
                        else f"ct-weight-{workload}-{repeat}")
                row = completion(args.base, prompts[index], salt)
                row.update(workload=workload, repeat=repeat, prompt_index=index)
                result["requests"].append(row)
                save()
        for repeat in range(3):
            result["cohorts"].append({"repeat": repeat, **cohort(f"ct-weight-c4-{repeat}")})
            save()
        result["serving_metrics_after"] = serving_metrics(args.base)
        result["final_workers"] = rpc(args.base, "ct_weight_control_report", [], 2)
        audit_stability(result["initial_workers"], result["final_workers"], args.mode)
        metrics = ("elapsed_seconds", "ttft_seconds", "decode_tokens_per_second")
        result["summary"] = {
            workload: {metric: statistics.median(row[metric] for row in result["requests"]
                                                if row["workload"] == workload) for metric in metrics}
            for workload in ("short_cold", "short_warm", "long_cold")}
        result["summary"]["c4"] = {"aggregate_tokens_per_second": statistics.median(
            256 / row["elapsed_seconds"] for row in result["cohorts"])}
        result["summary"]["live_cuda_bytes"] = sum(row["cuda_allocated_bytes"] for row in result["final_workers"])
        result["distinct_outputs_by_workload"] = {
            key: len({tuple(row["token_ids"]) for row in result["requests"] if row["workload"] == key})
            for key in ("short_cold", "short_warm", "long_cold")}
        if reference:
            paired = []
            for raw, candidate in zip(reference["requests"], result["requests"], strict=True):
                if any(raw[key] != candidate[key] for key in ("workload", "repeat", "cache_salt", "prompt_tokens")):
                    raise RuntimeError("Requests do not match across arms")
                paired.append({"workload": raw["workload"], "repeat": raw["repeat"],
                               "token_ids_equal": raw["token_ids"] == candidate["token_ids"]})
            result["paired_outputs"] = paired
        result["status"] = "measured"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps({"status": result["status"], "mode": args.mode, "summary": result["summary"]}), flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
