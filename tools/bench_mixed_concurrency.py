# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Measure four simultaneous fixed-length requests for a mixed-storage ablation."""
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

from bench_mxfp4_model import PROMPTS
from bench_scale_cache_model import completion, request, rpc


def batch(pool, base: str, model: str, prompts: list, tokens: int) -> dict:
    gate = threading.Barrier(len(prompts) + 1, timeout=30)

    def run(index, prompt):
        gate.wait()
        started = time.perf_counter()
        try:
            result = completion(base, model, prompt, tokens)
            result["status"] = "passed"
        except Exception as error:
            result = {"status": "failed", "error": repr(error)}
        return {"prompt_index": index, "started_clock": started, **result}

    futures = [pool.submit(run, i, prompt) for i, prompt in enumerate(prompts)]
    started = time.perf_counter()
    gate.wait()
    rows = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    for row in rows:
        row["start_offset_seconds"] = row.pop("started_clock") - started
    passed = all(row["status"] == "passed" for row in rows)
    return {"status": "passed" if passed else "failed", "elapsed_seconds": elapsed,
            "completion_tokens": len(prompts) * tokens if passed else None,
            "completion_tokens_per_second": len(prompts) * tokens / elapsed if passed else None,
            "requests": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18796")
    parser.add_argument("--model", default="zipserv-mxfp4-test")
    parser.add_argument("--fp8-weights", choices=("off", "on"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--server-receipt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or not 2 <= args.tokens <= 512 or not 1 <= args.repeats <= 20 or args.workers < 1:
        parser.error("Require a fresh report and bounded positive workload")
    sources = [Path(__file__), Path(__file__).with_name("bench_scale_cache_model.py"),
               Path(__file__).with_name("bench_mxfp4_model.py")]
    result = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "model": args.model, "fp8_weights": args.fp8_weights, "concurrency": len(PROMPTS),
              "tokens_per_request": args.tokens, "repeats": args.repeats, "prompts": PROMPTS,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "server_receipt": json.loads(args.server_receipt.read_text()), "batches": [],
              "limits": ["BF16 weights and FP4 scales stay compressed in both arms.",
                         "Fixed prompt tokens and output lengths; generated token equality is reported separately.",
                         "Client timings include prefill, decode, scheduling and transport.",
                         "Three sequential batches do not establish production throughput or model quality."]}

    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(".incomplete")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.report)

    def audit():
        mixed = rpc(args.base, "zipserv_mixed_report", [True], args.workers)
        scales = rpc(args.base, "zipserv_mxfp4_report", [True], args.workers)
        for weights, scale in zip(mixed, scales, strict=True):
            fp8_count = weights["per_type"]["fp8"]["compressed_layers"]
            if (weights["status"] != "prepared" or not weights["raw_owners_released"]
                    or not weights["per_type"]["bf16"]["compressed_layers"]
                    or bool(fp8_count) != (args.fp8_weights == "on")):
                raise RuntimeError("Requested mixed compression was not active on every worker")
            if (scale["mode"] != "on" or not scale["compressed_scale_groups"]
                    or any(not row["raw_owner_released"] or not row["gpu_bytes_exact"]
                           for row in scale["ownership"])):
                raise RuntimeError("FP4 scale source verification or ownership failed")
        return {"mixed": mixed, "scales": scales}

    save()
    try:
        result["initial_workers"] = audit()
        rendered = [request(args.base, "/tokenize", {"model": args.model,
                    "messages": [{"role": "user", "content": prompt}], "add_generation_prompt": True})
                    for prompt in PROMPTS]
        prompts = [item["tokens"] for item in rendered]
        if any(not p or not all(type(t) is int and t >= 0 for t in p) for p in prompts):
            raise RuntimeError("Invalid rendered prompt tokens")
        result["rendered_prompts"] = rendered
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            result["warmup"] = batch(pool, args.base, args.model, prompts, args.tokens)
            save()
            if result["warmup"]["status"] != "passed":
                raise RuntimeError("Concurrent warmup failed; partial responses retained")
            for repeat in range(args.repeats):
                measured = batch(pool, args.base, args.model, prompts, args.tokens)
                measured["repeat"] = repeat
                result["batches"].append(measured)
                save()
                if measured["status"] != "passed":
                    raise RuntimeError("Measured batch failed; partial responses retained")
        result["final_workers"] = audit()
        rows = [row for measured in result["batches"] for row in measured["requests"]]
        result["summary"] = {
            "completed_requests": len(rows),
            "median_batch_completion_tokens_per_second": statistics.median(
                measured["completion_tokens_per_second"] for measured in result["batches"]),
            "batch_completion_tokens_per_second_range": [
                min(b["completion_tokens_per_second"] for b in result["batches"]),
                max(b["completion_tokens_per_second"] for b in result["batches"])],
            "median_request_ttft_seconds": statistics.median(row["ttft_seconds"] for row in rows),
            "median_request_elapsed_seconds": statistics.median(row["elapsed_seconds"] for row in rows),
            "repeatable_requests": sum(row["token_ids"] == result["warmup"]["requests"][row["prompt_index"]]["token_ids"]
                                       for row in rows),
        }
        result.update(status="passed", finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(result["summary"], indent=2))
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
