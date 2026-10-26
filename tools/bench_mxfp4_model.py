# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Audit all MXFP4 layers and measure fixed-workload requests on a private vLLM server."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from bench_scale_cache_model import completion, request, rpc

PROMPTS = [
    "Explain why the sky appears blue in two short sentences.",
    "Continue this sequence and explain its rule: 1, 4, 9, 16, 25.",
    "Write a Python function that returns the largest of three integers.",
    "Remember the code word: violet. " + "The visitor walked through the quiet garden. " * 128
    + "\nWhat was the code word at the start? Reply with the word and one short sentence.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18796")
    parser.add_argument("--model", default="zipserv-mxfp4-test")
    parser.add_argument("--mode", choices=("on", "off"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--layers", type=int, default=42)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if (args.report.exists() or not 2 <= args.tokens <= 512 or not 1 <= args.repeats <= 20
            or min(args.workers, args.layers) < 1):
        parser.error("Require a fresh report and bounded positive workload")
    result = {"status": "running", "mode": args.mode, "model": args.model,
        "started_utc": datetime.now(timezone.utc).isoformat(), "prompts": PROMPTS,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Whole model; every MXFP4 MoE layer; fixed prompts and output length",
        "limits": ["This is a serving smoke/performance test, not a broad quality evaluation.",
                   "Client streaming times include scheduling and transport.",
                   "Compare separate runs only with identical model and serving settings.",
                   "Graph replays are not counted as new Python dispatch calls."],
        "requests": []}

    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(".incomplete")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.report)

    def audit():
        workers = rpc(args.base, "zipserv_mxfp4_report", [True], args.workers)
        for worker in workers:
            count = args.layers * 2
            if worker["mode"] != args.mode or worker["layers_seen"] != args.layers:
                raise RuntimeError("Missing or unexpected MXFP4 layers on a worker")
            if args.mode == "on":
                if (worker["compressed_scale_groups"] != count or worker["stock_scale_groups"]
                        or any(not item["raw_owner_released"] or not item["gpu_bytes_exact"]
                               or not item["eager_or_capture_calls"] for item in worker["ownership"])):
                    raise RuntimeError("Incomplete compressed-scale coverage, ownership or execution")
            elif worker["stock_scale_groups"] != count or worker["compressed_scale_groups"]:
                raise RuntimeError("The off control did not retain all stock scales")
        return workers

    save()
    try:
        result["initial_workers"] = audit()
        rendered = [request(args.base, "/tokenize", {"model": args.model,
            "messages": [{"role": "user", "content": prompt}], "add_generation_prompt": True})
            for prompt in PROMPTS]
        prompts = [item["tokens"] for item in rendered]
        if any(not prompt or not all(type(t) is int and t >= 0 for t in prompt) for prompt in prompts):
            raise RuntimeError("Invalid rendered chat-template token IDs")
        result["rendered_prompts"] = rendered
        result["warmup"] = [completion(args.base, args.model, prompt, args.tokens) for prompt in prompts]
        save()
        for repeat in range(args.repeats):
            for index, prompt in enumerate(prompts):
                row = completion(args.base, args.model, prompt, args.tokens)
                row.update(repeat=repeat, prompt_index=index,
                           tokens_equal_warmup=row["token_ids"] == result["warmup"][index]["token_ids"])
                result["requests"].append(row)
                save()
        result["final_workers"] = audit()
        result["summary"] = {
            "repeatable_requests": sum(row["tokens_equal_warmup"] for row in result["requests"]),
            "request_count": len(result["requests"]),
            "median_decode_tokens_per_second": statistics.median(
                row["decode_tokens_per_second"] for row in result["requests"]),
            "median_ttft_seconds": statistics.median(row["ttft_seconds"] for row in result["requests"]),
            "total_retained_scale_bytes": sum(w["scale_retained_bytes"] for w in result["final_workers"]),
            "total_scale_saving_after_scratch_bytes": sum(
                w["scale_saving_after_scratch_bytes"] for w in result["final_workers"]),
            "total_cuda_allocated_bytes": sum(w["cuda_allocated_bytes"] for w in result["final_workers"]),
        }
        result.update(status="passed", finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
