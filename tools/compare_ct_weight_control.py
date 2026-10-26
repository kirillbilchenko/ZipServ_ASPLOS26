# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Require matched CT serving settings before comparing raw and CT weight recipes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench_ct_weight_control import audit_stability
from ct_weight_control import DISABLED

ENV_CHANGES = {*DISABLED, "MXWAVE_COMPARE_ARM", "MXWAVE_MOE_MODE",
               "MXWAVE_GLM53_VOCAB_MODE", "MXWAVE_GLM53_KDA_MODE",
               "MXWAVE_PALETTE_MODEL", "GLM_CERT_HEAD_DUMP", "ZIPSERV_CT_CONTROL_MODE"}
SHARED = ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "kv_cache_memory_bytes",
          "enable_prefix_caching", "head_weight_dtype", "head_quant_method", "speculation",
          "physical_graph_token_counts", "shared_head", "draft_layers", "draft_context_dtype",
          "overlay_source_hashes")


def serving_counters(result):
    """Use total counters, excluding similarly named creation timestamps."""
    names = {"drafts": "spec_decode_num_drafts_total", "draft_tokens": "spec_decode_num_draft_tokens_total",
             "accepted_tokens": "spec_decode_num_accepted_tokens_total",
             "prefix_queries": "prefix_cache_queries_total", "prefix_hits": "prefix_cache_hits_total"}
    out = {"scope": "Entire client run, including warmup requests"}
    for key, name in names.items():
        values = []
        for stage in ("before", "after"):
            found = [float(line.rsplit(" ", 1)[-1]) for line in result["serving_metrics_" + stage]
                     if line.startswith("vllm:" + name + "{")]
            if not found:
                raise ValueError("Serving counter missing: " + name)
            values.append(sum(found))
        if values[1] < values[0]:
            raise ValueError("Serving counter reset during measurement: " + name)
        out[key] = values[1] - values[0]
    if not out["drafts"] or not out["draft_tokens"]:
        raise ValueError("No speculative decoding occurred")
    out["accepted_per_draft"] = out["accepted_tokens"] / out["drafts"]
    out["acceptance_percent"] = 100 * out["accepted_tokens"] / out["draft_tokens"]
    return out


def compare_launch(raw, ct):
    for key in ("image", "template", "rank", "cmd", "host_config", "sources", "tools"):
        if raw[key] != ct[key]:
            raise ValueError("Launch mismatch: " + key)
    changed = {key for key in raw["env"].keys() | ct["env"].keys()
               if raw["env"].get(key) != ct["env"].get(key)}
    if changed - ENV_CHANGES:
        raise ValueError("Non-weight flags changed: " + str(sorted(changed - ENV_CHANGES)))
    for record in (raw, ct):
        if record["returncode"] or record["preflight"]:
            raise ValueError("Require successful serving launches")
    if (raw["mode"], ct["mode"]) != ("off", "ct"):
        raise ValueError("Require raw and normal-CT arms")
    left = {row["destination"]: row for row in raw["mounts"]}
    right = {row["destination"]: row for row in ct["mounts"]}
    if left.keys() != right.keys():
        raise ValueError("Mount coverage changed")
    for key in left:
        if key not in ("/cache", "/reports") and left[key] != right[key]:
            raise ValueError("A runtime/model mount changed: " + key)
    return {key: {"off": raw["env"].get(key), "ct": ct["env"].get(key)} for key in sorted(changed)}


def compare_results(raw, ct):
    if (raw["mode"], ct["mode"]) != ("off", "ct"):
        raise ValueError("Require the all-optional-weights-off baseline and normal CT")
    for result in (raw, ct):
        if result["status"] != "measured":
            raise ValueError("An arm did not finish measuring")
        audit_stability(result["initial_workers"], result["final_workers"], result["mode"])
    for key in ("source_hashes", "prompt_texts", "rendered_prompts", "scope"):
        if raw[key] != ct[key]:
            raise ValueError("Workload mismatch: " + key)
    before = {row["rank"]: row for row in raw["final_workers"]}
    after = {row["rank"]: row for row in ct["final_workers"]}
    payload = {"fp4_scales": 0, "bf16_embedding": 0, "kda_bf16_to_mxfp8": 0}
    for rank in before:
        a, b = before[rank], after[rank]
        if a["vllm_version"] != b["vllm_version"]:
            raise ValueError("Runtime version changed")
        for key in SHARED:
            if a["ct_recipe"][key] != b["ct_recipe"][key]:
                raise ValueError("CT serving configuration changed: " + key)
        recipe = b["ct_recipe"]
        if sum(row["bytes"] for row in a["raw_scales"]) != recipe["experts"]["scale_raw_bytes"]:
            raise ValueError("Expert-scale coverage changed")
        vocab = recipe["vocabulary"]
        if (vocab["resident_layers"] != 1 or a["embedding"]["sha256"] != vocab["layers"][0]["source_sha256"]
                or a["embedding"]["bytes"] != vocab["raw_bytes"]):
            raise ValueError("Embedding source changed")
        original = {row["name"]: row for row in a["kda"]}
        converted = {row["name"]: row for row in recipe["kda"]["layers"]}
        if original.keys() != converted.keys():
            raise ValueError("KDA coverage changed")
        for name, item in original.items():
            if item["sha256"] != converted[name]["source_sha256"] or item["shape"] != converted[name]["shape"]:
                raise ValueError("KDA source changed: " + name)
        payload["fp4_scales"] += recipe["experts"]["scale_raw_bytes"] - recipe["experts"]["scale_retained_bytes"]
        payload["bf16_embedding"] += vocab["raw_bytes"] - vocab["retained_bytes"]
        payload["kda_bf16_to_mxfp8"] += sum(row["bytes"] for row in original.values()) - sum(
            row["resident"]["weight_bytes"] + row["resident"]["scale_bytes"] for row in converted.values())
    speed = {}
    for workload in ("short_cold", "short_warm", "long_cold", "c4"):
        speed[workload] = {}
        for metric in raw["summary"][workload]:
            a, b = raw["summary"][workload][metric], ct["summary"][workload][metric]
            speed[workload][metric] = {"off": a, "ct": b, "ct_change_pct": 100 * (b / a - 1)}
    paired = []
    for a, b in zip(raw["requests"], ct["requests"], strict=True):
        if any(a[key] != b[key] for key in ("workload", "repeat", "cache_salt", "prompt_tokens")):
            raise ValueError("Measured requests are not matched")
        if len(a["token_ids"]) != 64 or len(b["token_ids"]) != 64:
            raise ValueError("Measured output length changed")
        paired.append(a["token_ids"] == b["token_ids"])
    paired_c4 = []
    for a, b in zip(raw["cohorts"], ct["cohorts"], strict=True):
        for x, y in zip(a["requests"], b["requests"], strict=True):
            if x["cache_salt"] != y["cache_salt"] or x["prompt_tokens"] != y["prompt_tokens"]:
                raise ValueError("Concurrent requests are not matched")
            if len(x["token_ids"]) != 64 or len(y["token_ids"]) != 64:
                raise ValueError("Concurrent output length changed")
            paired_c4.append(x["token_ids"] == y["token_ids"])
    a = sum(row["cuda_allocated_bytes"] for row in before.values())
    b = sum(row["cuda_allocated_bytes"] for row in after.values())
    return {"status": "matched", "speed": speed,
            "live_cuda": {"off_bytes": a, "ct_bytes": b, "saved_bytes": a - b,
                          "saved_gib": (a - b) / 2**30, "saved_percent": 100 * (a - b) / a},
            "payload_savings_bytes": payload,
            "serving_counters": {"off": serving_counters(raw), "ct": serving_counters(ct)},
            "c1_identical_outputs": {"matched": sum(paired), "total": len(paired)},
            "c4_identical_outputs": {"matched": sum(paired_c4), "total": len(paired_c4)},
            "limits": ["One launch per arm; small timing changes are not established gains.",
                       "CT includes lossy KDA conversion and optional head/mHC copies.",
                       "Payload savings exclude optional copies, scratch, graphs and allocator effects.",
                       "Configured context is 500K; measured prompts are much shorter.",
                       "The short_warm label repeats a cache salt; actual prefix reuse is reported by counters.",
                       "GSM8K quality is unmeasured for this exact pair."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--off-run", default="run1-off")
    parser.add_argument("--ct-run", default="run2-ct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing comparison reports")
    def read(host, run, name):
        return json.loads((args.evidence / host / run / name).read_text())
    changes = {host: compare_launch(read(host, args.off_run, "launch.json"),
                                    read(host, args.ct_run, "launch.json")) for host in ("spark", "spark2")}
    result = compare_results(read("spark", args.off_run, "benchmark.json"),
                             read("spark", args.ct_run, "benchmark.json"))
    result["launch_flag_changes"] = changes
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "live_cuda": result["live_cuda"], "speed": result["speed"]}))


if __name__ == "__main__":
    main()
