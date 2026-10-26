# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Full GSM8K on the pinned speculative CT runtime, with explicit runtime guards.

The existing vLLM 0.32 client/protocol stay unchanged. Scoring and requests are
imported from that hash-pinned client; this experiment has a separate identity.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import eval_ct_gsm8k as common
from bench_scale_cache_model import rpc

PARENT_SHA = "e481125863152b8fd854020bfd8d563eb145d05923f5e4683ae754be2ad8bd6f"
SCORER_SHA = "c29283fcd47057c250101a7cfac218ea15c8f1ff06610f31a5730f85afbb8e2f"
IMAGE = "sha256:39a03677234ea74e35544d301bd14716c1ae9695ad8d91b16e2c76545e26e8d2"
VERSION = "0.1.dev20051+g487ecf187"
WEIGHT_KEYS = {
    "GLM_DS_DRAFT_HEAD_FP8", "GLM_DS_DRAFT_NVFP4", "GLM_DS_DRAFT_HEAD_FP4",
    "GLM_CERT_HEAD", "GLM_CERT_HEAD_DRAFT", "GLM_MHC_BF16W", "GLM_CERT_HEAD_DUMP",
    "MXWAVE_COMPARE_ARM", "MXWAVE_MOE_MODE", "MXWAVE_GLM53_VOCAB_MODE",
    "MXWAVE_GLM53_KDA_MODE", "MXWAVE_PALETTE_MODEL", "ZIPSERV_CT_CONTROL_MODE",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def client_sources():
    result = {name: sha(Path(__file__).with_name(name).read_bytes()) for name in
              ("eval_exact_ct_gsm8k.py", "eval_ct_gsm8k.py", "bench_scale_cache_model.py")}
    if result["eval_ct_gsm8k.py"] != SCORER_SHA:
        raise ValueError("The frozen scoring/request implementation changed")
    return result


def contract(launch):
    if launch["image"] != IMAGE or launch["rank"] != 0 or launch["mode"] not in ("off", "ct"):
        raise ValueError("Require the pinned CT rank-zero launch")
    command = launch["cmd"]
    def option(key):
        return command[command.index(key) + 1]
    if (option("--max-model-len") != "500000" or option("--kv-cache-memory-bytes") != str(6 * 1024**3)
            or option("--max-num-seqs") != "32" or option("--max-num-batched-tokens") != "6919"
            or option("--kv-cache-dtype") != "fp8_e4m3" or "--enable-prefix-caching" not in command
            or option("--worker-extension-cls") != "ct_weight_control.ControlWorker"
            or json.loads(option("--speculative-config"))["method"] != "dflash"):
        raise ValueError("The exact CT serving configuration changed")
    return {key: launch[key] for key in ("image", "cmd", "host_config", "sources", "tools")} | {
        "env": {key: value for key, value in launch["env"].items() if key not in WEIGHT_KEYS}}


def freeze(parent_bytes, launch):
    if sha(parent_bytes) != PARENT_SHA:
        raise ValueError("Require the frozen complete GSM8K parent protocol")
    protocol = json.loads(parent_bytes)
    protocol.update(version=2, runtime_profile="exact_CT_DFlash_v1", parent_protocol_sha256=PARENT_SHA,
                    speculative_decoding=True, capture_sizes=[1, 2, 4, 6, 8, 12, 16, 18, 24, 32],
                    runtime_contract=contract(launch), scoring_source_sha256=SCORER_SHA,
                    physical_graph_token_counts=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 28, 32])
    protocol["limits"] += ["CT's pinned runtime and DFlash are enabled in both arms.",
                           "Normal CT includes lossy post-load KDA MXFP8 conversion.",
                           "Do not pool these scores with the different vLLM 0.32 runtime."]
    return protocol


def validate(protocol, parent_bytes, launch):
    expected = freeze(parent_bytes, launch)
    if protocol != expected:
        raise ValueError("Protocol, questions or exact runtime differ from the frozen experiment")
    if launch.get("preflight") or launch.get("returncode") != 0:
        raise ValueError("Require a successful serving launch")


def audit(base, mode, protocol):
    rows = rpc(base, "ct_weight_control_report", [], 2)
    if {r["rank"] for r in rows} != {0, 1}:
        raise ValueError("Both CT workers must acknowledge the evaluation")
    for row in rows:
        recipe = row["ct_recipe"]
        if (row["status"] != "verified" or row["mode"] != mode or row["vllm_version"] != VERSION
                or not recipe["speculation"] or recipe["draft_layers"] != 5 or not recipe["shared_head"]
                or recipe["max_model_len"] != 500000 or recipe["max_num_seqs"] != 32
                or recipe["max_num_batched_tokens"] != 6919
                or recipe["kv_cache_memory_bytes"] != 6 * 1024**3
                or not recipe["enable_prefix_caching"]
                or recipe["physical_graph_token_counts"] != protocol["physical_graph_token_counts"]):
            raise ValueError("Actual CT worker settings do not match the quality protocol")
        if mode == "off" and (len(row["kda"]) != 68 or len(row["raw_scales"]) != 84
                              or row["optional_head_twins"] or row["mhc_bf16_twins"]):
            raise ValueError("An optional weight representation remains enabled")
        if mode == "ct" and (recipe["experts"]["layers_seen"] != 42
                             or recipe["vocabulary"]["resident_layers"] != 1
                             or len(recipe["kda"]["layers"]) != 68):
            raise ValueError("Normal CT compression coverage is incomplete")
    return rows


def run(args):
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    launch = json.loads(args.server_receipt.read_text())
    validate(protocol, args.parent_protocol.read_bytes(), launch)
    if launch["mode"] != args.mode:
        raise ValueError("Requested mode differs from the actual launch")
    sources = client_sources()
    args.output.mkdir()
    (args.output / "protocol.json").write_bytes(protocol_bytes)
    (args.output / "server-launch.json").write_bytes(args.server_receipt.read_bytes())
    metadata = {"arm": args.mode, "protocol_sha256": sha(protocol_bytes), "model": "glm-5.3-flash",
                "client_sha256": sha(json.dumps(sources, sort_keys=True).encode()), "client_sources": sources,
                "client_identity_kind": "SHA256 of sorted source-digest JSON", "base": args.base,
                "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    results = []
    start = time.monotonic()
    def save():
        report = {**metadata, **common.summarize(results, 1319), "elapsed_seconds": time.monotonic() - start}
        tmp = args.output / "summary.incomplete"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.output / "summary.json")
    save()
    try:
        before = audit(args.base, args.mode, protocol)
        (args.output / "workers-before.json").write_text(json.dumps(before, indent=2) + "\n")
        with (args.output / "responses.jsonl").open("x") as log, ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(common.execute, case, protocol, args.base, "glm-5.3-flash")
                       for case in protocol["cases"]]
            for future in as_completed(futures):
                row = future.result()
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
                results.append(row)
                save()
                if len(results) % 25 == 0:
                    print(json.dumps(common.summarize(results, 1319)), flush=True)
        after = audit(args.base, args.mode, protocol)
        (args.output / "workers-after.json").write_text(json.dumps(after, indent=2) + "\n")
        if args.mode == "off":
            for a, b in zip(sorted(before, key=lambda r: r["rank"]), sorted(after, key=lambda r: r["rank"]), strict=True):
                if any(a[key] != b[key] for key in ("kda", "raw_scales", "embedding", "head")):
                    raise ValueError("Raw source bytes changed during evaluation")
        metadata.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--parent-protocol", required=True, type=Path)
    p.add_argument("--reference-launch", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("run")
    p.add_argument("--parent-protocol", required=True, type=Path)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--server-receipt", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--mode", choices=("off", "ct"), required=True)
    p.add_argument("--base", default="http://127.0.0.1:18797")
    p = sub.add_parser("compare")
    p.add_argument("control", type=Path)
    p.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        client_sources()
        value = freeze(args.parent_protocol.read_bytes(), json.loads(args.reference_launch.read_text()))
        with args.output.open("x") as f:
            json.dump(value, f, indent=2)
            f.write("\n")
    elif args.action == "run":
        run(args)
    else:
        print(json.dumps(common.compare(args.control, args.candidate), indent=2))


if __name__ == "__main__":
    main()
