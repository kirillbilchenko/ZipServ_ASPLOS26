# Copyright MxWave contributors.
# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Frozen, full-test GSM8K comparison for a private CT serving experiment.

Protocol/scoring adapted from MxWave; exact source identities are in CONTRIBUTING.md.
No retries, subset selection, external judge, or silent exclusion of errors.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

DATA_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def numeric_answer(text: str) -> Decimal | None:
    value = text.rsplit("####", 1)[-1].strip() if "####" in text else text.strip()
    if not re.fullmatch(r"[-+]?\$?\d[\d,]*(?:\.\d+)?\.?", value):
        return None
    return Decimal(value.replace("$", "").replace(",", "").rstrip("."))


def freeze(raw: bytes) -> dict:
    if digest(raw) != DATA_SHA256:
        raise ValueError("The pinned GSM8K test payload changed")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 1319 or len({row["question"] for row in rows}) != 1319:
        raise ValueError("Require all 1,319 unique GSM8K test questions")
    cases = []
    for index, row in enumerate(rows):
        expected = row["answer"].rsplit("####", 1)[1].strip()
        if numeric_answer(expected) is None:
            raise ValueError("Non-numeric ground truth")
        cases.append({"source_index": index, "question": row["question"],
            "expected": expected, "prompt": row["question"]
            + "\nSolve the problem. End your final answer with #### followed by the numeric answer."})
    return {"version": 1, "dataset": "GSM8K main/test", "source_sha256": DATA_SHA256,
        "cases": cases, "questions_per_arm": 1319, "concurrency": 4, "fewshot": 0,
        "seed": 42, "temperature": 0, "max_tokens": 1024, "reasoning_effort": "low",
        "capture_sizes": [1, 4], "speculative_decoding": False, "retries": 0,
        "primary_score": "Natural stop and final numeric match using last-number extraction",
        "secondary_score": "Natural stop and strict bare-number or ####-tail format",
        "limits": ["Zero-shot chat; not a published few-shot leaderboard protocol.",
                   "Variable answer lengths and scheduling can differ between arms.",
                   "Wilson intervals describe question sampling, not temporal reproducibility.",
                   "Both arms use the same already quantized CT checkpoint."]}


def score(response: dict, expected: str) -> dict:
    choice = response["choices"][0]
    text = choice["message"].get("content") or ""
    if not isinstance(text, str):
        raise TypeError("Expected a text answer")
    if "</think>" in text:
        final = text.rsplit("</think>", 1)[1].strip()
    else:
        final = "" if "<think>" in text else text.strip()
    numeric = final.rsplit("####", 1)[-1] if "####" in final else final
    values = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", numeric.replace("$", ""))
    predicted = Decimal(values[-1].replace(",", "")) if values else None
    truth = numeric_answer(expected)
    if truth is None:
        raise ValueError("Non-numeric ground truth")
    stopped = choice["finish_reason"] == "stop"
    return {"answer_exact_match": bool(stopped and predicted == truth),
        "strict_format_match": bool(stopped and numeric_answer(final) == truth),
        "final_answer": final, "extracted_numeric_answer": str(predicted) if predicted is not None else None,
        "finish_reason": choice["finish_reason"], "naturally_stopped": stopped,
        "truncated": choice["finish_reason"] == "length", "api_error": False}


def interval(passed: int, count: int) -> list[float]:
    if not 0 <= passed <= count or not count:
        raise ValueError("Invalid score count")
    z = 1.959963984540054
    p = passed / count
    d = 1 + z * z / count
    c = (p + z * z / (2 * count)) / d
    r = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / d
    return [max(0., c - r), min(1., c + r)]


def summarize(rows: list[dict], total: int) -> dict:
    if len({row["source_index"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question results")
    count = len(rows)
    passed = sum(row["answer_exact_match"] for row in rows)
    strict = sum(row["strict_format_match"] for row in rows)
    return {"complete": count == total, "completed_questions": count, "expected_questions": total,
        "answer_exact_matches": passed, "accuracy": passed / total,
        "accuracy_wilson95": interval(passed, total) if count == total else None,
        "strict_format_matches": strict, "strict_accuracy": strict / total,
        "truncations": sum(row.get("truncated", False) for row in rows),
        "api_errors": sum(row.get("api_error", False) for row in rows),
        "missing_questions": total - count}


def execute(case: dict, protocol: dict, base: str, model: str) -> dict:
    body = {"model": model, "temperature": protocol["temperature"], "seed": protocol["seed"],
        "max_tokens": protocol["max_tokens"], "messages": [{"role": "user", "content": case["prompt"]}],
        "chat_template_kwargs": {"reasoning_effort": protocol["reasoning_effort"]}}
    row = {"source_index": case["source_index"], "expected": case["expected"], "request": body,
           "started_utc": datetime.now(timezone.utc).isoformat()}
    start = time.monotonic()
    try:
        request = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read().decode("utf-8")
        row["raw_response"] = raw
        parsed = json.loads(raw)
        row["response"] = parsed
        row.update(score(parsed, case["expected"]))
    except Exception as error:
        if isinstance(error, urllib.error.HTTPError):
            row["raw_response"] = error.read().decode("utf-8", errors="replace")
        row.update(api_error=True, error=repr(error), answer_exact_match=False,
                   strict_format_match=False, truncated=False, naturally_stopped=False)
    row["elapsed_seconds"] = time.monotonic() - start
    return row


def run(protocol_path: Path, output: Path, base: str, model: str, arm: str, receipt: Path) -> None:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if (protocol["source_sha256"] != DATA_SHA256 or len(protocol["cases"]) != 1319
            or protocol["questions_per_arm"] != 1319 or protocol["concurrency"] != 4
            or protocol["retries"] != 0):
        raise ValueError("Require the frozen, complete GSM8K protocol")
    launch = json.loads(receipt.read_text())
    command = launch["command"]
    compilation = json.loads(command[command.index("--compilation-config") + 1])
    if (compilation["cudagraph_capture_sizes"] != protocol["capture_sizes"]
            or compilation["max_cudagraph_capture_size"] != max(protocol["capture_sizes"])
            or "--speculative-config" in command):
        raise ValueError("Server graph/speculation settings do not match the frozen protocol")
    output.mkdir()  # Exclusive: never overwrite an earlier or interrupted run.
    (output / "protocol.json").write_bytes(protocol_bytes)
    (output / "server-launch.json").write_bytes(receipt.read_bytes())
    metadata = {"arm": arm, "protocol_sha256": digest(protocol_bytes), "model": model,
        "base": base, "client_sha256": digest(Path(__file__).read_bytes()),
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    results = []
    start = time.monotonic()

    def save() -> None:
        report = {**metadata, **summarize(results, protocol["questions_per_arm"]),
                  "elapsed_seconds": time.monotonic() - start}
        temp = output / "summary.incomplete"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(output / "summary.json")

    save()
    try:
        with (output / "responses.jsonl").open("x") as log, ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(execute, case, protocol, base, model) for case in protocol["cases"]]
            for future in as_completed(futures):
                row = future.result()
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
                results.append(row)
                save()
                if len(results) % 25 == 0:
                    print(json.dumps(summarize(results, 1319)), flush=True)
        metadata.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        metadata.update(status="interrupted", error=repr(error))
        raise
    finally:
        save()


def compare(control: Path, candidate: Path) -> dict:
    def read(root: Path) -> tuple[dict, dict]:
        summary = json.loads((root / "summary.json").read_text())
        rows = [json.loads(line) for line in (root / "responses.jsonl").read_text().splitlines()]
        if (summary["status"] != "complete" or len(rows) != 1319
                or {row["source_index"] for row in rows} != set(range(1319))):
            raise ValueError("Paired comparison requires two complete, unique full-test runs")
        return summary, {row["source_index"]: row for row in rows}
    a, ar = read(control)
    b, br = read(candidate)
    if (a["protocol_sha256"] != b["protocol_sha256"] or a["model"] != b["model"]
            or a["client_sha256"] != b["client_sha256"]):
        raise ValueError("Unmatched protocols, scoring client or model identifiers")
    gains = [i for i in ar if not ar[i]["answer_exact_match"] and br[i]["answer_exact_match"]]
    losses = [i for i in ar if ar[i]["answer_exact_match"] and not br[i]["answer_exact_match"]]
    return {"control": a, "candidate": b, "accuracy_delta_percentage_points":
        100 * (b["accuracy"] - a["accuracy"]), "newly_correct_indices": gains,
        "newly_incorrect_indices": losses, "unchanged_correctness": 1319 - len(gains) - len(losses),
        "limits": "Question-level outcomes from one run per arm; repeats are needed to separate runtime variability."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base", default="http://127.0.0.1:18796")
    p.add_argument("--model", default="zipserv-mxfp4-test")
    p.add_argument("--arm", required=True)
    p.add_argument("--server-receipt", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("control", type=Path)
    p.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        with args.output.open("x") as f:
            json.dump(freeze(args.dataset.read_bytes()), f, indent=2)
            f.write("\n")
    elif args.command == "run":
        run(args.protocol, args.output, args.base, args.model, args.arm, args.server_receipt)
    else:
        print(json.dumps(compare(args.control, args.candidate), indent=2))


if __name__ == "__main__":
    main()
