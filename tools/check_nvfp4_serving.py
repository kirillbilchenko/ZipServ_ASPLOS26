# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Bounded readiness and basic-generation check, not a quality/speed benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18831")
    parser.add_argument("--model", default="fp4-checkpoint-smoke")
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.wait_seconds <= 1800:
        parser.error("Use a new output file and a bounded readiness timeout")
    record = {"status": "waiting", "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Text-only eager smoke; no quality equivalence or performance claim",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "requests": []}

    def save():
        temp = args.output.with_suffix(".incomplete")
        temp.write_text(json.dumps(record, indent=2) + "\n")
        temp.replace(args.output)

    def request(path, body=None, timeout=5):
        encoded = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.base.rstrip("/") + path, data=encoded,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
        return json.loads(raw) if raw else None

    save()
    try:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            state = subprocess.check_output(["docker", "inspect", "-f", "{{.State.Status}}", args.container], text=True).strip()
            if state != "running":
                raise RuntimeError("Test server container is " + state)
            try:
                request("/health")
                break
            except (OSError, ValueError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server did not become healthy before the deadline")
                time.sleep(5)
        record.update(status="generating", models=request("/v1/models"),
                      ready_utc=datetime.now(timezone.utc).isoformat())
        save()
        prompts = [("What is 7 times 8? Reply with only the number.", "56"),
                   ("What is the capital of France? Reply with only its name.", "paris"),
                   ("Remember the code word: violet. What is the code word? Reply with only the word.", "violet")]
        for prompt, expected in prompts:
            body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0, "seed": 42, "max_tokens": 256,
                    "chat_template_kwargs": {"reasoning_effort": "low"}}
            start = time.monotonic()
            response = request("/v1/chat/completions", body, timeout=300)
            choice = response["choices"][0]
            content = choice["message"].get("content") or ""
            passed = choice["finish_reason"] == "stop" and expected in content.strip().lower()
            record["requests"].append({"request": body, "response": response,
                                       "elapsed_seconds": time.monotonic() - start,
                                       "basic_check_passed": passed})
            save()
            if not passed:
                raise RuntimeError("Basic generation check failed or was truncated")
        record.update(status="passed", finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        save()
        print(json.dumps({"status": record["status"], "requests": len(record["requests"]),
                          "report": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
