# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Run BIZ's unchanged, pinned NLL scorer with durable request/launch evidence.

Supply a checkout of Bizuayeu/GLM-5.3-Flash-DGX-Sparks-BIZ at BIZ_COMMIT.
No model code is built or patched. The scorer's 16-document corpus, token-ID
prompts, two repeats, top-5 alternatives and position-weighted NLL are retained.
This measures teacher-forced NLL, not speculative-decode speed or task accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error

BIZ_COMMIT = "5fe884697c772617a65762f0c339fb2b3a4bf6ed"
BIZ_FILES = {
    "v3/glm53_tf/__init__.py": "95a8189ff7e1364016debce827da7894fdf71d56e08ba30e833e17f004831942",
    "v3/glm53_tf/config.py": "38c0dbb7bd483e3925d991cd42e729e6e4f685e7766299f62d684bab0210fabd",
    "v3/glm53_tf/score_nll.py": "40fda725906792f104c54a2b48689cdb98f7074d960288f4fbb20c47f3f77842",
    "v3/glm53_tf/agreement.py": "7677801acec971169268e4efd410b532c62d9d781cae98336590cf30913ede6b",
    "v3/config/nll_set.json": "60904e4043642bf0d75f39bb2f74554e5c07100d09ccb27bcaacc54a8b006521",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_scorer(root: Path):
    for name, expected in BIZ_FILES.items():
        if sha((root / name).read_bytes()) != expected:
            raise ValueError("The pinned BIZ scoring contract changed: " + name)
    if any(name == "glm53_tf" or name.startswith("glm53_tf.") for name in sys.modules):
        raise RuntimeError("Run in a fresh Python process to avoid another imported scorer")
    sys.path.insert(0, str(root / "v3"))
    scorer = importlib.import_module("glm53_tf.score_nll")
    if Path(scorer.__file__).resolve() != (root / "v3/glm53_tf/score_nll.py").resolve():
        raise RuntimeError("A different scorer was imported")
    return scorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--server-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    scorer = load_scorer(args.upstream_root.resolve())
    corpus = args.upstream_root / "v3/config/nll_set.json"
    data = json.loads(corpus.read_text())
    found = scorer.problems(data, args.tokenizer.read_bytes())
    if found:
        raise ValueError("; ".join(found))
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prompts = {}
    for item in data["texts"]:
        ids = tokenizer.encode(item["text"], add_special_tokens=True).ids
        if len(ids) != item["tokens"]:
            raise ValueError("Token count differs from the published set: " + item["name"])
        prompts[tuple(ids)] = item["name"]
    launch_bytes = args.server_receipt.read_bytes()
    json.loads(launch_bytes)  # Reject a non-JSON receipt before creating output.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "server-launch.json").write_bytes(launch_bytes)
    (args.output / "nll_set.json").write_bytes(corpus.read_bytes())
    record = {"status": "running", "arm": args.arm, "model": args.model,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "biz_commit": BIZ_COMMIT, "source_hashes": BIZ_FILES,
              "client_sha256": sha(Path(__file__).read_bytes()),
              "server_receipt_sha256": sha(launch_bytes),
              "tokenizer_sha256": sha(args.tokenizer.read_bytes()),
              "documents": len(data["texts"]),
              "positions_per_repeat": sum(t["tokens"] - 1 for t in data["texts"]),
              "requests_completed": 0,
              "limits": ["16-document NLL screen; not the separate 128-document BF16 agreement test.",
                         "No BF16 reference distributions or KL result.",
                         "Published TensorFold results remain cross-engine comparisons.",
                         "One generated token per request; elapsed time is not decode throughput."]}

    def save():
        temporary = args.output / "receipt.incomplete"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(args.output / "receipt.json")

    original_post = scorer.post
    save()
    with (args.output / "requests.jsonl").open("x") as trace:
        def logged_post(url, path, body=None, timeout=900.0):
            row = {"path": path, "request": body,
                   "document": prompts.get(tuple(body.get("prompt", []))) if body else None}
            start = time.monotonic()
            try:
                response = original_post(url, path, body, timeout)
                row["response"] = response
                return response
            except urllib.error.HTTPError as error:
                # Reading error bodies here would consume what the unchanged
                # upstream probe uses to explain unsupported prompt_logprobs.
                row["error"] = {"type": type(error).__name__, "status": error.code}
                raise
            except Exception as error:
                row["error"] = {"type": type(error).__name__, "message": str(error)}
                raise
            finally:
                row["elapsed_seconds"] = time.monotonic() - start
                trace.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                trace.flush()
                os.fsync(trace.fileno())
                record["requests_completed"] += 1
                save()
                print(json.dumps({"document": row["document"],
                                  "seconds": round(row["elapsed_seconds"], 3),
                                  "error": row.get("error")}), flush=True)

        scorer.post = logged_post
        try:
            code = scorer.main(["--url", args.url, "--tokenizer", str(args.tokenizer),
                                "--set", str(corpus), "--model", args.model,
                                "--seed", "42", "--out", str(args.output / "nll.json")])
            record.update(status="complete" if code == 0 else "failed", exit_code=code)
        except BaseException as error:
            record.update(status="failed", error=repr(error))
            raise
        finally:
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
