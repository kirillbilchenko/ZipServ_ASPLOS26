# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Launch one isolated TP2 NVFP4 smoke-test rank from an existing checkpoint.

Run this on each Spark. The output must be new, the GPU must be idle, and no
existing service is stopped. A read-only hard-linked model view permits a
pinned upstream config update without changing the original checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--api-port", type=int, required=True)
    parser.add_argument("--network-env", type=Path, required=True)
    parser.add_argument("--flashinfer-autotune", choices=("auto", "off"), default="auto",
                        help="Disable only for a labelled startup diagnostic, not a speed comparison")
    parser.add_argument("--cache-dir", type=Path,
                        help="Optional existing per-rank cache for a labelled warm-start retry")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory; previous evidence is immutable")
    active = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("GPU has active compute processes; leaving them untouched: " + active)
    if subprocess.run(["docker", "container", "inspect", args.name],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError("Container name already exists")
    source = args.source.resolve()
    index = json.loads((source / "model.safetensors.index.json").read_text())
    if not all((source / name).is_file() for name in set(index["weight_map"].values())):
        raise RuntimeError("Checkpoint shard missing")
    config = json.loads(args.config.read_text())
    if config.get("quantization_config", {}).get("quant_algo") != "NVFP4":
        raise ValueError("This first smoke test requires the plain NVIDIA NVFP4 checkpoint")
    allowed = {"NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "VLLM_HOST_IP",
               "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "NCCL_IB_DISABLE"}
    network = json.loads(args.network_env.read_text())
    if not isinstance(network, list) or any(e.split("=", 1)[0] not in allowed or "=" not in e for e in network):
        raise ValueError("Unexpected network environment variable")
    output = args.output.resolve()
    (output / "model").mkdir(parents=True)
    for file in source.iterdir():
        if file.is_file() and file.name != "config.json":
            # Hard links share immutable source storage; Docker sees a read-only mount.
            (output / "model" / file.name).hardlink_to(file)
    (output / "model/config.json").write_bytes(args.config.read_bytes())
    cache = args.cache_dir.resolve() if args.cache_dir else output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    server = ["-m", "vllm.entrypoints.cli.main", "serve", "/model",
              "--served-model-name", "fp4-checkpoint-smoke", "--host", "127.0.0.1",
              "--port", str(args.api_port), "--dtype", "bfloat16", "--language-model-only",
              "--tensor-parallel-size", "2", "--distributed-executor-backend", "mp",
              "--nnodes", "2", "--node-rank", str(args.rank),
              "--master-addr", args.master_addr, "--master-port", str(args.master_port),
              "--max-model-len", "4096", "--max-num-seqs", "4",
              "--max-num-batched-tokens", "512", "--enable-chunked-prefill",
              "--kv-cache-dtype", "fp8", "--kv-cache-memory-bytes", str(2 * 2**30),
              "--gpu-memory-utilization", "0.90", "--enforce-eager",
              "--reasoning-parser", "glm45", "--seed", "42",
              "--distributed-timeout-seconds", "900"]
    if args.rank:
        server.append("--headless")
    if args.flashinfer_autotune == "off":
        server.extend(("--kernel-config", '{"enable_flashinfer_autotune":false}'))
    command = ["docker", "run", "-d", "--name", args.name, "--gpus", "all",
               "--network", "host", "--ipc", "host", "--device", "/dev/infiniband",
               "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
               "-v", str(output / "model") + ":/model:ro",
               "-v", str(cache) + ":/root/.cache"]
    environment = [*network, "VLLM_PLUGINS=", "HF_HUB_OFFLINE=1",
                   "TRANSFORMERS_OFFLINE=1", "VLLM_WORKER_MULTIPROC_METHOD=spawn",
                   "OMP_NUM_THREADS=4"]
    for value in environment:
        command.extend(("-e", value))
    command += ["--entrypoint", "python3", args.image, *server]
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image], text=True))[0]
    receipt = {"status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Text-only eager serving smoke; no compression, MTP, or benchmark claim",
               "rank": args.rank, "source": str(source), "checkpoint_revision": args.revision,
               "flashinfer_autotune": args.flashinfer_autotune,
               "cache_dir": str(cache), "cache_reused": args.cache_dir is not None,
               "index_sha256": digest(source / "model.safetensors.index.json"),
               "source_config_sha256": digest(source / "config.json"),
               "used_config_sha256": digest(args.config), "image_id": image["Id"],
               "launcher_sha256": digest(Path(__file__)), "command": command,
               "tensor_payload_bytes": index["metadata"]["total_size"],
               "weight_validation": "Shard presence and upstream index identity; no full payload rehash yet"}
    path = output / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        identity = subprocess.check_output(command, text=True).strip()
        receipt.update(status="launched", container_id=identity)
    except Exception as error:
        receipt.update(status="launch_failed", error=repr(error))
        raise
    finally:
        path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "launched", "rank": args.rank, "receipt": str(path)}))


if __name__ == "__main__":
    main()
