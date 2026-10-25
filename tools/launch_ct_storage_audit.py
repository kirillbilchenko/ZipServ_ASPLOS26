# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Launch a bounded private observer of CT's unchanged normal weight recipe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess

from launch_ct_weight_control import IMAGE, PARENT, private_environment, source_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    root = args.workdir.resolve()
    if root.parent != PARENT or not root.name.startswith("zipserv-ct-coverage-") or not re.fullmatch("[a-z0-9]+", args.run):
        parser.error("Require a private persistent CT coverage directory and alphanumeric trial label")
    template = f"mxwave-glm53-ct-live-20261022-r{args.rank}"
    original = json.loads(subprocess.check_output(["docker", "inspect", template]))[0]
    if original["State"]["Running"] or original["Image"] != IMAGE:
        raise RuntimeError("Original CT template must remain stopped and match its pinned image")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU is occupied; do not disturb another workload")
    if not args.preflight:
        for port in ((18801, 29772) if args.rank == 0 else (29772,)):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
    run = root / args.run
    run.mkdir()
    (run / "reports").mkdir()
    (run / "cache").mkdir()
    environment = private_environment(original["Config"]["Env"], "ct")
    command = list(original["Config"]["Cmd"])
    for flag, value in (("--host", "127.0.0.1"), ("--port", "18801"), ("--master-port", "29772"),
                        ("--worker-extension-cls", "ct_storage_audit.CoverageWorker")):
        command[command.index(flag) + 1] = value
    if any(arg.startswith("--api-key") for arg in command):
        raise RuntimeError("Do not clone inline authentication")
    mounts, cache_seed = [], None
    for mount in original["Mounts"]:
        dest, source = mount["Destination"], mount["Source"]
        if dest.startswith("/run/secrets"):
            continue
        if dest == "/cache":
            cache_seed, source = source, str(run / "cache")
        elif dest == "/reports":
            source = str(run / "reports")
        elif mount["RW"]:
            raise RuntimeError("Unexpected writable production mount: " + dest)
        mounts.append({"source": source, "destination": dest, "rw": mount["RW"]})
    if cache_seed is None:
        raise RuntimeError("Missing read-only CT cache seed")
    sources = source_records(original)
    scripts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "tools").glob("*.py")}
    if not args.preflight:
        previous = json.loads((root / "preflight/launch.json").read_text())
        if previous.get("returncode") != 0 or previous["sources"] != sources or previous["tools"] != scripts:
            raise RuntimeError("Require successful preflight of the identical sources")
        subprocess.run(["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "2g",
            "-v", cache_seed + ":/seed:ro", "-v", str(run / "cache") + ":/destination:rw",
            "--entrypoint", "timeout", IMAGE, "300", "cp", "-a", "--reflink=auto", "/seed/.", "/destination"], check=True)
    bootstrap = root / "bootstrap/zipserv_ct_weight_control-0.0.0.dist-info"
    bootstrap.mkdir(parents=True, exist_ok=True)
    (bootstrap / "METADATA").write_text("Metadata-Version: 2.1\nName: zipserv-ct-weight-control\nVersion: 0.0.0\n")
    (bootstrap / "entry_points.txt").write_text("[vllm.general_plugins]\nzipserv_ct_weight_control = ct_weight_control:install\n")
    mounts.append({"source": str(root), "destination": "/weight-control", "rw": False})
    name = f"{root.name}-{args.run}-r{args.rank}"
    argv = ["docker", "run", "--pull=never", "--name", name, "--network", "host", "--ipc", "host", "--gpus", "all"]
    host = original["HostConfig"]
    for device in host.get("Devices") or []:
        argv += ["--device", f"{device['PathOnHost']}:{device['PathInContainer']}:{device['CgroupPermissions']}"]
    for value in host.get("CapAdd") or []:
        argv += ["--cap-add", value]
    for value in host.get("SecurityOpt") or []:
        argv += ["--security-opt", value]
    for value in host.get("Ulimits") or []:
        argv += ["--ulimit", f"{value['Name']}={value['Soft']}:{value['Hard']}"]
    for key, value in sorted(environment.items()):
        argv += ["-e", key + "=" + value]
    for mount in mounts:
        argv += ["-v", f"{mount['source']}:{mount['destination']}:{'rw' if mount['rw'] else 'ro'}"]
    if args.preflight:
        argv += ["--cpus", "4", "--memory", "8g", "--entrypoint", "timeout", IMAGE, "180", "python3", "-c",
                 "import ct_weight_control as p; p.install(); p.install(); "
                 "from ct_storage_audit import CoverageWorker; import torch; "
                 "assert callable(CoverageWorker.zipserv_ct_capture_kda); "
                 "assert torch.cuda.memory_allocated() == 0; print('CT storage observer preflight passed')"]
    else:
        argv += ["-d", "--entrypoint", "timeout", IMAGE, "3600", "vllm", *command]
    receipt = {"image": IMAGE, "template": template, "name": name, "rank": args.rank,
        "preflight": args.preflight, "cmd": command, "env": environment, "mounts": mounts,
        "sources": sources, "tools": scripts, "compression_changes": False,
        "host_config": {key: host.get(key) for key in ("NetworkMode", "IpcMode", "Devices", "CapAdd", "SecurityOpt", "Ulimits")}}
    (run / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result = subprocess.run(argv, capture_output=True, text=True)
    (run / "launch.log").write_text(result.stdout + result.stderr)
    receipt["returncode"] = result.returncode
    (run / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"name": name, "returncode": result.returncode, "output": (result.stdout + result.stderr)[-2000:]}))
    result.check_returncode()


if __name__ == "__main__":
    main()
