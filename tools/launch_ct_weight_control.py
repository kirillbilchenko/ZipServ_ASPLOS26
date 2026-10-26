# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Clone the stopped CT template into a private, bounded weight-control trial."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess

from ct_weight_control import DISABLED

IMAGE = "sha256:39a03677234ea74e35544d301bd14716c1ae9695ad8d91b16e2c76545e26e8d2"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def private_environment(original, mode):
    env = {}
    for item in original:
        key, value = item.split("=", 1)
        upper = key.upper()
        if (any(word in upper for word in ("SECRET", "PASSWORD", "API_KEY", "ACCESS_KEY", "AUTHORIZATION"))
                or upper.endswith(("_TOKEN", "_TOKEN_FILE"))):
            continue
        if key.startswith("ZIPSERV_"):
            raise RuntimeError("The CT template unexpectedly includes extra ZipServ configuration")
        env[key] = value
    if env.get("MXWAVE_COMPARE_ARM") != "B" or env.get("MXWAVE_MOE_MODE") != "aligned":
        raise RuntimeError("The template is not the qualified CT recipe")
    if mode == "off":
        env.update(MXWAVE_COMPARE_ARM="A", MXWAVE_MOE_MODE="stock",
                   MXWAVE_GLM53_VOCAB_MODE="stock", MXWAVE_GLM53_KDA_MODE="stock",
                   MXWAVE_PALETTE_MODEL="0", GLM_CERT_HEAD_DUMP="")
        env.update(dict.fromkeys(DISABLED, "0"))
    env.update(ZIPSERV_CT_CONTROL_MODE=mode, VLLM_SERVER_DEV_MODE="1",
               VLLM_PLUGINS="zipserv_ct_weight_control",
               PYTHONPATH=env.get("PYTHONPATH", "") + ":/weight-control/bootstrap:/weight-control/tools")
    return env


def source_records(original):
    """Hash code/config inputs; exclude checkpoints, caches and secret mounts."""
    records = {}
    for mount in original["Mounts"]:
        dest = mount["Destination"]
        source = Path(mount["Source"])
        if dest.startswith("/run/secrets") or dest in ("/cache", "/reports", "/audit", "/experiment"):
            continue
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file():
            records[dest] = digest(source)
            continue
        if dest == "/overlay":
            source = source / "overlay"
            dest += "/overlay"
        if dest in ("/model", "/draft"):
            paths = sorted(p for p in source.iterdir() if p.is_file() and
                           p.suffix in (".json", ".jinja", ".jinja2"))
        else:
            paths = sorted(p for p in source.rglob("*") if p.is_file() and
                           p.suffix in (".py", ".so", ".json") and
                           not any(part in (".git", "__pycache__", "cache") for part in p.relative_to(source).parts))
        for path in paths:
            records[dest + "/" + path.relative_to(source).as_posix()] = digest(path)
    for item in original["Config"]["Env"]:
        key, value = item.split("=", 1)
        if not key.startswith(("MXWAVE_", "GLM_", "SPEC_PROBE_")) or not value.startswith("/"):
            continue
        for mount in sorted(original["Mounts"], key=lambda m: -len(m["Destination"])):
            dest = mount["Destination"]
            if value.startswith(dest + "/"):
                path = Path(mount["Source"]) / value.removeprefix(dest + "/")
                if path.is_file():
                    records[value] = digest(path)
                break
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--experiments-root", required=True, type=Path,
                        help="Approved parent directory for private experiment workdirs")
    parser.add_argument("--mode", choices=("off", "ct"), required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    root = args.workdir.resolve()
    if root.parent != args.experiments_root.resolve() or not root.name.startswith("zipserv-ct-weight-control-") or not re.fullmatch(r"[a-z0-9]+", args.run):
        parser.error("Require a private persistent experiment directory and alphanumeric label")
    template = f"mxwave-glm53-ct-live-20261022-r{args.rank}"
    original = json.loads(subprocess.check_output(["docker", "inspect", template]))[0]
    if original["State"]["Running"] or original["Image"] != IMAGE:
        raise RuntimeError("The original CT template must stay stopped and match its pinned image")
    if not args.preflight:
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if active.strip():
            raise RuntimeError("GPU is occupied; do not disturb another workload")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 18797 if args.rank == 0 else 29768))
    run_root = root / (args.run + "-" + args.mode)
    run_root.mkdir()
    for name in ("reports", "cache"):
        (run_root / name).mkdir()
    env = private_environment(original["Config"]["Env"], args.mode)
    command = list(original["Config"]["Cmd"])
    for flag, value in (("--host", "127.0.0.1"), ("--port", "18797"), ("--master-port", "29768"),
                        ("--worker-extension-cls", "ct_weight_control.ControlWorker")):
        if flag not in command:
            raise RuntimeError("The template is missing " + flag)
        command[command.index(flag) + 1] = value
    if any(flag.startswith("--api-key") for flag in command):
        raise RuntimeError("Do not copy inline authentication")
    mounts = []
    seed = None
    for mount in original["Mounts"]:
        dest = mount["Destination"]
        if dest.startswith("/run/secrets"):
            continue
        source = mount["Source"]
        if dest == "/cache":
            seed = source
            source = str(run_root / "cache")
        elif dest == "/reports":
            source = str(run_root / "reports")
        elif mount["RW"]:
            raise RuntimeError("Unexpected writable production mount: " + dest)
        mounts.append({"source": source, "destination": dest, "rw": mount["RW"]})
    if seed is None:
        raise RuntimeError("CT's cache seed is missing")
    sources = source_records(original)
    tool_hashes = {p.name: digest(p) for p in (root / "tools").glob("*.py")}
    if not args.preflight:
        preflight = json.loads((root / ("preflight-" + args.mode) / "launch.json").read_text())
        if (preflight.get("returncode") != 0 or preflight["sources"] != sources
                or preflight["tools"] != tool_hashes):
            raise RuntimeError("A successful preflight of these exact sources is required")
        subprocess.run(["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "2g",
                        "-v", seed + ":/seed:ro", "-v", str(run_root / "cache") + ":/destination:rw",
                        "--entrypoint", "timeout", IMAGE, "300", "cp", "-a", "--reflink=auto",
                        "/seed/.", "/destination"], check=True)
    bootstrap = root / "bootstrap" / "zipserv_ct_weight_control-0.0.0.dist-info"
    bootstrap.mkdir(parents=True, exist_ok=True)
    (bootstrap / "METADATA").write_text("Metadata-Version: 2.1\nName: zipserv-ct-weight-control\nVersion: 0.0.0\n")
    (bootstrap / "entry_points.txt").write_text("[vllm.general_plugins]\nzipserv_ct_weight_control = ct_weight_control:install\n")
    mounts.append({"source": str(root), "destination": "/weight-control", "rw": False})
    name = f"zipserv-ct-weight-{root.name.removeprefix('zipserv-ct-weight-control-')}-{args.run}-{args.mode}-r{args.rank}"
    launch = ["docker", "run", "--name", name, "--network", "host", "--ipc", "host", "--gpus", "all"]
    host = original["HostConfig"]
    for device in host.get("Devices") or []:
        launch += ["--device", f"{device['PathOnHost']}:{device['PathInContainer']}:{device['CgroupPermissions']}"]
    for capability in host.get("CapAdd") or []:
        launch += ["--cap-add", capability]
    for option in host.get("SecurityOpt") or []:
        launch += ["--security-opt", option]
    for limit in host.get("Ulimits") or []:
        launch += ["--ulimit", f"{limit['Name']}={limit['Soft']}:{limit['Hard']}"]
    for key, value in sorted(env.items()):
        launch += ["-e", key + "=" + value]
    for mount in mounts:
        launch += ["-v", f"{mount['source']}:{mount['destination']}:{'rw' if mount['rw'] else 'ro'}"]
    if args.preflight:
        launch += ["--cpus", "4", "--memory", "8g", "--entrypoint", "timeout", IMAGE, "180", "python3", "-c",
                   "import ct_weight_control as p; p.install(); p.install(); import torch; "
                   "assert torch.cuda.memory_allocated() == 0; "
                   "print('CT weight control preflight passed; no CUDA tensors allocated')"]
    else:
        launch += ["-d", "--entrypoint", "timeout", IMAGE, "21600", "vllm", *command]
    receipt = {"template": template, "name": name, "image": IMAGE, "mode": args.mode,
               "rank": args.rank, "cmd": command, "env": env, "mounts": mounts, "sources": sources,
               "preflight": args.preflight, "host_config": {k: host.get(k) for k in
                   ("NetworkMode", "IpcMode", "Devices", "CapAdd", "SecurityOpt", "Ulimits")},
               "tools": tool_hashes}
    target = run_root / "launch.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    process = subprocess.run(launch, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (run_root / "launch.log").write_text(process.stdout)
    receipt["returncode"] = process.returncode
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"name": name, "run_root": str(run_root), "returncode": process.returncode,
                      "output_tail": process.stdout[-3000:]}), flush=True)
    if process.returncode:
        raise RuntimeError(f"Trial failed with exit {process.returncode}; see {run_root / 'launch.log'}")


if __name__ == "__main__":
    main()
