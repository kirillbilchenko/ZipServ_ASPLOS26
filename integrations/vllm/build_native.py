# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Build the Torch-ABI-independent CPU or CUDA bridge and record source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Build a CPU-only encoder or select an explicit CUDA architecture."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--arch", choices=["80", "86", "89", "90", "100", "120", "121"])
    mode.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experiment-popcount", action="store_true",
                        help="Measure popcount instead of the GEMM decoder's total-count shuffle")
    parser.add_argument("--experiment-shmem-once", action="store_true",
                        help="Set per-device/kernel shared-memory limits once; no CUDA context resets")
    args = parser.parse_args()
    if args.cpu_only and (args.experiment_popcount or args.experiment_shmem_once):
        parser.error("Kernel experiments require a CUDA build")
    root = Path(__file__).resolve().parents[2]
    native = root / "integrations/vllm/native"
    host = native / "host_bridge.cpp"
    if args.cpu_only:
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        default_output = Path("build/libzipserv_cpu" + suffix)
        command = [
            *compiler,
            "-std=c++17",
            "-O3",
            "-dynamiclib" if sys.platform == "darwin" else "-shared",
            "-fPIC",
            "-pthread",
            str(host),
        ]
        sources = [
            host,
            native / "host_bridge.h",
            root / "csrc/HostPacker.h",
            root / "csrc/CodecUtils.h",
        ]
    else:
        cuda_path = os.environ.get("CUDA_PATH")
        compiler = [
            str(Path(cuda_path) / "bin/nvcc")
            if cuda_path
            else (shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
        ]
        default_output = Path("build/libzipserv_vllm.so")
        command = [
            *compiler,
            "-std=c++17",
            "-O3",
            "--shared",
            "-Xcompiler=-fPIC,-pthread",
            f"-gencode=arch=compute_{args.arch},code=sm_{args.arch}",
            str(root / "csrc/L_API.cu"),
            str(host),
            str(native / "bridge.cu"),
        ]
        sources = sorted((root / "csrc").glob("*")) + [
            root / "build/L_API.cuh",
            host,
            native / "host_bridge.h",
            native / "bridge.cu",
        ]
    output = (args.output or default_output).resolve()
    if args.experiment_popcount:
        command.append("-DZIPSERV_EXPERIMENT_POPCOUNT=1")
    if args.experiment_shmem_once:
        command.append("-DZIPSERV_EXPERIMENT_SHMEM_ONCE=1")
    output.parent.mkdir(parents=True, exist_ok=True)
    command += ["-o", str(output)]
    subprocess.run(command, check=True)
    manifest = {
        "abi_version": 2,
        "cuda": not args.cpu_only,
        "arch": args.arch,
        "experiments": {"popcount": args.experiment_popcount,
                        "shmem_once": args.experiment_shmem_once},
        "fp32_partial_symbol": not args.cpu_only,
        "command": command,
        "compiler": subprocess.check_output([*compiler, "--version"], text=True),
        "sources_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
            if path.is_file() and not path.name.startswith(".")
        },
        "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
