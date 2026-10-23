# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Build this checkout's Torch-ABI-independent CUDA bridge and record source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    """Compile a library for an explicit GPU architecture without patching dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", required=True, choices=["80", "86", "89", "90", "100", "120", "121"]
    )
    parser.add_argument("--output", type=Path, default=Path("build/libzipserv_vllm.so"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cuda_path = os.environ.get("CUDA_PATH")
    compiler = (
        str(Path(cuda_path) / "bin/nvcc")
        if cuda_path
        else (shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bridge = root / "integrations/vllm/native/bridge.cu"
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "--shared",
        "-Xcompiler=-fPIC",
        f"-gencode=arch=compute_{args.arch},code=sm_{args.arch}",
        str(root / "csrc/L_API.cu"),
        str(bridge),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    sources = sorted((root / "csrc").glob("*")) + [root / "build/L_API.cuh", bridge]
    manifest = {
        "abi_version": 2,
        "arch": args.arch,
        "command": command,
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
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
