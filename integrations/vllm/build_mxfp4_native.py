# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Build the standalone ZipServ MXFP4 scale consumer from pinned Marlin sources."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path: Path) -> str:
    """Return the exact source or library identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Stage licensed source, fix shared-memory reuse, and build an explicit architecture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Hash-pinned vLLM source checkout")
    parser.add_argument("--output", type=Path, required=True, help="New build directory")
    parser.add_argument("--arch", choices=("80", "86", "89", "90", "100", "120", "121"), required=True)
    parser.add_argument("--torch-include", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory; preserve previous build identities")
    package = Path(__file__).resolve().parent
    vendor = package / "native/mxfp4"
    provenance = json.loads((package / "zipserv_vllm/mxfp4-provenance.json").read_bytes())
    for name, expected in provenance["native_files_sha256"].items():
        if digest(vendor / name) != expected:
            raise ValueError(f"Vendored CT source differs from its recorded provenance: {name}")
    spec = importlib.util.spec_from_file_location("zipserv_mxfp4_stager", vendor / "build_moe.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the bundled source stager")
    stager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stager)
    original = stager.stage(args.source.resolve(), args.output.resolve(), ("sparse",),
                            affine_only=False, shared_stage=True)
    staged = args.output.resolve() / "staged"
    template = staged / "libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"
    source = template.read_text()
    site = "    if (red_off >= 1) {\n"
    if source.count(site) != 1:
        raise ValueError("Pinned shared-memory reduction changed")
    source = source.replace(site, site
        + "      // ZipServ: finish shared-B reads before reduction reuses the same storage.\n"
        + "      __syncthreads();\n")
    template.write_text(source)
    host = staged / "host.cu"
    with host.open("a") as stream:
        stream.write('\nextern "C" int zipserv_mxfp4_abi() { return 1; }\n'
                     'extern "C" int zipserv_mxfp4_features() { return 7; }\n')
    manifest = {"format": "zipserv-mxfp4-native-v1", "abi_version": 1, "features": 7,
                "architecture": "SM" + args.arch, "gpu_validation_scope": "SM121 only",
                "representation": "sparse_palette", "scale_dtype": "E8M0", "group_size": 32,
                "shared_u8_stage": True, "generic_palette": True, "shared_reuse_barrier": True,
                "vllm_python_version": "0.32.0", "upstream": original["source"],
                "vendor_source_sha256": provenance["native_files_sha256"],
                "builder_sha256": digest(Path(__file__)),
                "source_sha256": {str(p.relative_to(args.output.resolve())): digest(p)
                                  for p in sorted(staged.rglob("*")) if p.is_file()}}
    if not args.stage_only:
        include = args.torch_include
        if include is None:
            import torch
            include = Path(torch.__file__).parent / "include"
        cuda_path = os.environ.get("CUDA_PATH")
        compiler = str(Path(cuda_path) / "bin/nvcc") if cuda_path else shutil.which("nvcc")
        if compiler is None:
            raise RuntimeError("CUDA nvcc compiler is required")
        library = args.output.resolve() / "libzipserv_mxfp4.so"
        definitions = {"MARLIN_NAMESPACE_NAME": "zipserv_mxfp4", "MXWAVE_MOE_ASYNC": 1,
                       "MXWAVE_MOE_PRMT": 0, "MXWAVE_MOE_STAGE": 1, "MXWAVE_MOE_BF16_STAGE": 0,
                       "MXWAVE_MOE_EARLY": 0, "MXWAVE_MOE_NIBBLES": 0,
                       "MXWAVE_MOE_ALIGNED_NIBBLES": 0, "MXWAVE_MOE_SPARSE": 1,
                       "MXWAVE_MOE_AFFINE_ONLY": 0, "MXWAVE_MOE_SPARSE_STAGE": 1}
        command = [compiler, "-std=c++17", "-O3", "-arch=sm_" + args.arch, "--shared",
                   "-Xcompiler=-fPIC", "--use_fast_math", "--expt-extended-lambda",
                   "--expt-relaxed-constexpr", "-lineinfo", "-Xptxas=-v", "-I" + str(staged),
                   "-I" + str(staged / "libtorch_stable/moe/marlin_moe_wna16"),
                   "-I" + str(include), *[f"-D{k}={v}" for k, v in definitions.items()],
                   str(host), "-o", str(library)]
        with (args.output / "build.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest.update(command=command, compiler=subprocess.check_output([compiler, "--version"], text=True),
                        library_sha256=digest(library))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "staged" if args.stage_only else "built", "output": str(args.output)}))


if __name__ == "__main__":
    main()
