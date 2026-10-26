"""Build isolated, source-identical CT controls and an optional FP4 scale expansion.

Requires the explicitly supplied, hash-pinned MxWave builder and vLLM source
files. Neither installed packages nor the external source tree are modified.
See FP4_SCALE_NOTICE for source attribution and the supported experiment scope.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


CT_FILES = {
    "build_moe.py": "739b9369852f870003ba6abdc1ec96ada2fba318417886234c7f343b3efdd574",
    "moe_bridge.cu": "cbc93a9737cc9f376cf4a0d96f4e70744327f412c1a7ced10daaf7e795584597",
    "moe_scale_decode.cuh": "83aaf867c5190fd36206c99d41f31392a0f6fd1cb84853ea64dad8120ac86c71",
    "moe_sparse_scales.cuh": "cd1eabe99c2ef423839eabc38b5d3fca995ede06dfb2848a12f82800f47b1296",
    "moe_sources.json": "97972b9d24a08db12f4f759f904314262141eea63ad4a977eff34e088d0c8c10",
}
CT_LIBRARY_SHA256 = "9f5d96dcb0762d7e5ed5436ca296f70d590e73797d40522e4d8c7d6f3902f5f0"


def digest(path: Path) -> str:
    """Hash one input without importing or executing it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Verify the frozen baseline, then build selected opt-in expansion variants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct-source", type=Path, required=True)
    parser.add_argument("--ct-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-include", type=Path)
    parser.add_argument("--expansions", nargs="+", choices=("palette", "affine"),
                        default=["palette", "affine"])
    parser.add_argument("--tail-prefetch", choices=("skip", "keep", "barrier"), default="barrier",
                        help="Guard the unused read, retain original behavior, or synchronize reuse")
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous evidence; output must be a new directory")
    if len(set(args.expansions)) != len(args.expansions):
        parser.error("Choose unique expansion variants")
    if not args.stage_only and args.torch_include is None:
        parser.error("Native build requires --torch-include")
    for name, expected in CT_FILES.items():
        if digest(args.ct_source / name) != expected:
            raise ValueError(f"Pinned CT source changed: {name}")
    baseline = json.loads(args.ct_manifest.read_bytes())
    if (baseline["libraries_sha256"]["libmxwave_moe_sparse.so"] != CT_LIBRARY_SHA256
            or baseline["sparse_palette_policy"] != "affine_only"
            or baseline["sparse_shared_stage"] is not True):
        raise ValueError("Require the qualified CT sparse shared-U8 baseline")
    spec = importlib.util.spec_from_file_location("zipserv_pinned_ct_builder",
                                                args.ct_source / "build_moe.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load verified external CT builder")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    args.output.mkdir(parents=True)
    report = {"format": "zipserv-fp4-scale-expansion-v1", "status": "staging",
              "ct_source_sha256": CT_FILES, "ct_manifest_sha256": digest(args.ct_manifest),
              "ct_library_sha256": CT_LIBRARY_SHA256, "variants": {},
              "scope": "MXFP4 E8M0 group32; CT sparse-palette shared-U8 decoder only",
              "builder_sha256": digest(Path(__file__)),
              "extra_weight_or_scale_storage_bytes": 0,
              "tail_prefetch": args.tail_prefetch,
              "serving_plugin_installed": False}
    if not args.stage_only:
        report["nvcc"] = subprocess.check_output(["nvcc", "--version"], text=True)
    for mode in args.expansions:
        directory = args.output / mode
        staged_report = builder.stage(args.source, directory, ("sparse",),
                                      affine_only=True, shared_stage=True)
        if staged_report["staged_files_sha256"] != baseline["staged_files_sha256"]:
            raise ValueError("Recreated CT sources do not match the qualified baseline")
        staged = directory / "staged"
        if args.tail_prefetch == "skip":
            path = staged / "libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"
            original = path.read_text()
            site = "        fetch_to_registers(k + 1, pipe % stages);"
            if original.count(site) != 1:
                raise ValueError("Pinned final-register-prefetch site changed")
            replacement = ("        // The last speculative load has no following MMA. Skipping it\n"
                           "        // also prevents a dead shared-B read racing shared reduction.\n"
                           "        if (k + 1 < b_sh_wr_iters || slice_iters > 1) {\n"
                           + site + "\n        }")
            updated = original.replace(site, replacement)
            path.write_text(updated)
            (directory / "tail-prefetch.patch").write_text("".join(difflib.unified_diff(
                original.splitlines(True), updated.splitlines(True),
                fromfile="ct/marlin_template.h", tofile=mode + "/marlin_template.h")))
        elif args.tail_prefetch == "barrier":
            path = staged / "libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"
            original = path.read_text()
            site = "    if (red_off >= 1) {\n"
            if original.count(site) != 1:
                raise ValueError("Pinned shared-memory reduction site changed")
            updated = original.replace(site, site
                + "      // Finish every warp's shared-B reads before reusing that storage.\n"
                + "      __syncthreads();\n")
            path.write_text(updated)
            (directory / "shared-reuse.patch").write_text("".join(difflib.unified_diff(
                original.splitlines(True), updated.splitlines(True),
                fromfile="ct/marlin_template.h", tofile=mode + "/marlin_template.h")))
        if mode == "affine":
            helper = Path(__file__).with_name("fp4_scale_expansion.cuh")
            shutil.copy2(helper, staged / helper.name)
            path = staged / "moe_sparse_scales.cuh"
            original = path.read_text()
            site = ("__device__ __forceinline__ int2 mxwave_sparse_raw_half("
                    "uint32_t palette, uint32_t q) {\n")
            if original.count(site) != 1:
                raise ValueError("Pinned CT expansion site changed")
            replacement = ('#include "fp4_scale_expansion.cuh"\n\n' + site
                           + "  if (palette & (1u << 24)) return zipserv_fp4_affine_half(palette, q);\n")
            updated = original.replace(site, replacement)
            path.write_text(updated)
            (directory / "scale-expansion.patch").write_text("".join(difflib.unified_diff(
                original.splitlines(True), updated.splitlines(True),
                fromfile="ct/moe_sparse_scales.cuh", tofile="affine/moe_sparse_scales.cuh")))
        host = staged / "host.cu"
        with host.open("a") as stream:
            stream.write('\nextern "C" int zipserv_fp4_scale_expansion() { return '
                         + str(int(mode == "affine")) + '; }\n')
            stream.write('extern "C" int zipserv_fp4_skip_unused_prefetch() { return '
                         + str(int(args.tail_prefetch == "skip")) + '; }\n')
            stream.write('extern "C" int zipserv_fp4_tail_policy() { return '
                         + str({"keep": 0, "skip": 1, "barrier": 2}[args.tail_prefetch]) + '; }\n')
        manifest = {"baseline_sources_verified": True,
                    "staged_sha256": {str(p.relative_to(directory)): digest(p)
                                      for p in sorted(staged.rglob("*")) if p.is_file()}}
        if not args.stage_only:
            library = directory / "libzipserv_fp4_scales.so"
            defines = {"MARLIN_NAMESPACE_NAME": "mxwave_grouped_sparse", "MXWAVE_MOE_ASYNC": 1,
                       "MXWAVE_MOE_PRMT": 0, "MXWAVE_MOE_STAGE": 1, "MXWAVE_MOE_BF16_STAGE": 0,
                       "MXWAVE_MOE_EARLY": 0, "MXWAVE_MOE_NIBBLES": 0,
                       "MXWAVE_MOE_ALIGNED_NIBBLES": 0, "MXWAVE_MOE_SPARSE": 1,
                       "MXWAVE_MOE_AFFINE_ONLY": 1, "MXWAVE_MOE_SPARSE_STAGE": 1}
            command = ["nvcc", "-std=c++17", "-O3", "-arch=sm_121", "--shared", "-Xcompiler=-fPIC",
                       "--use_fast_math", "--expt-extended-lambda", "--expt-relaxed-constexpr",
                       "-lineinfo", "-Xptxas=-v", "-I" + str(staged),
                       "-I" + str(staged / "libtorch_stable/moe/marlin_moe_wna16"),
                       "-I" + str(args.torch_include),
                       *[f"-D{key}={value}" for key, value in defines.items()],
                       str(host), "-o", str(library)]
            with (directory / "build.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            manifest.update(command=command, library_sha256=digest(library))
        report["variants"][mode] = manifest
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"variant": mode, "status": "staged" if args.stage_only else "built"}),
              flush=True)
    report["status"] = "staged" if args.stage_only else "built"
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
