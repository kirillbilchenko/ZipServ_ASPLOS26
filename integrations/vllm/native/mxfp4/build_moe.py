"""Build isolated BF16/MXFP4 grouped Marlin controls and exact scale decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def stage(
    source: Path, output: Path, variants: tuple[str, ...] = ("raw", "async"), *,
    affine_only: bool = False, shared_stage: bool = False,
) -> dict[str, object]:
    """Hash-check Apache-licensed sources and adapt only scale loading/host plumbing."""
    if (not variants or len(set(variants)) != len(variants)
            or any(v not in ("raw", "async", "prmt", "stage", "early", "bf16stage", "nibbles", "aligned", "sparse")
                   for v in variants)):
        raise ValueError("Choose unique supported native variants")
    if affine_only and "sparse" not in variants:
        raise ValueError("Affine specialization requires a sparse variant")
    if shared_stage and "sparse" not in variants:
        raise ValueError("Shared palette staging requires a sparse variant")
    package = Path(__file__).resolve().parent
    provenance = json.loads((package / "moe_sources.json").read_bytes())
    staged = output / "staged"
    staged.mkdir(parents=True, exist_ok=True)
    for relative, expected in provenance["files"].items():
        raw = (source / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Pinned grouped Marlin source changed: {relative}")
        target = staged / relative.removeprefix("csrc/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    moe = staged / "libtorch_stable/moe/marlin_moe_wna16"
    template_path = moe / "marlin_template.h"
    template = template_path.read_text()

    def replace(old: str, new: str, count: int = 1) -> None:
        nonlocal template
        if template.count(old) != count:
            raise ValueError(f"Unexpected grouped kernel patch site: {old}")
        template = template.replace(old, new)

    replace('#include "core/scalar_type.hpp"', '#include "core/scalar_type.hpp"\n'
            '#include "moe_scale_decode.cuh"')
    replace(
        "  // Each threadblock processes one",
        """  #if MXWAVE_MOE_ASYNC
  #if MXWAVE_MOE_SPARSE
  // Snapshot read-only descriptor fields once. A global descriptor pointer
  // otherwise permits repeated metadata loads in the unrolled scale decoder.
  const MxWaveMoeScales mxwave_metadata =
      *reinterpret_cast<const MxWaveMoeScales*>(scales_ptr);
  const auto* mxwave_descriptor = &mxwave_metadata;
  const uint32_t* mxwave_records = mxwave_descriptor->records;
  #else
  const auto* mxwave_descriptor = reinterpret_cast<const MxWaveMoeScales*>(scales_ptr);
  const int2* mxwave_records = mxwave_descriptor->records;
  #if MXWAVE_MOE_NIBBLES
  const uint32_t mxwave_base_pair = uint32_t(mxwave_descriptor->base) * 0x00800080u;
  #else
  const int4* mxwave_exceptions = mxwave_descriptor->exceptions;
  #endif
  #endif
  #endif
  // Each threadblock processes one""",
    )
    pointer = "scales_ptr += (expert_id - old_expert_id) * scales_expert_stride;"
    replace(
        pointer,
        "#if MXWAVE_MOE_ASYNC\n"
        "    mxwave_records += (expert_id - old_expert_id) * scales_expert_stride;\n"
        "    #else\n    " + pointer + "\n    #endif",
    )
    for name, count in (("sh_s_stage", 1), ("sh_s", 2)):
        original = f"cp_async4(&{name}[s_sh_wr], &scales_ptr[s_gl_rd]);"
        replace(
            original,
            "#if MXWAVE_MOE_SPARSE\n"
            f"mxwave_sparse_copy(&{name}[s_sh_wr], &mxwave_records[s_gl_rd]);\n"
            "#elif MXWAVE_MOE_BF16_STAGE\n"
            f"cp_async2_ca_pred(&{name}[2 * s_sh_wr], &mxwave_records[s_gl_rd]);\n"
            "#elif MXWAVE_MOE_EARLY\n"
            f"mxwave_decode_to_shared(&{name}[s_sh_wr], mxwave_records[s_gl_rd], "
            "mxwave_exceptions);\n"
            "#elif MXWAVE_MOE_ASYNC\n"
            f"cp_async2_ca_pred(&{name}[s_sh_wr], &mxwave_records[s_gl_rd]);\n"
            "#else\n" + original + "\n#endif",
            count,
        )
    read = (
        "reinterpret_cast<int2*>(&frag_s[k % 2])[0] = reinterpret_cast<int2*>(\n"
        "            sh_s_stage)[s_sh_rd + cur_group_id * (2 * s_sh_stride)];"
    )
    replace(
        read,
        "#if MXWAVE_MOE_SPARSE && !MXWAVE_MOE_SPARSE_STAGE\n"
        "{\n  const int index = s_sh_rd + cur_group_id * (2 * s_sh_stride);\n"
        "  const auto* record = reinterpret_cast<const uint32_t*>(&sh_s_stage[index >> 1]);\n"
        "  const uint32_t q = record[0];\n"
        "  reinterpret_cast<int4*>(&frag_s[k % 2])[0] =\n"
        "      mxwave_sparse_fragment<MXWAVE_MOE_AFFINE_ONLY>(mxwave_descriptor, q, index & 1);\n}\n"
        "#elif MXWAVE_MOE_NIBBLES\n"
        "{\n  const int index = s_sh_rd + cur_group_id * (2 * s_sh_stride);\n"
        "  const uint32_t q = reinterpret_cast<const uint32_t*>(\n"
        "      &sh_s_stage[index >> 1])[index & 1];\n"
        "  reinterpret_cast<int4*>(&frag_s[k % 2])[0] = mxwave_nibble_bf16(q, mxwave_base_pair);\n}\n"
        "#elif MXWAVE_MOE_BF16_STAGE\n"
        "reinterpret_cast<int4*>(&frag_s[k % 2])[0] =\n"
        "    sh_s_stage[s_sh_rd + cur_group_id * (2 * s_sh_stride)];\n"
        "#elif MXWAVE_MOE_ASYNC && !MXWAVE_MOE_STAGE && !MXWAVE_MOE_EARLY\n"
        "reinterpret_cast<int2*>(&frag_s[k % 2])[0] = mxwave_shared_half(\n"
        "    sh_s_stage, mxwave_exceptions, s_sh_rd + cur_group_id * (2 * s_sh_stride));\n"
        "#else\n" + read + "\n#endif",
    )
    replace("constexpr int sh_s_size = stages * s_sh_stage;",
            "constexpr int sh_s_size = stages * s_sh_stage * (1 + MXWAVE_MOE_BF16_STAGE);")
    replace("int4* sh_s_stage = sh_s + s_sh_stage * pipe;",
            "int4* sh_s_stage = sh_s + s_sh_stage * pipe * (1 + MXWAVE_MOE_BF16_STAGE);", 2)
    copy_half = ("reinterpret_cast<int2*>(&frag_s[1])[0] =\n"
                 "            reinterpret_cast<int2*>(&frag_s[0])[0];")
    replace(copy_half, "#if MXWAVE_MOE_BF16_STAGE || MXWAVE_MOE_NIBBLES || "
            "(MXWAVE_MOE_SPARSE && !MXWAVE_MOE_SPARSE_STAGE)\n"
            "reinterpret_cast<int4*>(&frag_s[1])[0] = reinterpret_cast<int4*>(&frag_s[0])[0];\n"
            "#else\n" + copy_half + "\n#endif")
    replace("if constexpr (s_type == vllm::kFE4M3fn || s_type == vllm::kFE8M0fnu) {",
            "if constexpr ((s_type == vllm::kFE4M3fn || s_type == vllm::kFE8M0fnu) &&\n"
            "                  !MXWAVE_MOE_BF16_STAGE && !MXWAVE_MOE_NIBBLES &&\n"
            "                  !(MXWAVE_MOE_SPARSE && !MXWAVE_MOE_SPARSE_STAGE)) {")
    replace("        fetch_scales_to_registers(k + 1, pipe);",
            """        #if MXWAVE_MOE_SPARSE && !MXWAVE_MOE_SPARSE_STAGE
        // The final speculative register load has no following MMA. Its shared
        // slot may never have been filled. Unlike inline scale bits, an escape
        // record would dereference that uninitialized value as a global index.
        if (k + 1 < b_sh_wr_iters || slice_iters > 1) {
          fetch_scales_to_registers(k + 1, pipe);
        }
        #else
        fetch_scales_to_registers(k + 1, pipe);
        #endif""")
    replace("auto wait_for_stage = [&]() {",
            "auto wait_for_stage = [&](int ready_pipe, bool valid_stage = true) {")
    replace(
        "    cp_async_wait<stages - 2>();\n    __syncthreads();",
        """    cp_async_wait<stages - 2>();
    #if MXWAVE_MOE_STAGE
    // Each issuing thread waits for its own 8-byte copy, expands its unique
    // 16-byte slot, then the existing barrier publishes it to every consumer.
    // Do not reinterpret a stale, already expanded slot while winding down.
    if (valid_stage && s_sh_wr_pred) {
      int4* destination = sh_s + s_sh_stage * ready_pipe * (1 + MXWAVE_MOE_BF16_STAGE);
      const int slot = s_sh_wr * (1 + MXWAVE_MOE_BF16_STAGE);
      #if MXWAVE_MOE_SPARSE_STAGE
      const uint32_t record = *reinterpret_cast<const uint32_t*>(&destination[slot]);
      destination[slot] = mxwave_sparse_raw(mxwave_descriptor, record);
      #else
      const int2 record = *reinterpret_cast<const int2*>(&destination[slot]);
      #if MXWAVE_MOE_BF16_STAGE
      mxwave_decode_bf16_to_shared(destination + slot, record, mxwave_exceptions);
      #else
      const int2 lo = mxwave_record_half(record, mxwave_exceptions, 0);
      const int2 hi = mxwave_record_half(record, mxwave_exceptions, 1);
      destination[slot] = make_int4(lo.x, lo.y, hi.x, hi.y);
      #endif
      #endif
    }
    #endif
    __syncthreads();""",
    )
    replace("    wait_for_stage();", "    wait_for_stage(0);", 2)
    # The second call is inside the unrolled loop, where pipe already points
    # at the next stage and slice_iters still includes the current tile.
    replace("          wait_for_stage(0);",
            "          wait_for_stage(pipe % stages, slice_iters > 1);")
    template_path.write_text(template)
    shutil.copyfile(package / "moe_scale_decode.cuh", staged / "moe_scale_decode.cuh")
    shutil.copyfile(package / "moe_sparse_scales.cuh", staged / "moe_sparse_scales.cuh")
    selectors = []
    for blocks, small in ((1, True), (1, False), (2, False), (3, False), (4, False)):
        for k, n, threads in ((128, 128, 256), (64, 256, 256), (64, 128, 128), (128, 64, 128)):
            if threads == 256 and ((blocks == 1) != (k == 128)):
                continue
            selectors.append(
                f"if (thread_m_blocks == {blocks} && thread_k_blocks == {k // 16} && "
                f"thread_n_blocks == {n // 16} && threads == {threads} && "
                f"m_block_size_8 == {'true' if small else 'false'}) "
                "kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), "
                f"vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(), {threads}, {blocks}, "
                f"{n // 16}, {k // 16}, {'true' if small else 'false'}, 4, 2, false>;"
            )
    (moe / "kernel_selector.h").write_text("\n".join(selectors) + "\n")
    original = (moe / "ops.cu").read_text()
    license_notice = original[: original.index('#include "kernel.h"')]
    start = original.index("typedef struct {")
    end = original.index("}  // namespace MARLIN_NAMESPACE_NAME", start)
    host = license_notice + """
#include <algorithm>
#include <stdexcept>
#include <string>
#include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"
#include "libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"
using std::min;
using std::max;
namespace MARLIN_NAMESPACE_NAME {
__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS) {}
using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);
""" + original[start:end] + "\n}\n" + (package / "moe_bridge.cu").read_text()
    (staged / "host.cu").write_text(host)
    (output / "CMakeLists.txt").write_text("""cmake_minimum_required(VERSION 3.26)
project(mxwave_marlin_moe_scales LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 121-real)
set(MXWAVE_SPARSE_AFFINE """ + str(int(affine_only)) + """ )
set(MXWAVE_SPARSE_STAGE """ + str(int(shared_stage)) + """ )
foreach(variant """ + " ".join(variants) + """ )
  add_library(mxwave_moe_${variant} SHARED staged/host.cu)
  target_include_directories(mxwave_moe_${variant} PRIVATE
    staged staged/libtorch_stable/moe/marlin_moe_wna16 ${MXWAVE_TORCH_INCLUDE})
  target_compile_options(mxwave_moe_${variant} PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--expt-extended-lambda;--expt-relaxed-constexpr;-lineinfo;-Xptxas=-v>")
  target_compile_definitions(mxwave_moe_${variant} PRIVATE
    MARLIN_NAMESPACE_NAME=mxwave_grouped_${variant})
  if(variant STREQUAL raw)
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_ASYNC=0)
  else()
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_ASYNC=1)
  endif()
  if(variant STREQUAL prmt OR variant STREQUAL stage OR variant STREQUAL early OR variant STREQUAL bf16stage)
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_PRMT=1)
  else()
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_PRMT=0)
  endif()
  if(variant STREQUAL stage OR variant STREQUAL bf16stage OR
     (variant STREQUAL sparse AND MXWAVE_SPARSE_STAGE))
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_STAGE=1)
  else()
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_STAGE=0)
  endif()
  if(variant STREQUAL bf16stage)
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_BF16_STAGE=1)
  else()
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_BF16_STAGE=0)
  endif()
  if(variant STREQUAL early)
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_EARLY=1)
  else()
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_EARLY=0)
  endif()
  if(variant STREQUAL nibbles OR variant STREQUAL aligned)
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_NIBBLES=1)
  else()
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_NIBBLES=0)
  endif()
  if(variant STREQUAL aligned)
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_ALIGNED_NIBBLES=1)
  else()
    target_compile_definitions(mxwave_moe_${variant} PRIVATE MXWAVE_MOE_ALIGNED_NIBBLES=0)
  endif()
  if(variant STREQUAL sparse)
    target_compile_definitions(mxwave_moe_${variant} PRIVATE
      MXWAVE_MOE_SPARSE=1 MXWAVE_MOE_AFFINE_ONLY=${MXWAVE_SPARSE_AFFINE}
      MXWAVE_MOE_SPARSE_STAGE=${MXWAVE_SPARSE_STAGE})
  else()
    target_compile_definitions(mxwave_moe_${variant} PRIVATE
      MXWAVE_MOE_SPARSE=0 MXWAVE_MOE_AFFINE_ONLY=0 MXWAVE_MOE_SPARSE_STAGE=0)
  endif()
endforeach()
""")
    manifest: dict[str, object] = {
        "format": "mxwave-grouped-marlin-scales-build-v1",
        "source": provenance,
        "architecture": "SM121",
        "activations": "BF16",
        "weights": "MXFP4, E8M0 scales, group32",
        "kernel_configurations_per_variant": len(selectors),
        "variants": variants,
        "sparse_palette_policy": "affine_only" if affine_only else "generic",
        "sparse_shared_stage": shared_stage,
        "staged_files_sha256": {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(staged.rglob("*")) if p.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Build external component libraries without modifying installed vLLM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-include", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--affine-only", action="store_true")
    parser.add_argument("--shared-stage", action="store_true",
                        help="Expand palette records once into the existing U8 shared-memory slots")
    parser.add_argument("--variants", nargs="+",
                        choices=("raw", "async", "prmt", "stage", "early", "bf16stage",
                                 "nibbles", "aligned", "sparse"),
                        default=("raw", "async"))
    args = parser.parse_args()
    manifest = stage(args.source, args.output, tuple(args.variants),
                     affine_only=args.affine_only, shared_stage=args.shared_stage)
    if not args.stage_only:
        if args.torch_include is None:
            parser.error("Build requires --torch-include")
        subprocess.run(
            ["cmake", "-S", str(args.output), "-B", str(args.output / "build"),
             "-G", "Ninja", f"-DMXWAVE_TORCH_INCLUDE={args.torch_include}"], check=True
        )
        subprocess.run(["cmake", "--build", str(args.output / "build"), "--parallel", "1"],
                       check=True)
        manifest["libraries_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((args.output / "build").glob("*.so"))
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "staged" if args.stage_only else "built"}))


if __name__ == "__main__":
    main()
