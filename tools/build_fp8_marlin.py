"""Stage/build isolated raw, ZFR1 and ZFA1 Marlin variants from pinned vLLM 0.32.

Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
No installed vLLM files are changed. Fetching requires an explicit --fetch;
staging, source checks and format tests need neither CUDA nor PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

HERE = Path(__file__).resolve().parent
PROVENANCE = HERE / "marlin_fp8_sources.json"
MARLIN = "csrc/quantization/marlin/"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(source: Path) -> None:
    """Fetch only the explicitly pinned public source files, checking every hash."""
    pin = json.loads(PROVENANCE.read_text())
    for relative, expected in pin["files"].items():
        path = source / relative
        if path.exists():
            if sha(path.read_bytes()) != expected:
                raise ValueError(f"Existing source differs from pin: {relative}")
            continue
        url = f"https://raw.githubusercontent.com/vllm-project/vllm/{pin['revision']}/{relative}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if sha(data) != expected:
            raise ValueError(f"Downloaded source differs from pin: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Pinned Marlin patch site changed: {old[:100]}")
    return text.replace(old, new)


def selector() -> str:
    """Same 15 BF16/MXFP8/group32 specializations as the pinned generator."""
    lines = ["// Isolated BF16 / E4M3 / E8M0 / group32 dispatch only."]
    for m, small in ((1, True), (1, False), (2, False), (3, False), (4, False)):
        for k, n, threads in ((128, 128, 256), (64, 256, 256),
                              (64, 128, 128), (128, 64, 128)):
            if threads == 256 and (k, n) != ((128, 128) if m == 1 else (64, 256)):
                continue
            flag = str(small).lower()
            condition = ("a_type == vllm::kBFloat16 && b_type == vllm::kFE4M3fn && "
                         "c_type == vllm::kBFloat16 && s_type == vllm::kFE8M0fnu && "
                         "!has_zp && !is_zp_float && group_blocks == 2 && stages == 4 && "
                         f"threads == {threads} && thread_m_blocks == {m} && "
                         f"thread_n_blocks == {n // 16} && thread_k_blocks == {k // 16} && "
                         f"m_block_size_8 == {flag}")
            lines.append(f"if ({condition}) return Marlin<vllm::kBFloat16.id(), "
                         "vllm::kFE4M3fn.id(), vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(), "
                         f"{threads}, {m}, {n // 16}, {k // 16}, {flag}, 4, 2, false>;")
    return "\n".join(lines) + "\n"


def stage(source: Path, output: Path, arch: str = "121",
          variants: tuple[str, ...] = ("raw", "range16"),
          async_decode: str = "bytes", async_exceptions: str = "split",
          exact_scale_cache: bool = False, async_layout: str = "packed7",
          async_tile_policy: str = "mixed", async_exception_loads: str = "predicated") -> dict:
    """Check pins before changing weight loads and optional bit conversion only."""
    if not re.fullmatch(r"(?:8[06-9]|9[0]|1[012][0-9])", arch):
        raise ValueError("Choose a numeric SM80+ target, for example 80, 89, 90, 120 or 121")
    if not variants or len(set(variants)) != len(variants) or set(variants) - {"raw", "range16", "async"}:
        raise ValueError("Choose unique raw/range16/async variants")
    if async_decode not in ("bytes", "direct", "producer"):
        raise ValueError("Choose bytes, direct or producer async decoding")
    if async_exceptions not in ("split", "uniform") or (async_decode == "direct" and async_exceptions != "split"):
        raise ValueError("Choose split/uniform exceptions; uniform requires bytes or producer decoding")
    if async_layout not in ("packed7", "byteplane") or (async_decode == "direct" and async_layout != "packed7"):
        raise ValueError("Choose packed7/byteplane; byteplane requires bytes or producer decoding")
    if async_tile_policy not in ("mixed", "compact-only") or (async_tile_policy == "compact-only" and async_decode != "producer"):
        raise ValueError("Choose mixed/compact-only tiles; compact-only requires producer decoding")
    compact_only = async_tile_policy == "compact-only"
    if async_exception_loads not in ("predicated", "indexed") or (
            async_exception_loads == "indexed" and (async_exceptions != "uniform" or async_decode == "direct")):
        raise ValueError("Indexed exception loads require uniform byte/producer decoding")
    source, output = source.resolve(), output.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and output must be separate directories")
    pin = json.loads(PROVENANCE.read_text())
    inputs = {relative: (source / relative).read_bytes() for relative in pin["files"]}
    for relative, data in inputs.items():
        if sha(data) != pin["files"][relative]:
            raise ValueError(f"Pinned vLLM source changed: {relative}")
    template = inputs[MARLIN + "marlin_template.h"].decode()
    template = replace_once(template, '#include "core/scalar_type.hpp"',
                            '#include "core/scalar_type.hpp"\n'
                            '// ZipServ modification: optional lossless weight loading.\n'
                            '#if ZIPSERV_MARLIN_RANGE16\n'
                            '#include "fp8_marlin_decode.cuh"\n#endif\n'
                            '#if ZIPSERV_MARLIN_ASYNC\n#include "fp8_async_decode.cuh"\n#endif')
    template = replace_once(template, '  // Each threadblock processes one "stripe"',
                            '#if ZIPSERV_MARLIN_RANGE16 || ZIPSERV_MARLIN_ASYNC\n'
                            '  static_assert(a_type_id == vllm::kBFloat16.id() &&\n'
                            '                b_type_id == vllm::kFE4M3fn.id() &&\n'
                            '                s_type_id == vllm::kFE8M0fnu.id() && group_blocks == 2);\n'
                            '  const auto* zfr_data = reinterpret_cast<const uint8_t*>(B);\n'
                            '  const uint32_t zfr_base = zfr_data[4];\n#endif\n'
                            '  // Each threadblock processes one "stripe"')
    original_copy = '        cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_idx]);'
    template = replace_once(template, original_copy,
                            '#if ZIPSERV_MARLIN_ASYNC\n'
                            '        zipserv_async_copy(sh_b_stage, zfa_flags + pipe * (b_sh_stage / 64),\n'
                            '                           threads * i + threadIdx.x, zfr_data, b_gl_idx);\n'
                            '#elif ZIPSERV_MARLIN_RANGE16\n'
                            '        zipserv_range16_copy(&sh_b_stage[threads * i + threadIdx.x],\n'
                            '                             zfr_data, b_gl_idx, zfr_base);\n'
                            '#else\n' + original_copy + '\n#endif')
    copy_start = '''      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < (b_sh_wr_iters * b_thread_vecs); i++) {'''
    template = replace_once(template, copy_start, '''      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#if ZIPSERV_MARLIN_ASYNC && ZIPSERV_MARLIN_ASYNC_PRODUCER
      // Preserve each destination's native global address, but assign complete
      // 1024-byte tiles to warps so expansion needs no extra CTA barrier.
      const int origin = b_gl_rd - b_gl_stride * (threadIdx.x / b_sh_stride)
                         - (threadIdx.x % b_sh_stride);
  #pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        const int local = 2 * threads * i + (threadIdx.x / 32) * 64 + threadIdx.x % 32;
        const int global = origin + b_gl_stride * (local / b_sh_stride)
                           + local % b_sh_stride;
        zipserv_async_copy_tile_lane(sh_b_stage, zfa_flags + pipe * (b_sh_stage / 64),
                                     local, zfr_data, global);
      }
#else
  #pragma unroll
      for (int i = 0; i < (b_sh_wr_iters * b_thread_vecs); i++) {''')
    template = replace_once(template, '      b_gl_rd += b_gl_rd_delta_o;',
                            '#endif\n      b_gl_rd += b_gl_rd_delta_o;')
    template = replace_once(template, '  int4* sh_a = sh_s + sh_s_size;',
                            '  int4* sh_a = sh_s + sh_s_size;\n'
                            '#if ZIPSERV_MARLIN_ASYNC\n'
                            '  static_assert(b_thread_vecs == 2 && b_sh_stage % 64 == 0);\n'
                            '  auto* zfa_flags = reinterpret_cast<uint32_t*>(sh_a + stages * a_sh_stage);\n'
                            '#endif')
    template = replace_once(template, '  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {',
                            '  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {\n'
                            '#if ZIPSERV_MARLIN_ASYNC\n'
                            '    // A tail prefetch may be consumed speculatively but never used for math.\n'
                            '    // Mark its tile raw, so uninitialized exception metadata is never read.\n'
                            '    if (!pred && threadIdx.x < b_sh_stage / 64)\n'
                            '      zfa_flags[pipe * (b_sh_stage / 64) + threadIdx.x] = 1;\n'
                            '#endif')
    read = '''  #pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) {
      frag_b_quant[k % 2][i] = *reinterpret_cast<I4*>(
          &sh_b_stage[b_sh_stride * (k % b_sh_wr_iters) + b_sh_rd + i]);
    }'''
    template = replace_once(template, read,
                            '#if ZIPSERV_MARLIN_ASYNC && ZIPSERV_MARLIN_ASYNC_DIRECT\n'
                            '    zfa_raw[k % 2] = zipserv_async_read_packed(\n'
                            '        frag_b_quant[k % 2][0], frag_b_quant[k % 2][1],\n'
                            '        sh_b_stage, zfa_flags + pipe * (b_sh_stage / 64),\n'
                            '        b_sh_stride * (k % b_sh_wr_iters) + b_sh_rd);\n'
                            '#elif ZIPSERV_MARLIN_ASYNC && !ZIPSERV_MARLIN_ASYNC_PRODUCER\n'
                            '    zipserv_async_read(frag_b_quant[k % 2][0], frag_b_quant[k % 2][1],\n'
                            '        sh_b_stage, zfa_flags + pipe * (b_sh_stage / 64),\n'
                            '        b_sh_stride * (k % b_sh_wr_iters) + b_sh_rd, zfr_base);\n'
                            '#else\n' + read + '\n#endif')
    template = replace_once(template, '  auto wait_for_stage = [&]() {',
                            '  auto wait_for_stage = [&](int ready_pipe, bool valid_stage = true) {')
    template = replace_once(template, '    cp_async_wait<stages - 2>();\n    __syncthreads();', '''    cp_async_wait<stages - 2>();
#if ZIPSERV_MARLIN_ASYNC && ZIPSERV_MARLIN_ASYNC_PRODUCER
    // Do not expand the final speculative read: its stage can already contain
    // expanded bytes, which are no longer valid compressed metadata.
    if (valid_stage) {
  #pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        const int tile = (threadIdx.x / 32) * 64 + i * threads * 2;
        zipserv_async_expand_tile(sh_b + ready_pipe * b_sh_stage,
                                  zfa_flags + ready_pipe * (b_sh_stage / 64), tile, zfr_base);
      }
    }
#endif
    __syncthreads();''')
    template = replace_once(template, '    wait_for_stage();\n    fetch_to_registers(0, 0);',
                            '    wait_for_stage(0);\n    fetch_to_registers(0, 0);')
    template = replace_once(template, '          wait_for_stage();',
                            '          wait_for_stage(pipe % stages, slice_iters > 1);')
    template = replace_once(template, '  I4 frag_b_quant[2][b_thread_vecs];',
                            '  I4 frag_b_quant[2][b_thread_vecs];\n'
                            '#if ZIPSERV_MARLIN_ASYNC && ZIPSERV_MARLIN_ASYNC_DIRECT\n'
                            '  static_assert(!dequant_skip_flop);\n'
                            '  bool zfa_raw[2];\n#endif')
    conversion_start = '      int b_quant_0, b_quant_1;'
    conversion_end = '      dequant_data(b_quant_1, reinterpret_cast<scalar_32bit_t*>(&frag_b1));'
    template = replace_once(template, conversion_start,
                            '#if ZIPSERV_MARLIN_ASYNC && ZIPSERV_MARLIN_ASYNC_DIRECT\n'
                            '      if (!zfa_raw[k2]) {\n'
                            '        zipserv_async_dequant(frag_b_quant[k2], j * 2, zfr_base,\n'
                            '                              reinterpret_cast<uint32_t*>(&frag_b0));\n'
                            '        zipserv_async_dequant(frag_b_quant[k2], j * 2 + 1, zfr_base,\n'
                            '                              reinterpret_cast<uint32_t*>(&frag_b1));\n'
                            '      } else\n#endif\n      {\n' + conversion_start)
    template = replace_once(template, conversion_end, conversion_end + '\n      }')
    if compact_only:
        # The specialized producer never uses tile flags. Remove their storage
        # and all pointer arithmetic, including speculative-stage initialization.
        template = replace_once(template,
            '  auto* zfa_flags = reinterpret_cast<uint32_t*>(sh_a + stages * a_sh_stage);\n', '')
        template = replace_once(template,
            '    if (!pred && threadIdx.x < b_sh_stage / 64)\n'
            '      zfa_flags[pipe * (b_sh_stage / 64) + threadIdx.x] = 1;\n', '')
        template = template.replace('zfa_flags + pipe * (b_sh_stage / 64)', 'nullptr')
        template = replace_once(template, 'zfa_flags + ready_pipe * (b_sh_stage / 64)', 'nullptr')
        if 'zfa_flags' in template:
            raise ValueError("Unremoved flag use in compact-only staging")
    original = inputs[MARLIN + "marlin.cu"].decode()
    notice = original[:original.index('#ifndef MARLIN_NAMESPACE_NAME')]
    start = original.index("typedef struct {")
    end = original.index("}  // namespace marlin", start)
    host = notice + """
// ZipServ modification: isolated native ABI; original host selection and kernel.
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>
#include "quantization/marlin/kernel.h"
#include "quantization/marlin/marlin_template.h"
using std::min;
using std::max;
namespace MARLIN_NAMESPACE_NAME {
__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS) {}
using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);
""" + original[start:end] + "\n}\n" + (HERE / "fp8_marlin_bridge.cu").read_text()
    host = replace_once(host, '  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size;',
                        '  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size;\n'
                        '#if ZIPSERV_MARLIN_ASYNC && !ZIPSERV_MARLIN_ASYNC_COMPACT_ONLY\n'
                        '  total_size += stages * (tb_k * tb_n / 1024) * sizeof(uint32_t);\n'
                        '#endif')
    if exact_scale_cache:
        # This isolated ABI and selector exclusively use E8M0 U8 scales. The
        # upstream generic host estimate reserves two bytes per scale, although
        # the template's is_8bit_scale branch stores one. Apply to raw controls
        # too: a better launch configuration is not a compression-only gain.
        host = replace_once(host, '  int tb_scales = tb_groups * tb_n * 2;',
                            '  // ZipServ bounded E8M0 ABI: exact one-byte scale cache.\n'
                            '  int tb_scales = tb_groups * tb_n;')
    files = {relative.removeprefix("csrc/"): data for relative, data in inputs.items()}
    files["quantization/marlin/marlin_template.h"] = template.encode()
    files["quantization/marlin/kernel_selector.h"] = selector().encode()
    files["host.cu"] = host.encode()
    files["ZIPSERV_MARLIN_NOTICE"] = (HERE / "MARLIN_NOTICE").read_bytes()
    for name in ("fp8_marlin_decode.h", "fp8_marlin_decode.cuh", "fp8_async_decode.h", "fp8_async_decode.cuh"):
        files[name] = (HERE / name).read_bytes()
    staged = output / "staged"
    for relative, data in files.items():
        path = staged / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    cmake = """cmake_minimum_required(VERSION 3.26)
project(zipserv_marlin_fp8 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES @ARCH@-real)
foreach(variant @VARIANTS@)
  add_library(zipserv_marlin_fp8_${variant} SHARED staged/host.cu)
  target_include_directories(zipserv_marlin_fp8_${variant} PRIVATE
    staged staged/quantization/marlin ${ZIPSERV_TORCH_INCLUDE})
  target_compile_options(zipserv_marlin_fp8_${variant} PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--expt-extended-lambda;--expt-relaxed-constexpr;-lineinfo;-Xptxas=-v;-Xcompiler=-fvisibility=hidden>")
  target_compile_definitions(zipserv_marlin_fp8_${variant} PRIVATE
    MARLIN_NAMESPACE_NAME=zipserv_marlin_${variant}
    ZIPSERV_MARLIN_ASYNC_DIRECT=@DIRECT@
    ZIPSERV_MARLIN_ASYNC_PRODUCER=@PRODUCER@
    ZIPSERV_MARLIN_ASYNC_UNIFORM=@UNIFORM@
    ZIPSERV_MARLIN_ASYNC_BYTEPLANE=@BYTEPLANE@
    ZIPSERV_MARLIN_ASYNC_COMPACT_ONLY=@COMPACT_ONLY@
    ZIPSERV_MARLIN_ASYNC_INDEXED_EXCEPTIONS=@INDEXED_EXCEPTIONS@)
  if(variant STREQUAL range16)
    target_compile_definitions(zipserv_marlin_fp8_${variant} PRIVATE ZIPSERV_MARLIN_RANGE16=1)
  else()
    target_compile_definitions(zipserv_marlin_fp8_${variant} PRIVATE ZIPSERV_MARLIN_RANGE16=0)
  endif()
  if(variant STREQUAL async)
    target_compile_definitions(zipserv_marlin_fp8_${variant} PRIVATE ZIPSERV_MARLIN_ASYNC=1)
  else()
    target_compile_definitions(zipserv_marlin_fp8_${variant} PRIVATE ZIPSERV_MARLIN_ASYNC=0)
  endif()
endforeach()
""".replace("@ARCH@", arch).replace("@VARIANTS@", " ".join(variants)).replace(
        "@DIRECT@", str(int(async_decode == "direct"))).replace(
        "@PRODUCER@", str(int(async_decode == "producer"))).replace("@UNIFORM@", str(int(async_exceptions == "uniform"))).replace(
        "@BYTEPLANE@", str(int(async_layout == "byteplane"))).replace("@COMPACT_ONLY@", str(int(compact_only))).replace(
        "@INDEXED_EXCEPTIONS@", str(int(async_exception_loads == "indexed")))
    (output / "CMakeLists.txt").write_text(cmake)
    report = {"upstream": pin, "architecture": arch, "abi": 2 if compact_only else 1,
              "abi_by_variant": {v: 2 if v == "async" and compact_only else 1 for v in variants}, "variants": list(variants),
              "async_decode": async_decode,
              "async_exceptions": async_exceptions,
              "async_layout": async_layout,
              "async_tile_policy": async_tile_policy,
              "async_exception_loads": async_exception_loads,
              "exact_scale_cache": exact_scale_cache,
              "scope": "isolated BF16/MXFP8 group32; M=1..128, native Marlin N/K tile families",
              "cuda_compiled": False, "gpu_validated": False,
              "staged_sha256": {p: sha(d) for p, d in sorted(files.items())},
              "cmake_sha256": sha(cmake.encode()),
              "zipserv_source_sha256": {name: sha((HERE / name).read_bytes()) for name in
                  ("build_fp8_marlin.py", "fp8_marlin_decode.h", "fp8_marlin_decode.cuh",
                   "fp8_marlin_bridge.cu", "fp8_range.py", "fp8_palette.py", "MARLIN_NOTICE",
                   "fp8_async.py", "fp8_byteplane.py", "fp8_async_decode.h", "fp8_async_decode.cuh")}}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", default="121")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-tool", choices=("nvcc", "cmake"), default="nvcc")
    parser.add_argument("--variants", nargs="+", choices=("raw", "range16", "async"), default=["raw", "range16"])
    parser.add_argument("--async-decode", choices=("bytes", "direct", "producer"), default="bytes",
                        help="Experimental ZFA1 fragment bytes, direct BF16 or producer-stage byte expansion")
    parser.add_argument("--async-exceptions", choices=("split", "uniform"), default="split",
                        help="Experimental converged unpacking after conditional exception reads")
    parser.add_argument("--async-layout", choices=("packed7", "byteplane"), default="packed7",
                        help="Experimental ZFA1 packed codes or ZFB1 byte-aligned words")
    parser.add_argument("--async-tile-policy", choices=("mixed", "compact-only"), default="mixed",
                        help="Experimental producer without raw-tile flags; requires validated zero raw tiles")
    parser.add_argument("--exact-scale-cache", action="store_true",
                        help="Experimental exact U8 scale shared-memory accounting for this bounded E8M0 ABI")
    parser.add_argument("--async-exception-loads", choices=("predicated", "indexed"), default="predicated",
                        help="Experimental valid indexed correction loads instead of lane-predicated loads")
    parser.add_argument("--torch-include", type=Path)
    args = parser.parse_args()
    if args.build and args.torch_include is None:
        parser.error("--build requires --torch-include (the installed torch/include directory)")
    if args.fetch:
        fetch(args.source)
    report = stage(args.source, args.output, args.arch, tuple(args.variants), args.async_decode,
                   args.async_exceptions, args.exact_scale_cache, args.async_layout, args.async_tile_policy,
                   args.async_exception_loads)
    if args.build:
        if args.build_tool == "cmake":
            subprocess.run(["cmake", "-S", str(args.output), "-B", str(args.output / "build"),
                            "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                            f"-DZIPSERV_TORCH_INCLUDE={args.torch_include.resolve()}"], check=True)
            subprocess.run(["cmake", "--build", str(args.output / "build"), "--parallel", "2"], check=True)
        else:
            native = args.output.resolve()
            (native / "build").mkdir(exist_ok=True)
            commands = []
            for variant in args.variants:
                command = ["nvcc", "-std=c++17", "-O3", "-shared", "-Xcompiler=-fPIC,-fvisibility=hidden",
                           f"-arch=sm_{args.arch}", "--use_fast_math", "--expt-extended-lambda",
                           "--expt-relaxed-constexpr", "-lineinfo", "-Xptxas=-v",
                           f"-DMARLIN_NAMESPACE_NAME=zipserv_marlin_{variant}",
                           f"-DZIPSERV_MARLIN_RANGE16={int(variant == 'range16')}",
                           f"-DZIPSERV_MARLIN_ASYNC_DIRECT={int(args.async_decode == 'direct')}",
                           f"-DZIPSERV_MARLIN_ASYNC_PRODUCER={int(args.async_decode == 'producer')}",
                           f"-DZIPSERV_MARLIN_ASYNC_UNIFORM={int(args.async_exceptions == 'uniform')}",
                           f"-DZIPSERV_MARLIN_ASYNC_COMPACT_ONLY={int(args.async_tile_policy == 'compact-only')}",
                           f"-DZIPSERV_MARLIN_ASYNC_BYTEPLANE={int(args.async_layout == 'byteplane')}",
                           f"-DZIPSERV_MARLIN_ASYNC_INDEXED_EXCEPTIONS={int(args.async_exception_loads == 'indexed')}",
                           f"-DZIPSERV_MARLIN_ASYNC={int(variant == 'async')}", "-I", str(native / "staged"),
                           "-I", str(native / "staged/quantization/marlin"),
                           "-I", str(args.torch_include.resolve()), str(native / "staged/host.cu"),
                           "-o", str(native / f"build/libzipserv_marlin_fp8_{variant}.so")]
                commands.append(command)
                subprocess.run(command, check=True)
            report["build_commands"] = commands
        report["build_tool"] = args.build_tool
        report["cuda_compiled"] = True
        report["libraries_sha256"] = {f"libzipserv_marlin_fp8_{v}.so":
            sha((args.output / "build" / f"libzipserv_marlin_fp8_{v}.so").read_bytes()) for v in args.variants}
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.output / "manifest.json"),
                      "cuda_compiled": report["cuda_compiled"], "gpu_validated": False}))


if __name__ == "__main__":
    main()
