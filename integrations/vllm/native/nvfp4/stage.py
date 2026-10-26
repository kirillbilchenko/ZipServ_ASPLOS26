# Copyright MxWave contributors.
# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Stage pinned Apache-licensed Marlin raw/NVFP4 scale-consumer components."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stage(source: Path, output: Path) -> dict[str, object]:
    """Preserve GEMM arithmetic; patch only exact scale-byte fetching in one variant."""
    package = Path(__file__).resolve().parent
    provenance = json.loads((package.parent / 'mxfp4/moe_sources.json').read_text())
    staged = output / 'staged'
    for relative, expected in provenance['files'].items():
        raw = (source / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'Pinned source mismatch: {relative}')
        path = staged / relative.removeprefix('csrc/')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    moe = staged / 'libtorch_stable/moe/marlin_moe_wna16'
    path = moe / 'marlin_template.h'
    template = path.read_text()

    def replace(old: str, new: str, count: int = 1) -> None:
        nonlocal template
        if template.count(old) != count:
            raise ValueError(f'Unexpected patch site: {old}')
        template = template.replace(old, new)

    replace('#include "core/scalar_type.hpp"',
            '#include "core/scalar_type.hpp"\n#include "scale_decode.cuh"')
    replace('  float global_scale_f32 = 1.0f;', '''  float global_scale_f32 = 1.0f;
#if NV_COMPRESSED
  const NvScales nv_meta = *reinterpret_cast<const NvScales*>(scales_ptr);
  __shared__ uint32_t nv_palette[64];
#endif''')
    pointer = 'scales_ptr += (expert_id - old_expert_id) * scales_expert_stride;'
    replace(pointer, '''#if NV_COMPRESSED
    if (threadIdx.x < nv_meta.num_parts * 32) {
      const NvPart& p = nv_meta.parts[expert_id * nv_meta.num_parts + threadIdx.x / 32];
      nv_palette[threadIdx.x] = p.raw ? 0 : p.palette[threadIdx.x % 32];
    }
    // Existing read_moe_block_data barriers publish this palette before scale fetches.
#else
    ''' + pointer + '\n#endif')
    for name, count in (('sh_s_stage', 1), ('sh_s', 2)):
        old = f'cp_async4(&{name}[s_sh_wr], &scales_ptr[s_gl_rd]);'
        replace(old, '#if NV_COMPRESSED >= 2\n' +
                f'nv_fetch(&{name}[s_sh_wr], nv_meta, expert_id, s_gl_rd);\n' +
                '#elif NV_COMPRESSED\n' +
                f'{name}[s_sh_wr] = nv_decode(nv_meta, expert_id, s_gl_rd, nv_palette);\n' +
                '#else\n' + old + '\n#endif', count)
    replace('auto wait_for_stage = [&]() {',
            'auto wait_for_stage = [&](int ready_pipe, bool valid_stage = true) {')
    replace('    cp_async_wait<stages - 2>();\n    __syncthreads();', '''
    cp_async_wait<stages - 2>();
#if NV_COMPRESSED >= 2
    if (valid_stage && s_sh_wr_pred) {
      const int part = ((s_sh_stride * slice_col) >> nv_meta.part_shift) &
                       (nv_meta.num_parts - 1);
      if (!nv_meta.parts[expert_id * nv_meta.num_parts + part].raw) {
        int4* slot = &sh_s[s_sh_stage * ready_pipe + s_sh_wr];
        const int4 record = *slot;
#if NV_COMPRESSED == 3
        *slot = nv_expand_delta(make_uint2(record.x, record.y),
                                uint32_t(record.z) >> (record.w & 16),
                                nv_palette[(record.w >> 5) * 32]);
#else
        *slot = nv_expand(make_uint2(record.x, record.y),
                          uint32_t(record.z) >> (record.w & 16),
                          record.w >> 5, nv_palette);
#endif
      }
    }
#endif
    __syncthreads();''')
    replace('    wait_for_stage();', '    wait_for_stage(0);', 2)
    replace('          wait_for_stage(0);',
            '          wait_for_stage(pipe % stages, slice_iters > 1);')
    site = '    if (red_off >= 1) {\n'
    replace(site, site + '      // Finish shared-B reads before reduction reuses its storage.\n      __syncthreads();\n')
    path.write_text(template)
    (staged / 'scale_decode.cuh').write_bytes((package / 'scale_decode.cuh').read_bytes())
    selectors = []
    for blocks, small in ((1, True), (1, False), (2, False), (3, False), (4, False)):
        for k, n, threads in ((128, 128, 256), (64, 256, 256), (64, 128, 128), (128, 64, 128)):
            if threads == 256 and ((blocks == 1) != (k == 128)):
                continue
            selectors.append(
                f'if (thread_m_blocks == {blocks} && thread_k_blocks == {k // 16} && '
                f'thread_n_blocks == {n // 16} && threads == {threads} && '
                f'm_block_size_8 == {str(small).lower()}) '
                'kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), '
                f'vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), {threads}, {blocks}, '
                f'{n // 16}, {k // 16}, {str(small).lower()}, 4, 1, false>;'
            )
    (moe / 'kernel_selector.h').write_text('\n'.join(selectors) + '\n')
    original = (moe / 'ops.cu').read_text()
    host = original[:original.index('#include "kernel.h"')] + '''
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
'''
    host += original[original.index('typedef struct {'):
                     original.index('}  // namespace MARLIN_NAMESPACE_NAME')]
    allocation = '  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,'
    if host.count(allocation) != 1:
        raise ValueError('Unexpected dynamic shared-memory opt-in site')
    host = host.replace(allocation, '''#if NV_COMPRESSED
  // The decoder owns static shared palette bytes. A one-block-per-SM plan
  // must leave room for them instead of opting in to the entire device limit
  // as dynamic memory. Keep the original tile and reduction configuration.
  cudaFuncAttributes attributes;
  auto attribute_status = cudaFuncGetAttributes(&attributes, kernel);
  STD_TORCH_CHECK(attribute_status == cudaSuccess,
                  "Cannot inspect NVFP4 kernel shared-memory requirements");
  max_shared_mem -= static_cast<int>(attributes.sharedSizeBytes);
  STD_TORCH_CHECK(sh_cache_size <= max_shared_mem,
                  "NVFP4 static and dynamic shared-memory requirements exceed the tile budget");
#endif
''' + allocation)
    host += '\n}\n' + (package / 'bridge.cu').read_text()
    (staged / 'host.cu').write_text(host)
    result = {'source': provenance, 'architecture': 'sm_121',
              'format': 'NVFP4 group16/S0E5M3, BF16 activations, FP32 reduction',
              'variants': ['raw', 'direct5', 'async5', 'delta5'],
              'kernel_configurations': len(selectors),
              'files': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(staged.rglob('*')) if p.is_file()}}
    (output / 'source-manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main() -> None:
    """Stage only: CUDA compilation is separately bounded and logged on Spark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage(args.source, args.output)


if __name__ == '__main__':
    main()
