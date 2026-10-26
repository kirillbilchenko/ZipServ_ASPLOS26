// SPDX-License-Identifier: Apache-2.0
// Copyright MxWave contributors.
#pragma once
#include <cstdint>
namespace MARLIN_NAMESPACE_NAME {
struct NvPart {
  const uint2* low;
  const uint16_t* high;
  const uint8_t* palette;
  const int4* raw;
};
struct NvScales {
  const NvPart* parts;
  int row_shift;
  int part_shift;
  int num_parts;
  int reserved;
};
static_assert(sizeof(NvPart) == 32);
static_assert(sizeof(NvScales) == 24);
__device__ __forceinline__ int4 nv_expand_delta(uint2 low, uint32_t high, uint32_t base) {
  uint32_t out[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t q = (i < 2 ? low.x : low.y) >> (16 * (i % 2));
    const uint32_t lo = __byte_perm(q & 0x0f0f, (q >> 4) & 0x0f0f, 0x5140);
    const uint32_t h = (high >> (4 * i)) & 15;
    const uint32_t hi = ((h * 0x00204081u) & 0x01010101u) << 4;
    // Host verifies base + every observed code <= 255: no cross-byte carries.
    out[i] = (lo | hi) + base * 0x01010101u;
  }
  return make_int4(out[0], out[1], out[2], out[3]);
}
__device__ __forceinline__ int4 nv_expand(
    uint2 low, uint32_t high, int part, const uint32_t* palette) {
  uint32_t out[4] = {};
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    uint32_t word = i < 8 ? low.x : low.y;
    uint32_t code = ((word >> (4 * (i % 8))) & 15) |
                    (((high >> i) & 1) << 4);
    out[i / 4] |= palette[part * 32 + code] << (8 * (i % 4));
  }
  return make_int4(out[0], out[1], out[2], out[3]);
}
__device__ __forceinline__ void nv_fetch(
    int4* destination, const NvScales& d, int expert, int vector) {
  int part = (vector >> d.part_shift) & (d.num_parts - 1);
  int local = ((vector >> d.row_shift) << d.part_shift) |
              (vector & ((1 << d.part_shift) - 1));
  const NvPart& p = d.parts[expert * d.num_parts + part];
  if (p.raw) {
    cp_async4(destination, p.raw + local);
  } else {
    cp_async2_ca_pred(destination, p.low + local);
    cp_async1_ca_pred(&destination->z,
                     reinterpret_cast<const uint32_t*>(p.high) + (local >> 1));
    destination->w = ((local & 1) * 16) | (part * 32);
  }
}
__device__ __forceinline__ int4 nv_decode(
    const NvScales& d, int expert, int vector, const uint32_t* palette) {
  int part = (vector >> d.part_shift) & (d.num_parts - 1);
  int local = ((vector >> d.row_shift) << d.part_shift) |
              (vector & ((1 << d.part_shift) - 1));
  const NvPart& p = d.parts[expert * d.num_parts + part];
  if (p.raw) return p.raw[local];
  const uint2 low = p.low[local];
  const uint32_t high = p.high[local];
#if NV_COMPRESSED == 3
  return nv_expand_delta(low, high, palette[part * 32]);
#else
  return nv_expand(low, high, part, palette);
#endif
}
__global__ void nv_decode_all(const NvScales* descriptor, int4* output,
                              int vectors_per_expert, int vectors) {
  const NvScales d = *descriptor;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vectors) return;
  int expert = index / vectors_per_expert;
  uint32_t palette[64];
  for (int part = 0; part < d.num_parts; ++part) {
    const NvPart& p = d.parts[expert * d.num_parts + part];
    if (!p.raw) for (int i = 0; i < 32; ++i) palette[part * 32 + i] = p.palette[i];
  }
  output[index] = nv_decode(d, expert, index % vectors_per_expert, palette);
}
}
