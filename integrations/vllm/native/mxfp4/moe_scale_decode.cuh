// SPDX-License-Identifier: Apache-2.0
// Copyright MxWave contributors.
#pragma once
#include <cstdint>

#if MXWAVE_MOE_SPARSE
#include "moe_sparse_scales.cuh"
#else

namespace MARLIN_NAMESPACE_NAME {
struct MxWaveMoeScales {
  const int2* records;
  #if MXWAVE_MOE_NIBBLES
  uint64_t base;
  #else
  const int4* exceptions;
  #endif
};
static_assert(sizeof(MxWaveMoeScales) == 16);

__device__ __forceinline__ int4 mxwave_nibble_bf16(uint32_t q, uint32_t base_pair) {
  // Loader ordering: [s0,s1,s4,s5,s2,s3,s6,s7]. Each masked pair then maps
  // directly to stock's BF16 fragment ordering [s0,s2,s1,s3,s4,s6,s5,s7].
  #if MXWAVE_MOE_ALIGNED_NIBBLES
  // The validated base is a multiple of 16. Exponent offsets occupy disjoint
  // bits, allowing a shift + lop3 per BF16 pair instead of mask/shift/add.
  constexpr uint32_t mask = 0x07800780u;
  return make_int4(((q << 7) & mask) | base_pair,
                   ((q << 3) & mask) | base_pair,
                   ((q >> 1) & mask) | base_pair,
                   ((q >> 5) & mask) | base_pair);
  #else
  return make_int4(((q & 0x000f000fu) << 7) + base_pair,
                   (((q >> 4) & 0x000f000fu) << 7) + base_pair,
                   (((q >> 8) & 0x000f000fu) << 7) + base_pair,
                   (((q >> 12) & 0x000f000fu) << 7) + base_pair);
  #endif
}

__device__ __forceinline__ int2 mxwave_nibble_bytes(uint32_t q, uint32_t base) {
  constexpr int order[8] = {0, 1, 4, 5, 2, 3, 6, 7};
  uint32_t lo = 0, hi = 0;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t value = base + ((q >> (4 * order[i])) & 15u);
    if (i < 4) lo |= value << (8 * i);
    else hi |= value << (8 * (i - 4));
  }
  return make_int2(lo, hi);
}

__device__ __forceinline__ uint32_t mxwave_expand(uint32_t value) {
  return (value & 3u) | ((value & 12u) << 6) |
         ((value & 48u) << 12) | ((value & 192u) << 18);
}

__device__ __forceinline__ int2 mxwave_prmt_half(uint32_t bits, uint32_t base) {
  // Spread eight 2-bit selectors to eight nibbles, then select from base+[0,1,2,3].
  // Legal records guarantee base+delta <= 255. Carries at larger, unused
  // table entries cannot change any selected byte, including bases 253..255.
  uint32_t selectors = (bits | (bits << 8)) & 0x00ff00ffu;
  selectors = (selectors | (selectors << 4)) & 0x0f0f0f0fu;
  selectors = (selectors | (selectors << 2)) & 0x33333333u;
  const uint32_t values = base * 0x01010101u + 0x03020100u;
  return make_int2(__byte_perm(values, 0, selectors),
                   __byte_perm(values, 0, selectors >> 16));
}

__device__ __forceinline__ int2 mxwave_record_half(
    int2 record, const int4* exceptions, int half) {
  if (record.y == 256) {
    const int4 raw = exceptions[uint32_t(record.x)];
    return half ? make_int2(raw.z, raw.w) : make_int2(raw.x, raw.y);
  }
  const uint32_t bits = uint32_t(record.x) >> (16 * half);
  #if MXWAVE_MOE_PRMT
  return mxwave_prmt_half(bits & 0xffffu, uint32_t(record.y));
  #else
  const uint32_t base = uint32_t(record.y) * 0x01010101u;
  return make_int2(mxwave_expand(bits) + base, mxwave_expand(bits >> 8) + base);
  #endif
}

__device__ __forceinline__ int2 mxwave_shared_half(
    const int4* shared_scales, const int4* exceptions, int index) {
  const int2 record = *reinterpret_cast<const int2*>(&shared_scales[index >> 1]);
  return mxwave_record_half(record, exceptions, index & 1);
}

__device__ __forceinline__ void mxwave_decode_to_shared(
    int4* destination, int2 record, const int4* exceptions) {
  const int2 lo = mxwave_record_half(record, exceptions, 0);
  const int2 hi = mxwave_record_half(record, exceptions, 1);
  *destination = make_int4(lo.x, lo.y, hi.x, hi.y);
}

__device__ __forceinline__ void mxwave_decode_bf16_to_shared(
    int4* destination, int2 record, const int4* exceptions) {
  const int2 lo = mxwave_record_half(record, exceptions, 0);
  const int2 hi = mxwave_record_half(record, exceptions, 1);
  int4 first, second;
  // Use the exact stock conversion, including its exponent-zero behavior and
  // pair ordering. Hoist it from all consumers to the unique scale producer.
  dequant_fp8_scales<nv_bfloat162, vllm::kFE8M0fnu.id()>(
      lo.x, reinterpret_cast<nv_bfloat162*>(&first));
  dequant_fp8_scales<nv_bfloat162, vllm::kFE8M0fnu.id()>(
      lo.y, reinterpret_cast<nv_bfloat162*>(&first) + 2);
  dequant_fp8_scales<nv_bfloat162, vllm::kFE8M0fnu.id()>(
      hi.x, reinterpret_cast<nv_bfloat162*>(&second));
  dequant_fp8_scales<nv_bfloat162, vllm::kFE8M0fnu.id()>(
      hi.y, reinterpret_cast<nv_bfloat162*>(&second) + 2);
  destination[0] = first;
  destination[1] = second;
}

__global__ void mxwave_decode_moe_bf16_scales(
    const MxWaveMoeScales* descriptor, int4* output, int vectors) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vectors) return;
  #if MXWAVE_MOE_NIBBLES
  const int2 q = descriptor->records[index];
  const uint32_t base_pair = uint32_t(descriptor->base) * 0x00800080u;
  output[2 * index] = mxwave_nibble_bf16(uint32_t(q.x), base_pair);
  output[2 * index + 1] = mxwave_nibble_bf16(uint32_t(q.y), base_pair);
  #else
  mxwave_decode_bf16_to_shared(output + 2 * index, descriptor->records[index],
                              descriptor->exceptions);
  #endif
}

__global__ void mxwave_decode_moe_scales(
    const MxWaveMoeScales* descriptor, int4* output, int vectors) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vectors) return;
  const int2 record = descriptor->records[index];
  #if MXWAVE_MOE_NIBBLES
  const int2 lo = mxwave_nibble_bytes(uint32_t(record.x), uint32_t(descriptor->base));
  const int2 hi = mxwave_nibble_bytes(uint32_t(record.y), uint32_t(descriptor->base));
  #else
  const int2 lo = mxwave_record_half(record, descriptor->exceptions, 0);
  const int2 hi = mxwave_record_half(record, descriptor->exceptions, 1);
  #endif
  output[index] = make_int4(lo.x, lo.y, hi.x, hi.y);
}
}  // namespace MARLIN_NAMESPACE_NAME
#endif
