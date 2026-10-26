// Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fp8_async_decode.h"
#ifndef ZIPSERV_MARLIN_ASYNC_INDEXED_EXCEPTIONS
#define ZIPSERV_MARLIN_ASYNC_INDEXED_EXCEPTIONS 0
#endif
namespace MARLIN_NAMESPACE_NAME {
__device__ __forceinline__ void zipserv_async_copy(
    int4* shared_stage, uint32_t* shared_flags, int local_vector,
    const uint8_t* encoded, int global_vector) {
  const uint64_t entry = zipserv_fp8::load64(encoded + 64 + (uint64_t(global_vector) >> 6) * 8);
  const uint8_t* payload = encoded + (entry & ~uint64_t(15));
  const unsigned lane = global_vector & 63;
  const unsigned copies = 64 - ((entry >> 1) & 7);
  cp_async4_pred(shared_stage + local_vector, payload + lane * 16, lane < copies);
  if ((local_vector & 63) == 0) shared_flags[local_vector >> 6] = entry & 1;
}

// Producer ownership: a warp copies an entire 1024-byte tile, one vector per
// lane from each half. Both instructions have contiguous 16-byte lane accesses;
// adjacent pairs per lane would double sector requests for aligned raw tiles.
// Each lane waits for its own copies before the warp expands in place.
__device__ __forceinline__ void zipserv_async_copy_tile_lane(
    int4* shared_stage, uint32_t* shared_flags, int local_vector,
    const uint8_t* encoded, int global_vector) {
  const uint64_t entry = zipserv_fp8::load64(encoded + 64 + (uint64_t(global_vector) >> 6) * 8);
  const uint8_t* payload = encoded + (entry & ~uint64_t(15));
  const unsigned lane = global_vector & 63;
  const unsigned copies = 64 - ((entry >> 1) & 7);
  cp_async4_pred(shared_stage + local_vector, payload + lane * 16, lane < copies);
  cp_async4_pred(shared_stage + local_vector + 32, payload + (lane + 32) * 16, lane + 32 < copies);
#if !ZIPSERV_MARLIN_ASYNC_COMPACT_ONLY
  if ((local_vector & 63) == 0) shared_flags[local_vector >> 6] = entry & 1;
#endif
}

__device__ __forceinline__ void zipserv_async_expand_tile(
    int4* shared_stage, const uint32_t* shared_flags, int tile_vector, uint32_t common) {
  // Caller has completed this warp's cp.async operations for the ready stage.
  // Make those copies and the issuing lane's flag visible to the whole warp.
  __syncwarp();
#if ZIPSERV_MARLIN_ASYNC_COMPACT_ONLY
  {  // The immutable container was checked for zero raw tiles before upload.
#else
  if (!shared_flags[tile_vector >> 6]) {
#endif
    const unsigned lane = threadIdx.x & 31;
    const auto pair = zipserv_fp8::async_fragment<ZIPSERV_MARLIN_ASYNC_UNIFORM,
        ZIPSERV_MARLIN_ASYNC_BYTEPLANE, ZIPSERV_MARLIN_ASYNC_INDEXED_EXCEPTIONS>(
        reinterpret_cast<const uint8_t*>(shared_stage + tile_vector), lane, false, common);
    // Read all compact inputs before any lane overwrites them with wider bytes.
    __syncwarp();
    shared_stage[tile_vector + lane * 2] =
        make_int4(pair.first.x, pair.first.y, pair.first.z, pair.first.w);
    shared_stage[tile_vector + lane * 2 + 1] =
        make_int4(pair.second.x, pair.second.y, pair.second.z, pair.second.w);
  }
  // The existing Marlin CTA barrier publishes the expanded stage to consumers.
}

__device__ __forceinline__ void zipserv_async_read(
    I4& first, I4& second, const int4* stage, const uint32_t* flags,
    int vector, uint32_t common) {
  const auto pair = zipserv_fp8::async_fragment<ZIPSERV_MARLIN_ASYNC_UNIFORM,
      ZIPSERV_MARLIN_ASYNC_BYTEPLANE, ZIPSERV_MARLIN_ASYNC_INDEXED_EXCEPTIONS>(
      reinterpret_cast<const uint8_t*>(stage + (vector & ~63)), (vector & 63) >> 1,
      flags[vector >> 6] != 0, common);
  reinterpret_cast<int4*>(&first)[0] = make_int4(pair.first.x, pair.first.y, pair.first.z, pair.first.w);
  reinterpret_cast<int4*>(&second)[0] = make_int4(pair.second.x, pair.second.y, pair.second.z, pair.second.w);
}

// Experimental direct-to-BF16 path. Seven packed words plus one correction
// word occupy the same eight registers as the original 32 decoded FP8 bytes.
__device__ __forceinline__ bool zipserv_async_read_packed(
    I4& first, I4& second, const int4* stage, const uint32_t* flags, int vector) {
  if (flags[vector >> 6]) {
    first = *reinterpret_cast<const I4*>(stage + vector);
    second = *reinterpret_cast<const I4*>(stage + vector + 1);
    return true;
  }
  const auto* tile = reinterpret_cast<const uint8_t*>(stage + (vector & ~63));
  const unsigned pair = (vector & 63) >> 1, group = pair * 2;
  const uint64_t mask = zipserv_fp8::load64(tile);
  const unsigned exceptions = (mask >> group) & 3;
  uint32_t flips = 0;
  if (exceptions) {
    const unsigned preceding = zipserv_fp8::popcount(mask & ((uint64_t(1) << group) - 1));
    const uint8_t* extra = tile + 912 + preceding * 2;
    if (exceptions & 1) flips = zipserv_fp8::load16(extra);
    if (exceptions & 2) flips |= zipserv_fp8::load16(extra + (exceptions & 1) * 2) << 16;
  }
  const auto* p = reinterpret_cast<const uint32_t*>(tile + 16 + pair * 28);
  reinterpret_cast<int4*>(&first)[0] = make_int4(p[0], p[1], p[2], p[3]);
  reinterpret_cast<int4*>(&second)[0] = make_int4(p[4], p[5], p[6], flips);
  return false;
}

__device__ __forceinline__ void zipserv_async_dequant(
    const I4* fragment, unsigned quad, uint32_t common, uint32_t* output) {
  const auto* words = reinterpret_cast<const uint32_t*>(fragment);
  // Constant-index alternative to the initial dynamic extraction. Both builds
  // still report a 32-byte local stack on small-M specializations; this alone
  // does not remove the compiler's temporary storage or qualify performance.
  uint32_t codes;
  switch (quad) {
    case 0: codes = words[0]; break;
    case 1: codes = (words[0] >> 28) | (words[1] << 4); break;
    case 2: codes = (words[1] >> 24) | (words[2] << 8); break;
    case 3: codes = (words[2] >> 20) | (words[3] << 12); break;
    case 4: codes = (words[3] >> 16) | (words[4] << 16); break;
    case 5: codes = (words[4] >> 12) | (words[5] << 20); break;
    case 6: codes = (words[5] >> 8) | (words[6] << 24); break;
    default: codes = words[6] >> 4; break;  // quad=7; internal unrolled loop.
  }
  const auto result = zipserv_fp8::direct7(codes, words[7] >> (quad * 4), common);
  output[0] = result.even;
  output[1] = result.odd;
  // Preserve the pinned BF16/E4M3/E8M0 dequant<..., false> bias multiply.
  // This is intentionally performed before the unchanged group-scale multiply.
  const nv_bfloat162 bias = __float2bfloat162_rn(__uint_as_float(247u << 23));
  auto* bf16 = reinterpret_cast<nv_bfloat162*>(output);
  bf16[1] = __hmul2(bf16[1], bias);
  bf16[0] = __hmul2(bf16[0], bias);
}
}  // namespace MARLIN_NAMESPACE_NAME
