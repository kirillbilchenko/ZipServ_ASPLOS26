// Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
// ZFR1, group=16, tile=2048. The caller validates the entire immutable
// container before upload. This is also compiled on the CPU by the tests.
#pragma once
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define ZFR_INLINE __device__ __forceinline__
#else
#define ZFR_INLINE inline
#endif

namespace zipserv_fp8 {

struct Words { uint32_t x, y, z, w; };
struct Block { const uint8_t* data; bool raw; bool raw_tile; };

ZFR_INLINE uint32_t load16(const uint8_t* p) {
#if defined(__CUDACC__)
  // Both 14-byte packed blocks and 16-byte raw blocks are TWO-byte aligned.
  return *reinterpret_cast<const uint16_t*>(p);
#else
  uint16_t value;
  std::memcpy(&value, p, sizeof(value));
  return value;
#endif
}

ZFR_INLINE uint64_t load64(const uint8_t* p) {
#if defined(__CUDACC__)
  return *reinterpret_cast<const uint64_t*>(p);
#else
  uint64_t value;
  std::memcpy(&value, p, sizeof(value));
  return value;
#endif
}

ZFR_INLINE unsigned popcount(uint64_t v) {
#if defined(__CUDACC__)
  return __popcll(v);
#else
  return __builtin_popcountll(v);
#endif
}

// One lookup for all sixteen bytes consumed by Marlin's original int4 load.
ZFR_INLINE Block locate(const uint8_t* encoded, uint64_t vector) {
  const uint64_t entry = load64(encoded + 64 + (vector >> 7) * 8);
  const uint8_t* tile = encoded + (entry & ~uint64_t(1));
  const unsigned group = vector & 127;
  if (entry & 1) return {tile + group * 16, true, true};
  const unsigned lane = group & 63;
  const uint64_t mask = load64(tile + (group >> 6) * 8);
  unsigned preceding = popcount(mask & ((uint64_t(1) << lane) - 1));
  if (group >= 64) preceding += popcount(load64(tile));
  return {tile + 16 + group * 14 + preceding * 2,
          bool((mask >> lane) & 1), false};
}

// Spread four nibbles / four 3-bit offsets into byte lanes using integer masks.
// Adding the base cannot carry between lanes: every exponent is at most 15.
ZFR_INLINE uint32_t expand4(uint32_t low, uint32_t exp, uint32_t base) {
  low = (low | (low << 8)) & 0x00ff00ffu;
  low = (low | (low << 4)) & 0x0f0f0f0fu;
  exp = (exp & 0x3fu) | ((exp & 0xfc0u) << 10);
  exp = (exp | (exp << 5)) & 0x07070707u;
  exp += base * 0x01010101u;
  return (low & 0x07070707u) | ((low & 0x08080808u) << 4) | (exp << 3);
}

ZFR_INLINE uint32_t raw4(const uint8_t* p) {
  return load16(p) | (load16(p + 2) << 16);
}

ZFR_INLINE Words decode(Block block, uint32_t base) {
  const uint8_t* p = block.data;
  if (block.raw) return {raw4(p), raw4(p + 4), raw4(p + 8), raw4(p + 12)};
  // Exactly six exponent bytes; never read beyond a 14-byte block.
  const uint32_t e0 = load16(p + 8), e1 = load16(p + 10), e2 = load16(p + 12);
  return {expand4(load16(p), e0, base),
          expand4(load16(p + 2), (e0 >> 12) | (e1 << 4), base),
          expand4(load16(p + 4), (e1 >> 8) | (e2 << 8), base),
          expand4(load16(p + 6), e2 >> 4, base)};
}

}  // namespace zipserv_fp8
#undef ZFR_INLINE
