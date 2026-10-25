// Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
// Portable ZFA1 fragment decoder. Inputs are validated, aligned staged tiles.
#pragma once
#include "fp8_marlin_decode.h"
#if defined(__CUDACC__)
#define ZFA_INLINE __device__ __forceinline__
#else
#define ZFA_INLINE inline
#endif
namespace zipserv_fp8 {
struct WordPair { Words first, second; };

// Same *unscaled* BF16 bit patterns as pinned Marlin's E4M3 conversion.
// The original scale/bias and MMA operations still run afterwards. Keeping
// packed codes until conversion avoids constructing an intermediate FP8 word.
struct Bf16Pair { uint32_t even, odd; };
ZFA_INLINE uint32_t direct7_pair(uint32_t codes, uint32_t flips, uint32_t common) {
  const uint32_t magnitude = ((codes & 0x3fu) << 4) |
                             ((codes & 0xfc000u) << 6);
  const uint32_t sign = ((codes & 0x40u) << 9) |
                        ((codes & 0x100000u) << 11);
  const uint32_t high = (common * 0x04000400u) ^
                        ((flips & 1u) << 10) ^ ((flips & 4u) << 24);
  return magnitude | sign | high;
}
ZFA_INLINE Bf16Pair direct7(uint32_t codes, uint32_t flips, uint32_t common) {
  return {direct7_pair(codes, flips, common),
          direct7_pair(codes >> 7, flips >> 1, common)};
}

ZFA_INLINE uint32_t load32(const uint8_t* p) {
#if defined(__CUDACC__)
  return *reinterpret_cast<const uint32_t*>(p);
#else
  uint32_t value;
  std::memcpy(&value, p, sizeof(value));
  return value;
#endif
}

ZFA_INLINE uint32_t expand7(uint32_t packed) {
  uint32_t v = (packed & 0x00003fffu) | ((packed & 0x0fffc000u) << 2);
  v = (v & 0x007f007fu) | ((v & 0x3f803f80u) << 1);
  return (v & 0x3f3f3f3fu) | ((v & 0x40404040u) << 1);
}

template <bool exceptions>
ZFA_INLINE uint32_t finish7(uint32_t packed, uint32_t high, uint32_t flips) {
  if constexpr (exceptions)
    high ^= (((flips & 15u) * 0x00204081u) & 0x01010101u) << 6;
  return expand7(packed) | high;
}

template <bool exceptions>
ZFA_INLINE uint32_t finish_byteplane(uint32_t word, uint32_t high, uint32_t flips) {
  if constexpr (exceptions)
    high ^= (((flips & 15u) * 0x00204081u) & 0x01010101u) << 6;
  return (word & 0xbfbfbfbfu) | high;
}

template <bool exceptions, bool byteplane = false>
ZFA_INLINE WordPair unpack32(const uint8_t* p, uint32_t high, uint32_t flips) {
  const uint32_t a = load32(p), b = load32(p + 4), c = load32(p + 8);
  const uint32_t d = load32(p + 12), e = load32(p + 16);
  const uint32_t f = load32(p + 20), g = load32(p + 24);
  if constexpr (byteplane) {
    const uint32_t tail = ((a >> 6) & 0x01010101u) | ((b >> 5) & 0x02020202u) |
                          ((c >> 4) & 0x04040404u) | ((d >> 3) & 0x08080808u) |
                          ((e >> 2) & 0x10101010u) | ((f >> 1) & 0x20202020u) |
                          ((g << 1) & 0x80808080u);
    return {{finish_byteplane<exceptions>(a, high, flips),
             finish_byteplane<exceptions>(b, high, flips >> 4),
             finish_byteplane<exceptions>(c, high, flips >> 8),
             finish_byteplane<exceptions>(d, high, flips >> 12)},
            {finish_byteplane<exceptions>(e, high, flips >> 16),
             finish_byteplane<exceptions>(f, high, flips >> 20),
             finish_byteplane<exceptions>(g, high, flips >> 24),
             finish_byteplane<exceptions>(tail, high, flips >> 28)}};
  }
  return {{finish7<exceptions>(a, high, flips),
           finish7<exceptions>((a >> 28) | (b << 4), high, flips >> 4),
           finish7<exceptions>((b >> 24) | (c << 8), high, flips >> 8),
           finish7<exceptions>((c >> 20) | (d << 12), high, flips >> 12)},
          {finish7<exceptions>((d >> 16) | (e << 16), high, flips >> 16),
           finish7<exceptions>((e >> 12) | (f << 20), high, flips >> 20),
           finish7<exceptions>((f >> 8) | (g << 24), high, flips >> 24),
           finish7<exceptions>(g >> 4, high, flips >> 28)}};
}

template <bool uniform = false, bool byteplane = false, bool indexed_exceptions = false>
ZFA_INLINE WordPair async_fragment(const uint8_t* tile, unsigned pair,
                                   bool raw, unsigned common) {
  static_assert(!indexed_exceptions || uniform,
                "Indexed exception loads require a uniform empty-mask guard");
  if (raw) {
    const uint8_t* p = tile + pair * 32;
    return {{load32(p), load32(p + 4), load32(p + 8), load32(p + 12)},
            {load32(p + 16), load32(p + 20), load32(p + 24), load32(p + 28)}};
  }
  const uint64_t mask = load64(tile);
  const unsigned group = pair * 2;
  const unsigned flags = (mask >> group) & 3;
  const uint8_t* p = tile + 16 + pair * 28;  // Always four-byte aligned.
  const uint32_t high = common * 0x40404040u;
  // A tile belongs to one warp in both native fragment and producer mappings.
  // Keep the no-exception tile fast path uniform across the whole warp. Mixed
  // tiles converge before unpacking, rather than executing two unpack bodies
  // under complementary lane predicates.
  if constexpr (uniform) {
    if (!mask) return unpack32<false, byteplane>(p, high, 0);
  } else {
    if (!flags) return unpack32<false, byteplane>(p, high, 0);
  }
  uint32_t flips = 0;
  if constexpr (indexed_exceptions) {
    // A nonempty tile always owns at least one correction. Lanes without a
    // correction read that first valid element and mask its value to zero.
    // This removes lane-divergent loads without reading alignment padding or
    // the next tile, including when the exception area ends at its boundary.
    const unsigned preceding = popcount(mask & ((uint64_t(1) << group) - 1));
    const unsigned has_lo = flags & 1, has_hi = (flags >> 1) & 1;
    const auto* extra = tile + 912;
    const uint32_t lo = load16(extra + (has_lo ? preceding : 0) * 2) * has_lo;
    const uint32_t hi = load16(extra + (has_hi ? preceding + has_lo : 0) * 2) * has_hi;
    flips = lo | (hi << 16);
  } else if (flags) {
    const unsigned preceding = popcount(mask & ((uint64_t(1) << group) - 1));
    const uint8_t* extra = tile + 912 + preceding * 2;
    const uint32_t lo = flags & 1 ? load16(extra) : 0;
    const uint32_t hi = flags & 2 ? load16(extra + (flags & 1) * 2) : 0;
    flips = lo | (hi << 16);
  }
  return unpack32<true, byteplane>(p, high, flips);
}
}  // namespace zipserv_fp8
#undef ZFA_INLINE
