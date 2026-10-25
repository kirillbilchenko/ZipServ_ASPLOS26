// Copyright 2026 Kyrylo Bilchenko
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Expand the existing CT selector order [0,1,4,5,2,3,6,7] to eight U8 scales.
// The caller must verify a consecutive three-value palette. Normal records
// contain only selectors 0..2; whole-vector exceptions use the original path.
// No byte can overflow: the palette's first value is at most 253.
__device__ __forceinline__ int2 zipserv_fp4_affine_half(uint32_t palette, uint32_t q) {
  const uint32_t spread = (q | (q << 8)) & 0x00ff00ffu;
  const uint32_t base = (palette & 255u) * 0x01010101u;
  const uint32_t low = ((spread | (spread << 6)) & 0x03030303u) + base;
  const uint32_t high = (((spread >> 4) | (spread << 2)) & 0x03030303u) + base;
  return make_int2(static_cast<int>(low), static_cast<int>(high));
}
