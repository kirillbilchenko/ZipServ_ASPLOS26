// Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fp8_marlin_decode.h"

namespace MARLIN_NAMESPACE_NAME {
__device__ __forceinline__ void zipserv_range16_copy(
    int4* destination, const uint8_t* encoded, int vector, uint32_t base) {
  const auto block = zipserv_fp8::locate(encoded, uint64_t(vector));
  if (block.raw_tile) {
    // Raw tiles retain Marlin's original aligned asynchronous copy.
    cp_async4(destination, block.data);
  } else {
    const auto v = zipserv_fp8::decode(block, base);
    *destination = make_int4(v.x, v.y, v.z, v.w);
  }
  // The original commit/wait + CTA barrier orders both paths before use.
}
}  // namespace MARLIN_NAMESPACE_NAME
