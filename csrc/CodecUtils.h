// Copyright 2026 Kyrylo Bilchenko
// SPDX-License-Identifier: Apache-2.0
// Host contracts and bit operations shared by the compressor and CUDA kernels.
#ifndef ZIPSERV_CODEC_UTILS_H
#define ZIPSERV_CODEC_UTILS_H
#include <cstdint>
#include <climits>
#include <stdexcept>

namespace zipserv {
inline bool valid_matrix(int m, int k) {
    return m > 0 && k > 0 && m % 64 == 0 && k % 64 == 0 &&
           int64_t(m) * k <= INT_MAX;
}

inline bool valid_gemm(int m, int n, int k, int split_k) {
    if (!valid_matrix(m, k) || n < 0 || split_k < 1 || (k / 64) % split_k != 0)
        return false;
    // Kernels use int32 linear offsets and CUDA's 16-bit grid.y limit.
    if (int64_t(m) * n > INT_MAX || int64_t(k) * n > INT_MAX)
        return false;
    return int64_t(m) * n * split_k <= INT_MAX &&
           int64_t(m / 64) * split_k <= 65535;
}

inline bool valid_counts(int common, int rare) {
    return common >= 0 && common <= 4096 && common % 16 == 0 &&
           rare >= 0 && rare <= 4096 && rare % 8 == 0 && common + rare >= 4096;
}

inline bool valid_exponents(const int* exponents) {
    if (!exponents || exponents[0] < 0 || exponents[0] > 249) return false;
    for (int i = 1; i < 7; ++i)
        if (exponents[i] != exponents[0] + i) return false;
    return true;
}

// Maximize common values, including windows with missing exponents. Ties use
// the lowest base. This minimizes unpadded payload; tile padding is separate.
inline int best_exponent_window(const uint64_t* counts, int width = 7) {
    if (!counts || width < 1 || width > 7)
        throw std::invalid_argument("exponent window width must be in [1, 7]");
    uint64_t current = 0;
    for (int i = 0; i < width; ++i) current += counts[i];
    uint64_t best = current;
    int base = 0;
    for (int start = 1; start <= 256 - width; ++start) {
        current -= counts[start - 1];
        current += counts[start + width - 1];
        if (current > best) { best = current; base = start; }
    }
    return base;
}

#ifdef __CUDACC__
__host__ __device__
#endif
inline uint16_t exponent_bits(uint8_t start_exp, uint8_t code) {
    // A window beginning at exponent zero stores start_exp == 255. Wrap the
    // exponent before shifting, or bit 8 of the sum corrupts the BF16 sign.
    return uint16_t(uint8_t(start_exp + code)) << 7;
}
}  // namespace zipserv
#endif
