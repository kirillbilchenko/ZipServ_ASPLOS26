// Copyright 2026 Kyrylo Bilchenko
// SPDX-License-Identifier: Apache-2.0
#include "../csrc/CodecUtils.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>

int main() {
    assert(zipserv::valid_gemm(64, 1, 128, 2));
    assert(zipserv::valid_gemm(128, 129, 512, 8));
    assert(zipserv::valid_gemm(64, 0, 64, 1));
    assert(!zipserv::valid_gemm(0, 8, 64, 1));
    assert(!zipserv::valid_gemm(64, -1, 64, 1));
    assert(!zipserv::valid_gemm(64, 8, 192, 2));
    assert(!zipserv::valid_gemm(64, 8, 64, 0));
    assert(!zipserv::valid_gemm(64, 8, 64, 2));
    assert(!zipserv::valid_gemm(64, INT_MAX, 64, 1));
    assert(!zipserv::valid_gemm(4194304, 1, 64, 1)); // grid.y
    assert(!zipserv::valid_matrix(INT_MAX, 64));
    assert(zipserv::valid_counts(4096, 0));
    assert(zipserv::valid_counts(0, 4096));
    assert(!zipserv::valid_counts(4095, 0));
    assert(!zipserv::valid_counts(0, 0));

    uint64_t counts[256] = {};
    counts[50] = 2; counts[51] = 2; counts[52] = 2;
    counts[53] = 2; counts[54] = 2; counts[55] = 2; counts[56] = 2;
    counts[120] = 10000; counts[126] = 10000;
    assert(zipserv::best_exponent_window(counts) == 120); // gaps are allowed
    std::mt19937 rng(42);
    for (int trial = 0; trial < 1000; ++trial) {
        for (auto& c : counts) c = rng() % 10000;
        for (int width = 1; width <= 7; ++width) {
            int chosen = zipserv::best_exponent_window(counts, width);
            uint64_t actual = 0, best = 0;
            for (int i = chosen; i < chosen + width; ++i) actual += counts[i];
            for (int start = 0; start <= 256 - width; ++start) {
                uint64_t sum = 0;
                for (int j = 0; j < width; ++j) sum += counts[start + j];
                best = std::max(best, sum);
            }
            assert(actual == best);
        }
    }
    // Exhaust every encoded sign/mantissa at every legal exponent window.
    for (int base = 0; base <= 249; ++base)
        for (int code = 1; code <= 7; ++code)
            for (int sm = 0; sm < 256; ++sm) {
                uint16_t expected = ((sm & 128) << 8) | ((base + code - 1) << 7) | (sm & 127);
                uint16_t actual = ((sm & 128) << 8) |
                    zipserv::exponent_bits(uint8_t(base - 1), code) | (sm & 127);
                assert(actual == expected);
            }
    bool rejected = false;
    try { zipserv::best_exponent_window(counts, 8); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    std::cout << "PASS: shape/overflow contracts, optimal windows, 448000 exact BF16 encodings\n";
}
