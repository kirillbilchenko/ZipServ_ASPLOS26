// Copyright 2026 Kyrylo Bilchenko
// SPDX-License-Identifier: Apache-2.0
// Bounded-memory, CPU-only analysis of raw row-major little-endian BF16.
// Estimates alternative layouts; does not encode them or claim runtime speed.
#include "../csrc/CodecUtils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <vector>

using Histogram = std::array<uint64_t, 256>;
uint64_t payload(int common) {
    return ((common + 15) / 16) * 16 + ((4096 - common + 7) / 8) * 16;
}
template<class Counts> double entropy(const Counts& hist, double n) {
    double result = 0;
    for (auto count : hist) if (count) { double p = count / n; result -= p * std::log2(p); }
    return result;
}
int legacy_base(const Histogram& counts) {
    std::vector<std::pair<int, uint64_t>> pairs;
    for (int i = 0; i < 256; ++i) if (counts[i]) pairs.emplace_back(i, counts[i]);
    std::sort(pairs.begin(), pairs.end(), [](auto a, auto b) { return a.second > b.second; });
    std::vector<int> top;
    for (int i = 0; i < std::min(7, int(pairs.size())); ++i) top.push_back(pairs[i].first);
    while (top.size() < 7) top.push_back(127 - top.size());
    std::sort(top.begin(), top.end());
    for (int start = top.front(); start <= top.back(); ++start) {
        bool present = start + 6 < 256;
        for (int j = 0; j < 7 && present; ++j) present = counts[start + j] != 0;
        if (present) return start;
    }
    int best = top.front(), longest = 1, length = 1;
    for (size_t i = 1; i < top.size(); ++i) {
        length = top[i] == top[i - 1] + 1 ? length + 1 : 1;
        if (length > longest) { longest = length; best = top[i] - length + 1; }
    }
    return best;
}

int main(int argc, char** argv) try {
    if (argc != 4) throw std::invalid_argument("usage: profile_bf16 weights.bf16 M K");
    int m = std::stoi(argv[2]), k = std::stoi(argv[3]);
    if (!zipserv::valid_matrix(m, k)) throw std::invalid_argument("positive 64-aligned int32 shape required");
    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    const uint64_t words = uint64_t(m) * k, bytes = words * 2, tiles = words / 4096;
    if (!input || input.tellg() != std::streamoff(bytes)) throw std::runtime_error("file size does not match shape");
    input.seekg(0);
    std::vector<unsigned char> rows(size_t(64) * k * 2);
    Histogram exponents{}, sm{};
    std::array<uint64_t, 65536> symbols{};
    std::array<Histogram, 4> costs{};  // exact aligned payload for each global window
    uint64_t local_three = 0, local_adaptive = 0;
    for (int row = 0; row < m; row += 64) {
        input.read(reinterpret_cast<char*>(rows.data()), rows.size());
        if (!input) throw std::runtime_error("short read");
        for (int col = 0; col < k; col += 64) {
            Histogram h{};
            for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) {
                size_t p = (size_t(r) * k + col + c) * 2;
                uint16_t bits = rows[p] | (uint16_t(rows[p + 1]) << 8);
                ++h[(bits >> 7) & 255]; ++sm[((bits >> 8) & 128) | (bits & 127)]; ++symbols[bits];
            }
            for (int i = 0; i < 256; ++i) exponents[i] += h[i];
            uint64_t best_adaptive = 8192; // raw tile option
            for (int bits = 1; bits <= 4; ++bits) {
                int width = (1 << bits) - 1, common = 0;
                for (int i = 0; i < width; ++i) common += h[i];
                uint64_t best = payload(common);
                for (int base = 0; base <= 256 - width; ++base) {
                    if (base) { common -= h[base - 1]; common += h[base + width - 1]; }
                    uint64_t cost = payload(common);
                    costs[bits - 1][base] += cost;
                    best = std::min(best, cost);
                }
                if (bits == 3) local_three += best + 1536 + 40 + 1;
                best_adaptive = std::min(best_adaptive, best + uint64_t(bits) * 512 + 40);
            }
            local_adaptive += best_adaptive + 2; // explicit mode and base per tile
        }
    }
    std::priority_queue<uint64_t, std::vector<uint64_t>, std::greater<uint64_t>> queue;
    for (auto n : exponents) if (n) queue.push(n);
    uint64_t huffman_bits = 0;
    while (queue.size() > 1) {
        uint64_t a = queue.top(); queue.pop(); uint64_t b = queue.top(); queue.pop();
        huffman_bits += a + b; queue.push(a + b);
    }
    int base = zipserv::best_exponent_window(exponents.data()), old = legacy_base(exponents);
    uint64_t metadata = tiles * 40 + 8; // median and global indexes
    auto total = [&](int b, int start) { return costs[b - 1][start] + tiles * (b * 512) + metadata; };
    auto item = [&](const char* name, uint64_t size, bool comma = true) {
        std::cout << "  \"" << name << "\": " << size << (comma ? "," : "") << '\n';
    };
    std::cout << std::setprecision(12) << "{\n  \"shape\": [" << m << ", " << k << "],\n";
    item("raw_bytes", bytes);
    std::cout << "  \"optimal_window_base\": " << base << ",\n  \"legacy_window_base\": " << old << ",\n";
    item("legacy_tca_tbe_bytes", old <= 249 ? total(3, old) : 0);
    item("optimal_coverage_tca_tbe_bytes", total(3, base));
    int byte_base = 0;
    for (int i = 1; i <= 249; ++i) if (costs[2][i] < costs[2][byte_base]) byte_base = i;
    item("minimum_padded_global_tca_tbe_bytes", total(3, byte_base));
    item("per_tile_three_bit_estimated_bytes", local_three + 8);
    item("per_tile_adaptive_1_to_4_bits_or_raw_estimated_bytes", local_adaptive + 8);
    std::cout << "  \"global_fixed_width_estimates\": [\n";
    std::vector<uint64_t> sorted(exponents.begin(), exponents.end());
    std::sort(sorted.begin(), sorted.end(), std::greater<uint64_t>());
    for (int b = 1; b <= 4; ++b) {
        int width = (1 << b) - 1, chosen = 0;
        uint64_t common = 0, best = 0, dict_common = 0;
        for (int i = 0; i < width; ++i) { common += exponents[i]; dict_common += sorted[i]; }
        best = common;
        for (int start = 1; start <= 256 - width; ++start) {
            common += exponents[start + width - 1]; common -= exponents[start - 1];
            if (common > best) { best = common; chosen = start; }
        }
        // Dictionary estimate excludes tile padding; explicitly a lower estimate.
        uint64_t dict_bytes = words + (words - dict_common) + words / 8 * b + metadata + width;
        std::cout << "    {\"bits\": " << b << ", \"window_base\": " << chosen
                  << ", \"window_bytes\": " << total(b, chosen)
                  << ", \"dictionary_bytes_without_padding\": " << dict_bytes << "}"
                  << (b == 4 ? "\n" : ",\n");
    }
    std::cout << "  ],\n  \"exponent_entropy_bits\": " << entropy(exponents, words)
              << ",\n  \"sign_mantissa_entropy_bits\": " << entropy(sm, words)
              << ",\n  \"bf16_symbol_entropy_bits\": " << entropy(symbols, words)
              << ",\n  \"exponent_huffman_average_bits\": " << double(huffman_bits) / words
              << ",\n  \"huffman_exponent_plus_raw_sm_bytes_without_indexes\": "
              << words + (huffman_bits + 7) / 8 << ",\n";
    item("tca_tbe_all_common_floor_bytes", words + words / 8 * 3 + metadata, false);
    std::cout << "}\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
