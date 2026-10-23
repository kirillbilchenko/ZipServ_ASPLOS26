// Copyright 2025 The ZipServ Authors. All rights reserved.
// Copyright 2026 Kyrylo Bilchenko
// SPDX-License-Identifier: Apache-2.0
// CPU-only encoder for the existing ZipServ triple-bitmap layout. Input words
// are raw BF16 bits in host byte order; no floating-point conversion is used.
#pragma once
#include "CodecUtils.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace zipserv {
namespace host_detail {
struct Free {
    void operator()(void* p) const noexcept { std::free(p); }
};
template<class T> using Buffer = std::unique_ptr<T, Free>;
template<class T> Buffer<T> allocate(size_t count) {
    auto* p = static_cast<T*>(std::malloc(std::max(size_t(1), count) * sizeof(T)));
    if (!p) throw std::bad_alloc();
    return Buffer<T>(p);
}
inline uint16_t word_at(const void* words, size_t index) noexcept {
    // memcpy avoids aliasing CUDA's BF16 class through a uint16_t pointer.
    uint16_t word;
    std::memcpy(&word, static_cast<const unsigned char*>(words) + index * 2, 2);
    return word;
}
inline unsigned worker_count(int tiles, unsigned requested) {
    if (requested > 64) throw std::invalid_argument("packing workers must be in [0, 64]");
    if (requested) return std::min(requested, unsigned(tiles));
    // Bound startup concurrency and avoid thread creation for small matrices.
    return std::min({8U, std::max(1U, std::thread::hardware_concurrency()),
                     unsigned(std::max(1, tiles / 128))});
}
// Callbacks must not throw. All allocation and validation happens before work
// starts. Join already-created threads even when a later thread cannot start.
template<class Function>
void parallel_ranges(int count, unsigned workers, const Function& function) {
    auto run = [&](unsigned worker) noexcept {
        const int begin = int(int64_t(count) * worker / workers);
        const int end = int(int64_t(count) * (worker + 1) / workers);
        function(begin, end, worker);
    };
    if (workers == 1) { run(0); return; }
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    try {
        for (unsigned i = 0; i + 1 < workers; ++i) threads.emplace_back(run, i);
    } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
    }
    run(workers - 1);
    for (auto& thread : threads) thread.join();
}
inline int choose_base(const void* words, int count, unsigned workers) {
    std::vector<std::array<uint64_t, 256>> histograms(workers);
    parallel_ranges(count, workers, [&](int begin, int end, unsigned worker) noexcept {
        auto& counts = histograms[worker];
        for (int i = begin; i < end; ++i) ++counts[(word_at(words, i) >> 7) & 255];
    });
    for (unsigned worker = 1; worker < workers; ++worker)
        for (int e = 0; e < 256; ++e) histograms[0][e] += histograms[worker][e];
    return best_exponent_window(histograms[0].data());
}
}  // namespace host_detail

struct HostPacked {
    host_detail::Buffer<uint8_t> common;
    host_detail::Buffer<uint16_t> rare;
    host_detail::Buffer<uint64_t> b1, b2, b3;
    host_detail::Buffer<int> small, medium, global;
    int tiles = 0, base = 0, max_common = 0, max_rare = 0;
    std::array<const void*, 8> buffers() const {
        return {common.get(), rare.get(), b1.get(), b2.get(), b3.get(),
                small.get(), medium.get(), global.get()};
    }
    std::array<size_t, 8> lengths() const {
        return {size_t(global.get()[2 * tiles]), size_t(global.get()[2 * tiles + 1]) * 2,
                size_t(tiles) * 512, size_t(tiles) * 512, size_t(tiles) * 512,
                small ? size_t(tiles) * 512 : 0, size_t(tiles) * 32,
                size_t(tiles + 1) * 8};
    }
};

// base == -1 selects the same optimal seven-exponent window as the bridge.
// workers == 0 selects up to eight workers, with serial packing below 1M words.
// Small-tile counts are optional: the GPU consumes only medium/global offsets.
// The input must remain immutable until this synchronous call returns.
inline HostPacked pack_bf16(const void* words, int m, int k, int base = -1,
                           bool include_small_offsets = false, unsigned requested_workers = 0) {
    if (!words || !valid_matrix(m, k) || base < -1 || base > 249)
        throw std::invalid_argument("BF16 packing requires a valid 64-aligned matrix and exponent window");
    HostPacked out;
    out.tiles = (m / 64) * (k / 64);
    const unsigned workers = host_detail::worker_count(out.tiles, requested_workers);
    out.base = base < 0 ? host_detail::choose_base(words, m * k, workers) : base;
    out.global = host_detail::allocate<int>(size_t(out.tiles + 1) * 2);
    int* offsets = out.global.get();
    offsets[0] = offsets[1] = 0;
    // First count each 64x64 tile. Rounded counts become exclusive offsets, so
    // encoding workers can write directly into disjoint, exactly-sized slices.
    host_detail::parallel_ranges(out.tiles, workers, [&](int begin, int end, unsigned) noexcept {
        for (int tile = begin; tile < end; ++tile) {
            const int row = (tile / (k / 64)) * 64;
            const int col = (tile % (k / 64)) * 64;
            int common = 0;
            for (int r = 0; r < 64; ++r)
                for (int c = 0; c < 64; ++c) {
                    const unsigned e = (host_detail::word_at(words, (row + r) * k + col + c) >> 7) & 255;
                    common += e - unsigned(out.base) < 7;
                }
            offsets[2 * (tile + 1)] = (common + 15) & ~15;
            offsets[2 * (tile + 1) + 1] = (4096 - common + 7) & ~7;
        }
    });
    for (int tile = 0; tile < out.tiles; ++tile) {
        out.max_common = std::max(out.max_common, offsets[2 * (tile + 1)]);
        out.max_rare = std::max(out.max_rare, offsets[2 * (tile + 1) + 1]);
        offsets[2 * (tile + 1)] += offsets[2 * tile];
        offsets[2 * (tile + 1) + 1] += offsets[2 * tile + 1];
    }
    out.common = host_detail::allocate<uint8_t>(offsets[2 * out.tiles]);
    out.rare = host_detail::allocate<uint16_t>(offsets[2 * out.tiles + 1]);
    out.b1 = host_detail::allocate<uint64_t>(size_t(out.tiles) * 64);
    out.b2 = host_detail::allocate<uint64_t>(size_t(out.tiles) * 64);
    out.b3 = host_detail::allocate<uint64_t>(size_t(out.tiles) * 64);
    out.medium = host_detail::allocate<int>(size_t(out.tiles) * 8);
    if (include_small_offsets) out.small = host_detail::allocate<int>(size_t(out.tiles) * 128);
    host_detail::parallel_ranges(out.tiles, workers, [&](int begin, int end, unsigned) noexcept {
        for (int tile = begin; tile < end; ++tile) {
            const int row = (tile / (k / 64)) * 64;
            const int col = (tile % (k / 64)) * 64;
            const int common_start = offsets[2 * tile], rare_start = offsets[2 * tile + 1];
            int common = common_start, rare = rare_start;
            int small = tile * 64;
            for (int medium = 0; medium < 4; ++medium) {
                out.medium.get()[tile * 8 + medium * 2] = common - common_start;
                out.medium.get()[tile * 8 + medium * 2 + 1] = rare - rare_start;
                // Preserve the original column-major 2x2 small-tile ordering.
                for (int group = 0; group < 4; ++group)
                    for (int column = 0; column < 2; ++column)
                        for (int row_tile = 0; row_tile < 2; ++row_tile, ++small) {
                            uint64_t b1 = 0, b2 = 0, b3 = 0;
                            const int previous_common = common, previous_rare = rare;
                            for (int r = 0; r < 8; ++r)
                                for (int c = 0; c < 8; ++c) {
                                    const int index = (row + medium * 16 + row_tile * 8 + r) * k +
                                                      col + group * 16 + column * 8 + c;
                                    const uint16_t word = host_detail::word_at(words, index);
                                    const unsigned exponent_index = unsigned((word >> 7) & 255) - unsigned(out.base);
                                    if (exponent_index < 7) {
                                        const unsigned code = exponent_index + 1;
                                        const int pos = r * 8 + c;
                                        b1 |= uint64_t(code & 1) << pos;
                                        b2 |= uint64_t((code >> 1) & 1) << pos;
                                        b3 |= uint64_t((code >> 2) & 1) << pos;
                                        out.common.get()[common++] = uint8_t((word & 127) | ((word >> 8) & 128));
                                    } else {
                                        out.rare.get()[rare++] = word;
                                    }
                                }
                            out.b1.get()[small] = b1;
                            out.b2.get()[small] = b2;
                            out.b3.get()[small] = b3;
                            if (out.small) {
                                out.small.get()[2 * small] = common - previous_common;
                                out.small.get()[2 * small + 1] = rare - previous_rare;
                            }
                        }
            }
            // Only padding needs clearing; every other output byte is written.
            std::fill(out.common.get() + common, out.common.get() + offsets[2 * (tile + 1)], uint8_t(0));
            std::fill(out.rare.get() + rare, out.rare.get() + offsets[2 * (tile + 1) + 1], uint16_t(0));
        }
    });
    return out;
}
}  // namespace zipserv
