// Copyright 2025 The ZipServ Authors. All rights reserved.
// Modified 2026 by Kyrylo Bilchenko: CPU-only regression oracle.
// SPDX-License-Identifier: Apache-2.0
// Frozen encoder from commit 612dc6c. Only CUDA BF16 types/conversions have
// been replaced by raw uint16_t words. Keep its algorithm unchanged so that
// tests and benchmarks compare against the pre-optimization implementation.
#pragma once
#include "../csrc/CodecUtils.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace zipserv_test {
inline int reference_pack_bf16(
    uint16_t* A_bf16,
    int M,
    int K,
    int tile_M,  // 8
    int tile_M_median,  // 16
    int tile_M_global,  // 64
    int tile_K,  // 8
    int tile_K_median,  // 64
    int tile_K_global,  // 64
    const int* top_exponents,  // 7 top frequent exponent values
    uint8_t** sign_mantissa,   // Sign bit + mantissa for high frequency exponents
    uint16_t** compressed_full, // Complete BF16 for non-high frequency exponents
    uint64_t** bitmap1,        // First bitmap
    uint64_t** bitmap2,        // Second bitmap
    uint64_t** bitmap3,        // Third bitmap
    int** TileOffsets,         // Small tile offsets
    int** TileOffsets_median,  // Medium tile offsets
    int** TileOffsets_global,  // Large tile offsets
    int& max_high_freq_count,  // Return max high frequency element count
    int& max_full_count)       // Return max non-high frequency element count
{
    // The device layout is fixed; accepting other tiles silently changes the ABI.
    if (!A_bf16 || !zipserv::valid_matrix(M, K) ||
        tile_M != 8 || tile_K != 8 || tile_M_median != 16 || tile_K_median != 64 ||
        tile_M_global != 64 || tile_K_global != 64 || !zipserv::valid_exponents(top_exponents) ||
        !sign_mantissa || !compressed_full || !bitmap1 || !bitmap2 || !bitmap3 ||
        !TileOffsets || !TileOffsets_median || !TileOffsets_global)
        return -1;
    // Calculate number of tiles
    int num_tiles_M = M / tile_M;
    int num_tiles_K = K / tile_K;
    int num_tiles = num_tiles_M * num_tiles_K;

    int num_median_tiles_M = M / tile_M_median;
    int num_median_tiles_K = K / tile_K_median;
    int num_median_tiles = num_median_tiles_M * num_median_tiles_K;

    int num_global_tiles_M = M / tile_M_global;
    int num_global_tiles_K = K / tile_K_global;
    int num_global_tiles = num_global_tiles_M * num_global_tiles_K;

    // Memory allocation
    *compressed_full = (uint16_t*)malloc(M * K * sizeof(uint16_t));
    *sign_mantissa = (uint8_t*)malloc(M * K);
    *bitmap1 = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));
    *bitmap2 = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));
    *bitmap3 = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));
    *TileOffsets = (int*)malloc(num_tiles * 2 * sizeof(int));
    *TileOffsets_median = (int*)malloc(num_median_tiles * 2 * sizeof(int));
    *TileOffsets_global = (int*)malloc((num_global_tiles + 1) * 2 * sizeof(int));

    if (*compressed_full == nullptr || *sign_mantissa == nullptr ||
        *bitmap1 == nullptr || *bitmap2 == nullptr || *bitmap3 == nullptr ||
        *TileOffsets == nullptr || *TileOffsets_median == nullptr ||
        *TileOffsets_global == nullptr) {
        return -1;
    }

    // Initialize memory
    memset(*compressed_full, 0, M * K * sizeof(uint16_t));
    memset(*sign_mantissa, 0, M * K);
    memset(*bitmap1, 0, num_tiles * sizeof(uint64_t));
    memset(*bitmap2, 0, num_tiles * sizeof(uint64_t));
    memset(*bitmap3, 0, num_tiles * sizeof(uint64_t));
    memset(*TileOffsets, 0, num_tiles * 2 * sizeof(int));
    memset(*TileOffsets_median, 0, num_median_tiles * 2 * sizeof(int));
    memset(*TileOffsets_global, 0, (num_global_tiles + 1) * 2 * sizeof(int));

    // Current offsets
    int full_offset = 0;
    int sign_mantissa_offset = 0;
    int tile_idx = 0;
    int median_offset_idx = 0;
    std::vector<int> global_high_freq_counts(num_global_tiles + 1, 0);
    std::vector<int> global_full_counts(num_global_tiles + 1, 0);

    max_high_freq_count = 0;
    max_full_count = 0;

    // Iterate over all global tiles
    for (int global_tile_m = 0; global_tile_m < num_global_tiles_M; ++global_tile_m) {
        for (int global_tile_k = 0; global_tile_k < num_global_tiles_K; ++global_tile_k) {
            int global_row_start = global_tile_m * tile_M_global;
            int global_col_start = global_tile_k * tile_K_global;
            int global_high_freq_count = 0;
            int global_full_count = 0;

            int median_high_freq_count = 0;
            int median_full_count = 0;

            // Store medium tile offsets
            (*TileOffsets_median)[median_offset_idx * 2] = 0;
            (*TileOffsets_median)[median_offset_idx * 2 + 1] = 0;
            median_offset_idx++;

            // Process medium tiles
            for (int median_tile_m = 0; median_tile_m < tile_M_global / tile_M_median; ++median_tile_m) {
                for (int median_tile_k = 0; median_tile_k < tile_K_global / tile_K_median; ++median_tile_k) {
                    int median_row_start = global_row_start + median_tile_m * tile_M_median;
                    int median_col_start = global_col_start + median_tile_k * tile_K_median;

                    int local_median_high_freq = 0;
                    int local_median_full = 0;

                    // Process 2x2 small tile groups
                    for (int local_tile_m_group = 0; local_tile_m_group < tile_M_median / tile_M; local_tile_m_group += 2) {
                        for (int local_tile_k_group = 0; local_tile_k_group < tile_K_median / tile_K; local_tile_k_group += 2) {
                            // Process 2x2 small tiles in column-major order
                            for (int j = 0; j < 2; ++j) {
                                for (int i = 0; i < 2; ++i) {
                                    int local_tile_k = local_tile_k_group + j;
                                    int local_tile_m = local_tile_m_group + i;

                                    int col_start = median_col_start + local_tile_k * tile_K;
                                    int row_start = median_row_start + local_tile_m * tile_M;

                                    uint64_t tile_bitmap1 = 0;
                                    uint64_t tile_bitmap2 = 0;
                                    uint64_t tile_bitmap3 = 0;
                                    int tile_high_freq_count = 0;
                                    int tile_full_count = 0;

                                    // Process all elements in small tile
                                    for (int row_offset = 0; row_offset < tile_M; ++row_offset) {
                                        for (int col_offset = 0; col_offset < tile_K; ++col_offset) {
                                            int row = row_start + row_offset;
                                            int col = col_start + col_offset;
                                            int pos = row_offset * tile_K + col_offset;

                                            if (row < M && col < K) {
                                                uint16_t val = A_bf16[row * K + col];

                                                // Extract BF16 components
                                                uint16_t bf16_bits = val;
                                                uint8_t sign = (bf16_bits >> 15) & 0x1;
                                                uint8_t exponent = (bf16_bits >> 7) & 0xFF;
                                                uint8_t mantissa = bf16_bits & 0x7F;

                                                // Find exponent position in high frequency list
                                                int exp_idx = -1;
                                                for (int e = 0; e < 7; e++) {
                                                    if (exponent == top_exponents[e]) {
                                                        exp_idx = e;
                                                        break;
                                                    }
                                                }

                                                bool is_high_freq = (exp_idx >= 0);

                                                if (is_high_freq) {
                                                    // High frequency exponent element
                                                    int bitmap_code = exp_idx + 1;  // 1-7

                                                    // Set three bitmaps
                                                    tile_bitmap1 |= ((bitmap_code & 0x1) ? 1ULL << pos : 0);
                                                    tile_bitmap2 |= ((bitmap_code & 0x2) ? 1ULL << pos : 0);
                                                    tile_bitmap3 |= ((bitmap_code & 0x4) ? 1ULL << pos : 0);

                                                    // Store sign+mantissa
                                                    uint8_t combined = ((sign & 0x1) << 7) | (mantissa & 0x7F);
                                                    (*sign_mantissa)[sign_mantissa_offset++] = combined;

                                                    tile_high_freq_count++;
                                                    local_median_high_freq++;
                                                    global_high_freq_count++;
                                                } else {
                                                    // Non-high frequency exponent element
                                                    (*compressed_full)[full_offset++] = val;

                                                    // Bitmap remains 000
                                                    tile_full_count++;
                                                    local_median_full++;
                                                    global_full_count++;
                                                }
                                            }
                                        }
                                    }

                                    // Store bitmaps and element counts
                                    (*bitmap1)[tile_idx] = tile_bitmap1;
                                    (*bitmap2)[tile_idx] = tile_bitmap2;
                                    (*bitmap3)[tile_idx] = tile_bitmap3;
                                    (*TileOffsets)[tile_idx * 2] = tile_high_freq_count;
                                    (*TileOffsets)[tile_idx * 2 + 1] = tile_full_count;
                                    ++tile_idx;
                                }
                            }
                        }
                    }

                    // Update medium tile offsets
                    if (median_tile_m < (tile_M_global / tile_M_median - 1) ||
                        median_tile_k < (tile_K_global / tile_K_median - 1)) {

                        median_high_freq_count += local_median_high_freq;
                        median_full_count += local_median_full;

                        (*TileOffsets_median)[median_offset_idx * 2] = median_high_freq_count;
                        (*TileOffsets_median)[median_offset_idx * 2 + 1] = median_full_count;
                        median_offset_idx++;
                    }
                }
            }

            // Add padding for high frequency elements (multiple of 16)
            int high_freq_padding = (16 - (global_high_freq_count % 16)) % 16;
            for (int p = 0; p < high_freq_padding; ++p) {
                (*sign_mantissa)[sign_mantissa_offset++] = 0;
            }
            global_high_freq_count += high_freq_padding;

            // Add padding for non-high frequency elements (multiple of 8)
            int full_padding = (8 - (global_full_count % 8)) % 8;
            for (int p = 0; p < full_padding; ++p) {
                (*compressed_full)[full_offset++] = uint16_t(0);
            }
            global_full_count += full_padding;

            // Record global tile counts
            global_high_freq_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1] = global_high_freq_count;
            global_full_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1] = global_full_count;

            // Update max counts
            if (global_high_freq_count > max_high_freq_count) {
                max_high_freq_count = global_high_freq_count;
            }
            if (global_full_count > max_full_count) {
                max_full_count = global_full_count;
            }
        }
    }

    // Calculate global tile cumulative offsets
    (*TileOffsets_global)[0] = 0;
    (*TileOffsets_global)[1] = 0;

    for (int i = 1; i <= num_global_tiles; ++i) {
        global_high_freq_counts[i] += global_high_freq_counts[i - 1];
        global_full_counts[i] += global_full_counts[i - 1];

        (*TileOffsets_global)[i * 2] = global_high_freq_counts[i];
        (*TileOffsets_global)[i * 2 + 1] = global_full_counts[i];
    }

    // Resize arrays
    if (void* resized = realloc(*sign_mantissa, std::max(1, sign_mantissa_offset)))
        *sign_mantissa = static_cast<uint8_t*>(resized);
    if (void* resized = realloc(*compressed_full, std::max(1, full_offset) * sizeof(uint16_t)))
        *compressed_full = static_cast<uint16_t*>(resized);

    return num_global_tiles;
}
struct ReferencePacked {
    uint8_t* common = nullptr;
    uint16_t* rare = nullptr;
    uint64_t *b1 = nullptr, *b2 = nullptr, *b3 = nullptr;
    int *small = nullptr, *medium = nullptr, *global = nullptr;
    int max_common = 0, max_rare = 0, tiles = 0;
    ReferencePacked(const uint16_t* words, int m, int k, int base) {
        int exponents[7];
        for (int i = 0; i < 7; ++i) exponents[i] = base + i;
        tiles = reference_pack_bf16(const_cast<uint16_t*>(words), m, k,
            8, 16, 64, 8, 64, 64, exponents, &common, &rare, &b1, &b2, &b3,
            &small, &medium, &global, max_common, max_rare);
        if (tiles <= 0) throw std::runtime_error("reference packing failed");
    }
    ReferencePacked(const ReferencePacked&) = delete;
    ~ReferencePacked() {
        free(common); free(rare); free(b1); free(b2); free(b3);
        free(small); free(medium); free(global);
    }
    std::array<const void*, 8> buffers() const {
        return {common, rare, b1, b2, b3, small, medium, global};
    }
};
}  // namespace zipserv_test
