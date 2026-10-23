// Modified 2026 by Kyrylo Bilchenko: correctness and MxWave backports.
/***************************************************************************
 * Copyright 2025 The ZipServ Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Packed inputs must come from InitBF16MatrixTripleBitmap with the fixed tile
// sizes below. M and K are positive multiples of 64; N may be zero. Split_K
// divides K/64. Indices and products must fit signed int32; grid.y <= 65535.
// Inputs, output and (when Split_K > 1) M*N*Split_K BF16 workspace reside on the
// caller's CUDA device. Calls enqueue work on stream; buffers must remain alive
// until it completes. The API validates shapes/pointers, not packed contents.
// Weight reconstruction is bit-exact. GEMM accumulation, especially BF16 split
// partials, can round differently from another backend. No concurrent-host-call
// guarantee is made for per-kernel dynamic shared-memory attribute changes.
cudaError_t BF16TripleBitmap_MM_API(
    cudaStream_t stream,
    const uint8_t* SignMantissa,          // Sign/mantissa for high-frequency elements
    const __nv_bfloat16* CompressedFull,  // Full BF16 values for low-frequency elements
    const uint64_t* Bitmap1,              // Primary bitmap
    const uint64_t* Bitmap2,              // Secondary bitmap
    const uint64_t* Bitmap3,              // Tertiary bitmap
    const int* TileOffsets_Median,        // Medium tile offsets
    const int* TileOffsets_Global,        // Global tile offsets
    const int max_high_freq_count,        // Max high-frequency element count
    const int max_full_count,             // Max low-frequency element count
    const uint8_t start_exp,
    const __nv_bfloat16* B,               // Matrix B
    __nv_bfloat16* C,                     // Output matrix
    const int M_Global,                   // Global M dimension
    const int N_Global,                   // Global N dimension
    const int K_Global,                   // Global K dimension
    __nv_bfloat16* Reduction_Workspace,   // Reduction workspace
    int Split_K);                          // K dimension split count
cudaError_t BF16TripleBitmap_Decompress_API(
    cudaStream_t stream,
    const uint8_t* SignMantissa,
    const __nv_bfloat16* CompressedFull,
    const uint64_t* Bitmap1,
    const uint64_t* Bitmap2,
    const uint64_t* Bitmap3,
    const int* TileOffsets_Median,
    const int* TileOffsets_Global,
    const int max_high_freq_count,
    const int max_full_count,
    const uint8_t start_exp,
    __nv_bfloat16* Output,
    const int M_Global,
    const int K_Global);
// CPU compression returns the number of global tiles, or -1 on failure.
// top_exponents contains seven consecutive integers in [0,255]; decoding uses
// uint8_t(top_exponents[0] - 1) as start_exp. All eight output allocations belong
// to the caller and must be freed with free(), including any partial allocations
// after an allocation failure. Initialize output pointers to nullptr first.
__host__ int InitBF16MatrixTripleBitmap(
    __nv_bfloat16* A_bf16,
    int M,
    int K,
    int tile_M,  // 8
    int tile_M_median,  // 16
    int tile_M_global,  // 64
    int tile_K,  // 8
    int tile_K_median,  // 64
    int tile_K_global,  // 64
    const int* top_exponents,  // 7 top frequent exponent values
    uint8_t** sign_mantissa,   // Sign bit + mantissa for high-freq exponents
    __nv_bfloat16** compressed_full, // Complete BF16 for non-high-freq exponents
    uint64_t** bitmap1,        // First bitmap
    uint64_t** bitmap2,        // Second bitmap 
    uint64_t** bitmap3,        // Third bitmap
    int** TileOffsets,         // Small tile offsets
    int** TileOffsets_median,  // Medium tile offsets
    int** TileOffsets_global,  // Global tile offsets
    int& max_high_freq_count,  // Return max high-freq element count
    int& max_full_count);       // Return max non-high-freq element count

