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
#include "MatMulUtilities.cuh"
#include "CodecUtils.h"
#include <vector>
#include <inttypes.h>
#define __STDC_FORMAT_MACROS

template<typename OutputType>
__device__ __forceinline__ OutputType zipserv_partial(float value) { return value; }

template<>
__device__ __forceinline__ __nv_bfloat16 zipserv_partial<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}


__device__ __forceinline__ void LoadBF16FragWithTripleBitmap_Optimized(  // good version
    uint32_t __restrict__ a[][4],
    const uint8_t* __restrict__ SharedSignMantissa,
    const __nv_bfloat16* __restrict__ SharedFullValues,
    const uint64_t* __restrict__ SharedBitmap1,
    const uint64_t* __restrict__ SharedBitmap2,
    const uint64_t* __restrict__ SharedBitmap3,
    int high_freq_start, int full_start,
    bool Pred = true)
{
    int lane_id = threadIdx.x % 32;
    
    if (Pred) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int tile_idx = i * 4 + j;
                uint64_t bitmap1 = SharedBitmap1[tile_idx];
                uint64_t bitmap2 = SharedBitmap2[tile_idx];
                uint64_t bitmap3 = SharedBitmap3[tile_idx];
                uint64_t high_freq_indicator = bitmap1 | bitmap2 | bitmap3;
                
                int lid_offset = lane_id << 1;
                int pos1 = lid_offset;
                int pos2 = lid_offset + 1;
                
                // Precompute code values to reduce dependency chains
                uint8_t code_pos1 = ((bitmap3 >> pos1) & 1ULL) << 2 | 
                                   ((bitmap2 >> pos1) & 1ULL) << 1 | 
                                   ((bitmap1 >> pos1) & 1ULL);
                uint8_t code_pos2 = ((bitmap3 >> pos2) & 1ULL) << 2 | 
                                   ((bitmap2 >> pos2) & 1ULL) << 1 | 
                                   ((bitmap1 >> pos2) & 1ULL);

                // Precompute exponent bits
                uint16_t exponent_bits_pos1 = (122 + code_pos1) << 7;
                uint16_t exponent_bits_pos2 = (122 + code_pos2) << 7;
                
                // Calculate bitmasks for __popcll computation
                uint64_t mask_before_pos1 = (1ULL << pos1) - 1;
                uint64_t mask_before_pos2 = (1ULL << pos2) - 1;
                
                // Calculate number of high-frequency elements before pos1
                int high_freq_before_pos1 = __popcll(high_freq_indicator & mask_before_pos1);
                
                // Calculate total number of elements before pos1, subtract high-freq count to get low-freq count
                int low_freq_before_pos1 = pos1 - high_freq_before_pos1;
                
                // Check if position pos1 is a high-frequency element
                bool is_high_freq_pos1 = (high_freq_indicator & (1ULL << pos1)) != 0;
                
                // Load value for pos1
                __nv_bfloat16 val1;
                if (is_high_freq_pos1) {
                    // High-frequency element - load sign and mantissa from SharedSignMantissa
                    uint8_t combined = SharedSignMantissa[high_freq_start + high_freq_before_pos1];
                    uint8_t sign = (combined >> 7) & 0x1;
                    uint8_t mantissa = combined & 0x7F;
                    
                    // Use precomputed exponent bits
                    uint16_t bf16_bits = ((sign & 0x1) << 15) | exponent_bits_pos1 | (mantissa & 0x7F);
                    val1 = __ushort_as_bfloat16(bf16_bits);
                } else {
                    // Non-high-frequency element - load complete BF16 value directly
                    val1 = SharedFullValues[full_start + low_freq_before_pos1];
                }
                
                // Now process pos2
                // Update count, if pos1 is high-freq, high_freq_before_pos2 needs +1
                int high_freq_before_pos2;
                if (is_high_freq_pos1) {
                    high_freq_before_pos2 = high_freq_before_pos1 + 1;
                } else {
                    high_freq_before_pos2 = high_freq_before_pos1;
                }
                
                int low_freq_before_pos2 = pos2 - high_freq_before_pos2;
                
                // Check if position pos2 is a high-frequency element
                bool is_high_freq_pos2 = (high_freq_indicator & (1ULL << pos2)) != 0;
                int num_high_freq_lane_31 = is_high_freq_pos2 + high_freq_before_pos2;
                
                // Load value for pos2
                __nv_bfloat16 val2;
                if (is_high_freq_pos2) {
                    // High-frequency element - load sign and mantissa from SharedSignMantissa
                    uint8_t combined = SharedSignMantissa[high_freq_start + high_freq_before_pos2];
                    uint8_t sign = (combined >> 7) & 0x1;
                    uint8_t mantissa = combined & 0x7F;
                    
                    // Use precomputed exponent bits
                    uint16_t bf16_bits = ((sign & 0x1) << 15) | exponent_bits_pos2 | (mantissa & 0x7F);
                    val2 = __ushort_as_bfloat16(bf16_bits);
                } else {
                    // Non-high-frequency element - load complete BF16 value directly
                    val2 = SharedFullValues[full_start + low_freq_before_pos2];
                }
                
                // Merge two values into a vector and store in register
                __nv_bfloat162 bf16_pair = make_bfloat162(val1, val2);
                a[i][j] = *reinterpret_cast<const uint32_t*>(&bf16_pair);
                
                // Update starting positions for high-freq and non-high-freq elements
                int num_high_freq = __shfl_sync(0xffffffff, num_high_freq_lane_31, 31);
                int num_full = 64 - num_high_freq;
                full_start += num_full;
                high_freq_start += num_high_freq;
            }
        }
    }
}




// 1. Single row loading function
__device__ __forceinline__ void LoadBF16FragWithTripleBitmap_SingleRow(
    uint32_t __restrict__ a_row[4],
    const uint8_t* __restrict__ SharedSignMantissa,
    const __nv_bfloat16* __restrict__ SharedFullValues,
    const uint64_t* __restrict__ SharedBitmap1,
    const uint64_t* __restrict__ SharedBitmap2,
    const uint64_t* __restrict__ SharedBitmap3,
    const uint8_t start_exp,
    int& high_freq_start,  // Use reference, support accumulation
    int& full_start,       // Use reference, support accumulation
    int row_idx,           // Current row index being loaded (0-3)
    bool Pred = true)
{
    int lane_id = threadIdx.x % 32;
    
    if (Pred) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int tile_idx = row_idx * 4 + j;
            uint64_t bitmap1 = SharedBitmap1[tile_idx];
            uint64_t bitmap2 = SharedBitmap2[tile_idx];
            uint64_t bitmap3 = SharedBitmap3[tile_idx];
            uint64_t high_freq_indicator = bitmap1 | bitmap2 | bitmap3;
            
            int lid_offset = lane_id << 1;
            int pos1 = lid_offset;
            int pos2 = lid_offset + 1;
            
            // Precompute code values
            uint8_t code_pos1 = ((bitmap3 >> pos1) & 1ULL) << 2 | 
                               ((bitmap2 >> pos1) & 1ULL) << 1 | 
                               ((bitmap1 >> pos1) & 1ULL);
            uint8_t code_pos2 = ((bitmap3 >> pos2) & 1ULL) << 2 | 
                               ((bitmap2 >> pos2) & 1ULL) << 1 | 
                               ((bitmap1 >> pos2) & 1ULL);

            // Precompute exponent bits
            uint16_t exponent_bits_pos1 = zipserv::exponent_bits(start_exp, code_pos1);
            uint16_t exponent_bits_pos2 = zipserv::exponent_bits(start_exp, code_pos2);
            
            // Calculate bitmasks
            uint64_t mask_before_pos1 = (1ULL << pos1) - 1;
            uint64_t mask_before_pos2 = (1ULL << pos2) - 1;
            
            // Calculate number of high-frequency elements before pos1
            int high_freq_before_pos1 = __popcll(high_freq_indicator & mask_before_pos1);
            int low_freq_before_pos1 = pos1 - high_freq_before_pos1;
            
            // Check if position pos1 is a high-frequency element
            bool is_high_freq_pos1 = (high_freq_indicator & (1ULL << pos1)) != 0;
            
            // Load value for pos1
            __nv_bfloat16 val1;
            if (is_high_freq_pos1) {
                uint8_t combined = SharedSignMantissa[high_freq_start + high_freq_before_pos1];
                uint8_t sign = (combined >> 7) & 0x1;
                uint8_t mantissa = combined & 0x7F;
                uint16_t bf16_bits = ((sign & 0x1) << 15) | exponent_bits_pos1 | (mantissa & 0x7F);
                val1 = __ushort_as_bfloat16(bf16_bits);
            } else {
                val1 = SharedFullValues[full_start + low_freq_before_pos1];
            }
            
            // Process pos2
            int high_freq_before_pos2;
            if (is_high_freq_pos1) {
                high_freq_before_pos2 = high_freq_before_pos1 + 1;
            } else {
                high_freq_before_pos2 = high_freq_before_pos1;
            }
            
            int low_freq_before_pos2 = pos2 - high_freq_before_pos2;
            bool is_high_freq_pos2 = (high_freq_indicator & (1ULL << pos2)) != 0;
            int num_high_freq_lane_31 = is_high_freq_pos2 + high_freq_before_pos2;
            
            // Load value for pos2
            __nv_bfloat16 val2;
            if (is_high_freq_pos2) {
                uint8_t combined = SharedSignMantissa[high_freq_start + high_freq_before_pos2];
                uint8_t sign = (combined >> 7) & 0x1;
                uint8_t mantissa = combined & 0x7F;
                uint16_t bf16_bits = ((sign & 0x1) << 15) | exponent_bits_pos2 | (mantissa & 0x7F);
                val2 = __ushort_as_bfloat16(bf16_bits);
            } else {
                val2 = SharedFullValues[full_start + low_freq_before_pos2];
            }
            
            // Merge and store
            __nv_bfloat162 bf16_pair = make_bfloat162(val1, val2);
            a_row[j] = *reinterpret_cast<const uint32_t*>(&bf16_pair);
            
            // Update starting positions (accumulate to reference parameters)
            // Each lane has the same complete 64-bit indicator. The optional
            // variant replaces only the total-count broadcast, not prefix ranks.
#ifdef ZIPSERV_EXPERIMENT_POPCOUNT
            int num_high_freq = __popcll(high_freq_indicator);
#else
            int num_high_freq = __shfl_sync(0xffffffff, num_high_freq_lane_31, 31);
#endif
            int num_full = 64 - num_high_freq;
            full_start += num_full;
            high_freq_start += num_high_freq;
        }
    }
}


// Single row loading function
/// 1. Single MMA computation fragment function
template<typename TilingConfig>
__device__ __forceinline__ void SingleMMASlice(
    float c[][REG_PER_C_TENSOR_16_16],
    uint32_t (*a)[4],
    uint32_t (*b)[4],
    int slice_id) // 0,1,2,3 represents position within BLOCK_K_TENSORS
{
    uint32_t (*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
    
    // Calculate correct read pointer - odd slice uses second buffer, even slice uses first buffer
    uint32_t (*a_read)[4] = a + (slice_id % 2) * WARP_ROW_TENSORS_BITMAP_V3;
    uint32_t (*b_read)[4] = b + (slice_id % 2) * TilingConfig::WARP_COL_TENSORS;
    
    #pragma unroll
    for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
        MMA_BF16_M16N8K16(c_uint32_t[j * WARP_ROW_TENSORS_BITMAP_V3], a_read[0], b_read[j]);
        if (!TilingConfig::N8)
            MMA_BF16_M16N8K16(c_uint32_t[j * WARP_ROW_TENSORS_BITMAP_V3] + 4, a_read[0], b_read[j] + 2);
    }
}

// 2. Single data loading fragment function
template<typename TilingConfig>
__device__ __forceinline__ void LoadNextSlice(
    uint32_t (*a)[4],
    uint32_t (*b)[4],
    const uint8_t* __restrict__ SharedSignMantissa,
    const __nv_bfloat16* __restrict__ SharedFullValues,
    const uint64_t* __restrict__ SharedBitmap1_Warp,
    const uint64_t* __restrict__ SharedBitmap2_Warp,
    const uint64_t* __restrict__ SharedBitmap3_Warp,
    const uint8_t start_exp,
    int& high_freq_start,
    int& full_start,
    __nv_bfloat16* __restrict__ SharedMemoryPTR,
    int warp_start_row,
    int warp_start_col,
    int next_slice_id) // Next slice index to load
{
    // Calculate write pointer - important fix: next_slice_id should determine which buffer to use
    uint32_t (*a_write)[4] = a + (next_slice_id % 2) * WARP_ROW_TENSORS_BITMAP_V3;
    uint32_t (*b_write)[4] = b + (next_slice_id % 2) * TilingConfig::WARP_COL_TENSORS;
    
    // Load A fragment
    LoadBF16FragWithTripleBitmap_SingleRow(
        a_write[0], SharedSignMantissa, SharedFullValues,
        SharedBitmap1_Warp, SharedBitmap2_Warp, SharedBitmap3_Warp, start_exp,
        high_freq_start, full_start, next_slice_id % 4);
    
    // Load B fragment
    B_FragLoadFromSharedToRegisters_BF16<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
        b_write, SharedMemoryPTR, warp_start_col, (next_slice_id % 4) * MMA_K);
}

// 3. Modified main kernel function with fine-grained interleaving
template<typename TilingConfig, typename OutputType = __nv_bfloat16>
__global__ void BF16TripleBitmap_MM_Kernel_Fast(
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
    const __nv_bfloat16* B,
    OutputType* Output,
    const int M_Global,
    const int N_Global,
    const int K_Global,
    int Split_K)
{
    const int BatchID = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x = blockIdx.x;
    const int y = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    
    const int NumKBlock = K_Global / TILE_K;
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock = AverageNumKBlock * Split_K;
    const int PaddingKBlock = RoundedKBlock - NumKBlock;
    int NumIter = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;
    
    const int BlockOffset = K_Global / TILE_K * y + BatchID * AverageNumKBlock;
    
    // Calculate shared memory size - add double buffering support
    extern __shared__ __align__(128) __nv_bfloat16 smem1[];
    
    // B matrix double buffering
    __nv_bfloat16* smem_B = smem1;
    // A matrix related data double buffering
    const int bitmap_size = TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3;
    
    // Bitmap double buffering
    uint64_t* smem_Bitmap1 = reinterpret_cast<uint64_t*>(smem_B + (TILE_K * TilingConfig::TILE_N * 2));
    uint64_t* smem_Bitmap2 = smem_Bitmap1 + bitmap_size * 2; // Double buffering
    uint64_t* smem_Bitmap3 = smem_Bitmap2 + bitmap_size * 2; // Double buffering
    
    // Compressed data double buffering
    __nv_bfloat16* smem_FullValues = reinterpret_cast<__nv_bfloat16*>(smem_Bitmap3 + bitmap_size * 2);
    uint8_t* smem_SignMantissa = reinterpret_cast<uint8_t*>(smem_FullValues + max_full_count * 2); // Double buffering
    
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const int Tile_Start_M = y * TilingConfig::TILE_M;
    const int Tile_Start_Bitmap = y * TilingConfig::TILE_BITMAP_M_V3;
    const int Tile_Start_N = x * TilingConfig::TILE_N;
    
    int Warp_i = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS_BITMAP_V3 * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    
    // Register allocation
    uint32_t __restrict__ a[WARP_ROW_TENSORS_BITMAP_V3 * 2][4]; // double buffering
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4]; // double buffering
    
    const int WarpOffset = BlockOffset * 4 + Warp_i;
    int global_tile_idx = BlockOffset;
    
    const int* high_freq_start_ptr = TileOffsets_Global + global_tile_idx * 2;
    const int* full_start_ptr = TileOffsets_Global + global_tile_idx * 2 + 1;
    
    int high_freq_start = high_freq_start_ptr[0];
    int full_start = full_start_ptr[0];
    int high_freq_count = high_freq_start_ptr[2] - high_freq_start;
    int full_count = full_start_ptr[2] - full_start;
    
    const __nv_bfloat16* BTileGlobalPTR = B + Tile_Start_N * K_Global +
        BatchID * AverageNumKBlock * TILE_K;
    
    const uint64_t* Bitmap1TileGlobalPTR = Bitmap1 + Tile_Start_Bitmap * K_Global +
        BatchID * AverageNumKBlock * TilingConfig::TILE_BITMAP_K_V3;
    const uint64_t* Bitmap2TileGlobalPTR = Bitmap2 + Tile_Start_Bitmap * K_Global +
        BatchID * AverageNumKBlock * TilingConfig::TILE_BITMAP_K_V3;
    const uint64_t* Bitmap3TileGlobalPTR = Bitmap3 + Tile_Start_Bitmap * K_Global +
        BatchID * AverageNumKBlock * TilingConfig::TILE_BITMAP_K_V3;
    
    // Initial load into the first double buffer
    CopyTripleBitmapToShared<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
        smem_Bitmap1, smem_Bitmap2, smem_Bitmap3,
        Bitmap1TileGlobalPTR, Bitmap2TileGlobalPTR, Bitmap3TileGlobalPTR);
    cp_async_group_commit();
    
    CopyCompressedDataToShared<TilingConfig>(
        smem_SignMantissa, smem_FullValues,
        SignMantissa + high_freq_start, CompressedFull + full_start,
        high_freq_count, full_count);
    cp_async_group_commit();
    
    // CopyTileFromGlobalToShared_X_64_BF16<TilingConfig::TILE_N2, TilingConfig>(
    //     smem_B, BTileGlobalPTR, K_Global, N_Global);
    
    CopyTileFromGlobalToShared_X_64_BF16<TilingConfig::TILE_N2, TilingConfig>(
        smem_B, BTileGlobalPTR, K_Global);
    cp_async_group_commit();
    
    // Initialize accumulators
    float c[WARP_ROW_TENSORS_BITMAP_V3 * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V3 * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    
    cp_async_wait_group<1>();
    
    const int* median_offset_high_warp_ptr = TileOffsets_Median + WarpOffset * 2;
    const int* median_offset_full_warp_ptr = TileOffsets_Median + WarpOffset * 2 + 1;
    
    // Only prefetch metadata for tiles in this Split-K partition. The offsets
    // array has one terminal entry, not two speculative look-ahead entries.
    int next_high_freq_start = 0, next_full_start = 0;
    int next_high_freq_count = 0, next_full_count = 0;
    if (NumIter > 1) {
        next_high_freq_start = high_freq_start_ptr[2];
        next_full_start = full_start_ptr[2];
        next_high_freq_count = high_freq_start_ptr[4] - next_high_freq_start;
        next_full_count = full_start_ptr[4] - next_full_start;
    }
    
    cp_async_wait_group<0>();
    __syncthreads();
    
    // ====== Preload tile 0 ======
    // Fetch current tile offset
    int current_high_freq_start = median_offset_high_warp_ptr[0];
    int current_full_start = median_offset_full_warp_ptr[0];
    
    // Current warp read pointer
    uint64_t* smem_Bitmap1_Warp = smem_Bitmap1 + Warp_i * 16;
    uint64_t* smem_Bitmap2_Warp = smem_Bitmap2 + Warp_i * 16;
    uint64_t* smem_Bitmap3_Warp = smem_Bitmap3 + Warp_i * 16;
    
    // Preload K=0 data into the first buffer
    LoadNextSlice<TilingConfig>(
        a, b, smem_SignMantissa, smem_FullValues,
        smem_Bitmap1_Warp, smem_Bitmap2_Warp, smem_Bitmap3_Warp,
        start_exp, current_high_freq_start, current_full_start,
        smem_B, warp_start_row, warp_start_col, 0);
    
    #pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
        high_freq_start = next_high_freq_start;
        full_start = next_full_start;
        high_freq_count = next_high_freq_count;
        full_count = next_full_count;
        
        if (tile_id_k + 2 < NumIter) {
            next_high_freq_start = high_freq_start_ptr[(tile_id_k+2)*2];
            next_full_start = full_start_ptr[(tile_id_k+2)*2];
            next_high_freq_count = high_freq_start_ptr[(tile_id_k+3)*2] - next_high_freq_start;
            next_full_count = full_start_ptr[(tile_id_k+3)*2] - next_full_start;
        } else {
            next_high_freq_start = next_full_start = 0;
            next_high_freq_count = next_full_count = 0;
        }
        
        BTileGlobalPTR = BTileGlobalPTR + TILE_K;
        Bitmap1TileGlobalPTR = Bitmap1TileGlobalPTR + TilingConfig::TILE_BITMAP_K_V3;
        Bitmap2TileGlobalPTR = Bitmap2TileGlobalPTR + TilingConfig::TILE_BITMAP_K_V3;
        Bitmap3TileGlobalPTR = Bitmap3TileGlobalPTR + TilingConfig::TILE_BITMAP_K_V3;
        
        // Compute double-buffer pointers
        __nv_bfloat16* __restrict__ smem_write_B_PTR = smem_B + ((tile_id_k + 1) % 2) * (TILE_K * TilingConfig::TILE_N);
        __nv_bfloat16* __restrict__ smem_read_B_PTR = smem_B + ((tile_id_k) % 2) * (TILE_K * TilingConfig::TILE_N);
        
        // Double-buffer pointers for A matrix data
        uint64_t* smem_write_Bitmap1 = smem_Bitmap1 + ((tile_id_k + 1) % 2) * bitmap_size;
        uint64_t* smem_write_Bitmap2 = smem_Bitmap2 + ((tile_id_k + 1) % 2) * bitmap_size;
        uint64_t* smem_write_Bitmap3 = smem_Bitmap3 + ((tile_id_k + 1) % 2) * bitmap_size;
        
        uint64_t* smem_read_Bitmap1 = smem_Bitmap1 + ((tile_id_k) % 2) * bitmap_size;
        uint64_t* smem_read_Bitmap2 = smem_Bitmap2 + ((tile_id_k) % 2) * bitmap_size;
        uint64_t* smem_read_Bitmap3 = smem_Bitmap3 + ((tile_id_k) % 2) * bitmap_size;
        
        __nv_bfloat16* smem_write_FullValues = smem_FullValues + ((tile_id_k + 1) % 2) * max_full_count;
        __nv_bfloat16* smem_read_FullValues = smem_FullValues + ((tile_id_k) % 2) * max_full_count;
        
        uint8_t* smem_write_SignMantissa = smem_SignMantissa + ((tile_id_k + 1) % 2) * max_high_freq_count;
        uint8_t* smem_read_SignMantissa = smem_SignMantissa + ((tile_id_k) % 2) * max_high_freq_count;
        
        // Current warp read pointer
        uint64_t* smem_read_Bitmap1_Warp = smem_read_Bitmap1 + Warp_i * 16;
        uint64_t* smem_read_Bitmap2_Warp = smem_read_Bitmap2 + Warp_i * 16;
        uint64_t* smem_read_Bitmap3_Warp = smem_read_Bitmap3 + Warp_i * 16;
        
        bool GlobalCopy = (tile_id_k + 1) < NumIter;
        
        // ====== Launch async load for the next tile =======
        // Load the next tile data into the write buffer
        CopyTripleBitmapToShared<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
            smem_write_Bitmap1, smem_write_Bitmap2, smem_write_Bitmap3,
            Bitmap1TileGlobalPTR, Bitmap2TileGlobalPTR, Bitmap3TileGlobalPTR, GlobalCopy);
        cp_async_group_commit();
        
        CopyCompressedDataToShared<TilingConfig>(
            smem_write_SignMantissa, smem_write_FullValues,
            SignMantissa + high_freq_start, CompressedFull + full_start,
            high_freq_count, full_count, GlobalCopy);
        cp_async_group_commit();
        
        // CopyTileFromGlobalToShared_X_64_BF16<TilingConfig::TILE_N2, TilingConfig>(
        //     smem_write_B_PTR, BTileGlobalPTR, K_Global, N_Global, GlobalCopy);
        CopyTileFromGlobalToShared_X_64_BF16<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_B_PTR, BTileGlobalPTR, K_Global, GlobalCopy);
        cp_async_group_commit();
        
        // ====== Key fix: improved interleaving =======
        
        // // 1. Load tile 1
        // current_high_freq_start = median_offset_high_warp_ptr[tile_id_k * 8];
        // current_full_start = median_offset_full_warp_ptr[tile_id_k * 8];
        
        LoadNextSlice<TilingConfig>(
            a, b, smem_read_SignMantissa, smem_read_FullValues,
            smem_read_Bitmap1_Warp, smem_read_Bitmap2_Warp, smem_read_Bitmap3_Warp, 
            start_exp, current_high_freq_start, current_full_start,
            smem_read_B_PTR, warp_start_row, warp_start_col, 1);
            
        // 2. Compute tile 0
        SingleMMASlice<TilingConfig>(c, a, b, 0);
        
        
        LoadNextSlice<TilingConfig>(
            a, b, smem_read_SignMantissa, smem_read_FullValues,
            smem_read_Bitmap1_Warp, smem_read_Bitmap2_Warp, smem_read_Bitmap3_Warp, 
            start_exp, current_high_freq_start, current_full_start,
            smem_read_B_PTR, warp_start_row, warp_start_col, 2);
        
        // 4. Compute tile 1
        SingleMMASlice<TilingConfig>(c, a, b, 1);
        
        
        LoadNextSlice<TilingConfig>(
            a, b, smem_read_SignMantissa, smem_read_FullValues,
            smem_read_Bitmap1_Warp, smem_read_Bitmap2_Warp, smem_read_Bitmap3_Warp, 
            start_exp, current_high_freq_start, current_full_start,
            smem_read_B_PTR, warp_start_row, warp_start_col, 3);
        
        // 6. Compute tile 2
        SingleMMASlice<TilingConfig>(c, a, b, 2);
        
        // 7. Wait for the next tile load to complete
        cp_async_wait_group<0>();
        __syncthreads();
        
        // 8. Compute tile 3
        SingleMMASlice<TilingConfig>(c, a, b, 3);
        
        // 9. If another tile exists, load its tile 0 data
        if (GlobalCopy) {
            current_high_freq_start = median_offset_high_warp_ptr[(tile_id_k+1) * 8];
            current_full_start = median_offset_full_warp_ptr[(tile_id_k+1) * 8];
            
            uint64_t* smem_write_Bitmap1_Warp = smem_write_Bitmap1 + Warp_i * 16;
            uint64_t* smem_write_Bitmap2_Warp = smem_write_Bitmap2 + Warp_i * 16;
            uint64_t* smem_write_Bitmap3_Warp = smem_write_Bitmap3 + Warp_i * 16;
            
            LoadNextSlice<TilingConfig>(
                a, b, smem_write_SignMantissa, smem_write_FullValues,
                smem_write_Bitmap1_Warp, smem_write_Bitmap2_Warp, smem_write_Bitmap3_Warp, 
                start_exp, current_high_freq_start, current_full_start,
                smem_write_B_PTR, warp_start_row, warp_start_col, 0);
        }
    }
    
    // Store results using the existing code
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem1);
    StoreToSharedMemoryFromRegisterBitmapV3<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    
    OutputType* OutputPTR = Output + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;
    
    #pragma unroll
    for (int i = warpId; i < TilingConfig::TILE_N2; i += TilingConfig::BLOCK_WARPS)
        #pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)
            OutputPTR[j + i * M_Global] = zipserv_partial<OutputType>((*(smem_CFrag + i))[j]);
}

// 3. Modify the main kernel for fine-grained interleaving
template<typename TilingConfig, typename OutputType = __nv_bfloat16>
__global__ void BF16TripleBitmap_MM_Kernel_Safe(
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
    const __nv_bfloat16* B,
    OutputType* Output,
    const int M_Global,
    const int N_Global,
    const int K_Global,
    int Split_K)
{
    const int BatchID = blockIdx.y / (M_Global / TilingConfig::TILE_M);
    const int IsLastBatch = (BatchID == (Split_K - 1));
    const int x = blockIdx.x;
    const int y = blockIdx.y % (M_Global / TilingConfig::TILE_M);
    
    const int NumKBlock = K_Global / TILE_K;
    const int AverageNumKBlock = (NumKBlock - 1) / Split_K + 1;
    const int RoundedKBlock = AverageNumKBlock * Split_K;
    const int PaddingKBlock = RoundedKBlock - NumKBlock;
    int NumIter = 0;
    if (IsLastBatch)
        NumIter = AverageNumKBlock - PaddingKBlock;
    else
        NumIter = AverageNumKBlock;
    
    const int BlockOffset = K_Global / TILE_K * y + BatchID * AverageNumKBlock;
    
    // Compute shared memory size with double buffering support
    extern __shared__ __align__(128) __nv_bfloat16 smem1[];
    
    // B matrix double buffering
    __nv_bfloat16* smem_B = smem1;

    // // === This explains why B matrix shared memory must be zeroed in advance ===
    if constexpr (true) {  // must always execute
        // Zero the entire shared memory region first
        constexpr int total_b_elements = TILE_K * TilingConfig::TILE_N * 2;
        constexpr int VEC_SIZE = 8;
        constexpr int total_vec_elements = (total_b_elements + VEC_SIZE - 1) / VEC_SIZE;
        
        int tid = threadIdx.x;
        
        #pragma unroll
        for (int i = tid; i < total_vec_elements; i += blockDim.x) {
            int element_idx = i * VEC_SIZE;
            if (element_idx < total_b_elements) {
                uint4* vec_ptr = reinterpret_cast<uint4*>(smem_B + element_idx);
                *vec_ptr = {0, 0, 0, 0};
            }
        }
        __syncthreads(); // Ensure zeroing is complete
        
        // Then perform the copy; untouched areas remain zero
    }




    // Double buffering for A-matrix related data
    const int bitmap_size = TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3;
    
    // Bitmap double buffering
    uint64_t* smem_Bitmap1 = reinterpret_cast<uint64_t*>(smem_B + (TILE_K * TilingConfig::TILE_N * 2));
    uint64_t* smem_Bitmap2 = smem_Bitmap1 + bitmap_size * 2; // double buffering
    uint64_t* smem_Bitmap3 = smem_Bitmap2 + bitmap_size * 2; // double buffering
    
    // Compressed data double buffering
    __nv_bfloat16* smem_FullValues = reinterpret_cast<__nv_bfloat16*>(smem_Bitmap3 + bitmap_size * 2);
    uint8_t* smem_SignMantissa = reinterpret_cast<uint8_t*>(smem_FullValues + max_full_count * 2); // double buffering
    
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const int Tile_Start_M = y * TilingConfig::TILE_M;
    const int Tile_Start_Bitmap = y * TilingConfig::TILE_BITMAP_M_V3;
    const int Tile_Start_N = x * TilingConfig::TILE_N;
    
    int Warp_i = warpId / TilingConfig::BLOCK_COL_WARPS;
    int Warp_j = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_row = WARP_ROW_TENSORS_BITMAP_V3 * MMA_M * Warp_i;
    int warp_start_col = TilingConfig::WARP_COL_TENSORS * MMA_N * Warp_j;
    
    // Register allocation
    uint32_t __restrict__ a[WARP_ROW_TENSORS_BITMAP_V3 * 2][4]; // double buffering
    uint32_t __restrict__ b[TilingConfig::WARP_COL_TENSORS * 2][4]; // double buffering
    
    const int WarpOffset = BlockOffset * 4 + Warp_i;
    int global_tile_idx = BlockOffset;
    
    const int* high_freq_start_ptr = TileOffsets_Global + global_tile_idx * 2;
    const int* full_start_ptr = TileOffsets_Global + global_tile_idx * 2 + 1;
    
    int high_freq_start = high_freq_start_ptr[0];
    int full_start = full_start_ptr[0];
    int high_freq_count = high_freq_start_ptr[2] - high_freq_start;
    int full_count = full_start_ptr[2] - full_start;
    
    const __nv_bfloat16* BTileGlobalPTR = B + Tile_Start_N * K_Global +
        BatchID * AverageNumKBlock * TILE_K;
    
    const uint64_t* Bitmap1TileGlobalPTR = Bitmap1 + Tile_Start_Bitmap * K_Global +
        BatchID * AverageNumKBlock * TilingConfig::TILE_BITMAP_K_V3;
    const uint64_t* Bitmap2TileGlobalPTR = Bitmap2 + Tile_Start_Bitmap * K_Global +
        BatchID * AverageNumKBlock * TilingConfig::TILE_BITMAP_K_V3;
    const uint64_t* Bitmap3TileGlobalPTR = Bitmap3 + Tile_Start_Bitmap * K_Global +
        BatchID * AverageNumKBlock * TilingConfig::TILE_BITMAP_K_V3;
    
    // Initial load into the first double buffer
    CopyTripleBitmapToShared<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
        smem_Bitmap1, smem_Bitmap2, smem_Bitmap3,
        Bitmap1TileGlobalPTR, Bitmap2TileGlobalPTR, Bitmap3TileGlobalPTR);
    cp_async_group_commit();
    
    CopyCompressedDataToShared<TilingConfig>(
        smem_SignMantissa, smem_FullValues,
        SignMantissa + high_freq_start, CompressedFull + full_start,
        high_freq_count, full_count);
    cp_async_group_commit();
    
    CopyTileFromGlobalToShared_X_64_BF16_Safe<TilingConfig::TILE_N2, TilingConfig>(
        smem_B, BTileGlobalPTR, K_Global, N_Global);
    
    cp_async_group_commit();
    
    // Initialize accumulators
    float c[WARP_ROW_TENSORS_BITMAP_V3 * TilingConfig::WARP_COL_TENSORS][REG_PER_C_TENSOR_16_16];
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V3 * TilingConfig::WARP_COL_TENSORS; i++)
        for (int j = 0; j < REG_PER_C_TENSOR_16_16; j++)
            c[i][j] = 0.0f;
    
    cp_async_wait_group<1>();
    
    const int* median_offset_high_warp_ptr = TileOffsets_Median + WarpOffset * 2;
    const int* median_offset_full_warp_ptr = TileOffsets_Median + WarpOffset * 2 + 1;
    
    // Only prefetch metadata for tiles in this Split-K partition. The offsets
    // array has one terminal entry, not two speculative look-ahead entries.
    int next_high_freq_start = 0, next_full_start = 0;
    int next_high_freq_count = 0, next_full_count = 0;
    if (NumIter > 1) {
        next_high_freq_start = high_freq_start_ptr[2];
        next_full_start = full_start_ptr[2];
        next_high_freq_count = high_freq_start_ptr[4] - next_high_freq_start;
        next_full_count = full_start_ptr[4] - next_full_start;
    }
    
    cp_async_wait_group<0>();
    __syncthreads();
    
    // ====== Preload tile 0 ======
    // Fetch current tile offset
    int current_high_freq_start = median_offset_high_warp_ptr[0];
    int current_full_start = median_offset_full_warp_ptr[0];
    
    // Current warp read pointer
    uint64_t* smem_Bitmap1_Warp = smem_Bitmap1 + Warp_i * 16;
    uint64_t* smem_Bitmap2_Warp = smem_Bitmap2 + Warp_i * 16;
    uint64_t* smem_Bitmap3_Warp = smem_Bitmap3 + Warp_i * 16;
    
    // Preload K=0 data into the first buffer
    LoadNextSlice<TilingConfig>(
        a, b, smem_SignMantissa, smem_FullValues,
        smem_Bitmap1_Warp, smem_Bitmap2_Warp, smem_Bitmap3_Warp,
        start_exp, current_high_freq_start, current_full_start,
        smem_B, warp_start_row, warp_start_col, 0);
    
    #pragma unroll(1)
    for (int tile_id_k = 0; tile_id_k < NumIter; tile_id_k++) {
        high_freq_start = next_high_freq_start;
        full_start = next_full_start;
        high_freq_count = next_high_freq_count;
        full_count = next_full_count;
        
        if (tile_id_k + 2 < NumIter) {
            next_high_freq_start = high_freq_start_ptr[(tile_id_k+2)*2];
            next_full_start = full_start_ptr[(tile_id_k+2)*2];
            next_high_freq_count = high_freq_start_ptr[(tile_id_k+3)*2] - next_high_freq_start;
            next_full_count = full_start_ptr[(tile_id_k+3)*2] - next_full_start;
        } else {
            next_high_freq_start = next_full_start = 0;
            next_high_freq_count = next_full_count = 0;
        }
        
        BTileGlobalPTR = BTileGlobalPTR + TILE_K;
        Bitmap1TileGlobalPTR = Bitmap1TileGlobalPTR + TilingConfig::TILE_BITMAP_K_V3;
        Bitmap2TileGlobalPTR = Bitmap2TileGlobalPTR + TilingConfig::TILE_BITMAP_K_V3;
        Bitmap3TileGlobalPTR = Bitmap3TileGlobalPTR + TilingConfig::TILE_BITMAP_K_V3;
        
        // Compute double-buffer pointers
        __nv_bfloat16* __restrict__ smem_write_B_PTR = smem_B + ((tile_id_k + 1) % 2) * (TILE_K * TilingConfig::TILE_N);
        __nv_bfloat16* __restrict__ smem_read_B_PTR = smem_B + ((tile_id_k) % 2) * (TILE_K * TilingConfig::TILE_N);
        
        // Double-buffer pointers for A matrix data
        uint64_t* smem_write_Bitmap1 = smem_Bitmap1 + ((tile_id_k + 1) % 2) * bitmap_size;
        uint64_t* smem_write_Bitmap2 = smem_Bitmap2 + ((tile_id_k + 1) % 2) * bitmap_size;
        uint64_t* smem_write_Bitmap3 = smem_Bitmap3 + ((tile_id_k + 1) % 2) * bitmap_size;
        
        uint64_t* smem_read_Bitmap1 = smem_Bitmap1 + ((tile_id_k) % 2) * bitmap_size;
        uint64_t* smem_read_Bitmap2 = smem_Bitmap2 + ((tile_id_k) % 2) * bitmap_size;
        uint64_t* smem_read_Bitmap3 = smem_Bitmap3 + ((tile_id_k) % 2) * bitmap_size;
        
        __nv_bfloat16* smem_write_FullValues = smem_FullValues + ((tile_id_k + 1) % 2) * max_full_count;
        __nv_bfloat16* smem_read_FullValues = smem_FullValues + ((tile_id_k) % 2) * max_full_count;
        
        uint8_t* smem_write_SignMantissa = smem_SignMantissa + ((tile_id_k + 1) % 2) * max_high_freq_count;
        uint8_t* smem_read_SignMantissa = smem_SignMantissa + ((tile_id_k) % 2) * max_high_freq_count;
        
        // Current warp read pointer
        uint64_t* smem_read_Bitmap1_Warp = smem_read_Bitmap1 + Warp_i * 16;
        uint64_t* smem_read_Bitmap2_Warp = smem_read_Bitmap2 + Warp_i * 16;
        uint64_t* smem_read_Bitmap3_Warp = smem_read_Bitmap3 + Warp_i * 16;
        
        bool GlobalCopy = (tile_id_k + 1) < NumIter;
        
        // ====== Launch async load for the next tile =======
        // Load the next tile data into the write buffer
        CopyTripleBitmapToShared<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
            smem_write_Bitmap1, smem_write_Bitmap2, smem_write_Bitmap3,
            Bitmap1TileGlobalPTR, Bitmap2TileGlobalPTR, Bitmap3TileGlobalPTR, GlobalCopy);
        cp_async_group_commit();
        
        CopyCompressedDataToShared<TilingConfig>(
            smem_write_SignMantissa, smem_write_FullValues,
            SignMantissa + high_freq_start, CompressedFull + full_start,
            high_freq_count, full_count, GlobalCopy);
        cp_async_group_commit();
        
        CopyTileFromGlobalToShared_X_64_BF16_Safe<TilingConfig::TILE_N2, TilingConfig>(
            smem_write_B_PTR, BTileGlobalPTR, K_Global, N_Global, GlobalCopy);
        cp_async_group_commit();
        
        // ====== Key fix: improved interleaving ======
        
        // // 1. Load tile 1
        // current_high_freq_start = median_offset_high_warp_ptr[tile_id_k * 8];
        // current_full_start = median_offset_full_warp_ptr[tile_id_k * 8];
        
        LoadNextSlice<TilingConfig>(
            a, b, smem_read_SignMantissa, smem_read_FullValues,
            smem_read_Bitmap1_Warp, smem_read_Bitmap2_Warp, smem_read_Bitmap3_Warp, 
            start_exp, current_high_freq_start, current_full_start,
            smem_read_B_PTR, warp_start_row, warp_start_col, 1);
            
        // 2. Compute tile 0
        SingleMMASlice<TilingConfig>(c, a, b, 0);
        
        
        LoadNextSlice<TilingConfig>(
            a, b, smem_read_SignMantissa, smem_read_FullValues,
            smem_read_Bitmap1_Warp, smem_read_Bitmap2_Warp, smem_read_Bitmap3_Warp, 
            start_exp, current_high_freq_start, current_full_start,
            smem_read_B_PTR, warp_start_row, warp_start_col, 2);
        
        // 4. Compute tile 1
        SingleMMASlice<TilingConfig>(c, a, b, 1);
        
        
        LoadNextSlice<TilingConfig>(
            a, b, smem_read_SignMantissa, smem_read_FullValues,
            smem_read_Bitmap1_Warp, smem_read_Bitmap2_Warp, smem_read_Bitmap3_Warp, 
            start_exp, current_high_freq_start, current_full_start,
            smem_read_B_PTR, warp_start_row, warp_start_col, 3);
        
        // 6. Compute tile 2
        SingleMMASlice<TilingConfig>(c, a, b, 2);
        
        // 7. Wait for the next tile load to complete
        cp_async_wait_group<0>();
        __syncthreads();
        
        // 8. Compute tile 3
        SingleMMASlice<TilingConfig>(c, a, b, 3);
        
        // 9. If another tile exists, load its tile 0 data
        if (GlobalCopy) {
            current_high_freq_start = median_offset_high_warp_ptr[(tile_id_k+1) * 8];
            current_full_start = median_offset_full_warp_ptr[(tile_id_k+1) * 8];
            
            uint64_t* smem_write_Bitmap1_Warp = smem_write_Bitmap1 + Warp_i * 16;
            uint64_t* smem_write_Bitmap2_Warp = smem_write_Bitmap2 + Warp_i * 16;
            uint64_t* smem_write_Bitmap3_Warp = smem_write_Bitmap3 + Warp_i * 16;
            
            LoadNextSlice<TilingConfig>(
                a, b, smem_write_SignMantissa, smem_write_FullValues,
                smem_write_Bitmap1_Warp, smem_write_Bitmap2_Warp, smem_write_Bitmap3_Warp, 
                start_exp, current_high_freq_start, current_full_start,
                smem_write_B_PTR, warp_start_row, warp_start_col, 0);
        }
    }
    
    // Store results using the existing code
    float(*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C] =
        reinterpret_cast<float(*)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C]>(smem1);
    StoreToSharedMemoryFromRegisterBitmapV3<TilingConfig>(smem_CFrag, c);
    __syncthreads();
    
    // At the kernel's final write-back section:
    OutputType* OutputPTR = Output + BatchID * (M_Global * N_Global) + Tile_Start_M + Tile_Start_N * M_Global;

    // === Fix: write back only the required columns ===
    int actual_cols = min(TilingConfig::TILE_N, N_Global - Tile_Start_N);

    #pragma unroll
    for (int i = warpId; i < actual_cols; i += TilingConfig::BLOCK_WARPS)
        #pragma unroll
        for (int j = threadIdx.x % WARP_SIZE; j < TilingConfig::TILE_M; j += WARP_SIZE)
            OutputPTR[j + i * M_Global] = zipserv_partial<OutputType>((*(smem_CFrag + i))[j]);
}



__device__ __forceinline__ void VectorizedWriteToGlobalMemory(
    const __nv_bfloat16 (*smem_output)[64 + PADDING_SHARED_MEM_FOR_DECOMP],  // 64x64 data in shared memory
    __nv_bfloat16* global_output,      // global memory output
    int global_start_m, int global_start_k,  // global tile start position
    int M_Global, int K_Global)        // global matrix dimensions
{
    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / 32;        // 0-3 identifies the active warp
    const int lane_id = thread_id % 32;        // 0-31 thread ID within the warp
    
    // Arrange each warp's 32 threads into a 4x8 layout
    const int warp_row = lane_id / 8;          // 0-3, row within the warp
    const int warp_col = lane_id % 8;          // 0-7, column within the warp
    
    // 4 warp lanes loop 4 times, each processing a 16x64 block
    #pragma unroll
    for (int cycle = 0; cycle < 4; ++cycle) {
        // Calculate the warp's base position in the 64x64 tile for the current cycle
        int base_row = cycle * 16 + warp_id * 4 + warp_row;  // Each warp processes 4 rows
        int base_col = warp_col * 8;  // Each thread handles 8 columns (one float4)

        int global_row = global_start_m + base_row;
        int global_col = global_start_k + base_col;
            
        // Load 8 bfloat16 values from shared memory as one float4
        const __nv_bfloat16* smem_ptr = smem_output[base_row] + base_col;
        float4 vec = *reinterpret_cast<const float4*>(smem_ptr);
                
        // Write back to global memory
        __nv_bfloat16* global_ptr = &global_output[global_row * K_Global + global_col];
        *reinterpret_cast<float4*>(global_ptr) = vec;  
    }
}
__device__ __forceinline__ void VectorizedWriteToGlobalMemory_128(
    const __nv_bfloat16 (*smem_output)[64 + PADDING_SHARED_MEM_FOR_DECOMP],  // 64x64 data in shared memory
    __nv_bfloat16* global_output,      // global memory output
    int global_start_m, int global_start_k,  // global tile start position
    int M_Global, int K_Global)        // global matrix dimensions
{
    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / 32;        // 0-3 identifies the active warp
    const int lane_id = thread_id % 32;        // 0-31 thread ID within the warp
    
    // Arrange each warp's 32 threads into a 4x8 layout
    const int warp_row = lane_id / 8;          // 0-3, row within the warp
    const int warp_col = lane_id % 8;          // 0-7, column within the warp
    
    // 8 warp lanes loop 4 times, each processing a 32x64 block
    #pragma unroll
    for (int cycle = 0; cycle < 4; ++cycle) {
        // Calculate the warp's base position in the 64x64 tile for the current cycle
        int base_row = cycle * 32 + warp_id * 4 + warp_row;  // Each warp processes 4 rows
        int base_col = warp_col * 8;  // Each thread handles 8 columns (one float4)

        int global_row = global_start_m + base_row;
        int global_col = global_start_k + base_col;
            
        // Load 8 bfloat16 values from shared memory as one float4
        const __nv_bfloat16* smem_ptr = smem_output[base_row] + base_col;
        float4 vec = *reinterpret_cast<const float4*>(smem_ptr);
                
        // Write back to global memory
        __nv_bfloat16* global_ptr = &global_output[global_row * K_Global + global_col];
        *reinterpret_cast<float4*>(global_ptr) = vec;  
    }
}
__device__ __forceinline__ void VectorizedWriteToGlobalMemory_256(
    const __nv_bfloat16 (*smem_output)[64 + PADDING_SHARED_MEM_FOR_DECOMP],  // 64x64 data in shared memory
    __nv_bfloat16* global_output,      // global memory output
    int global_start_m, int global_start_k,  // global tile start position
    int M_Global, int K_Global)        // global matrix dimensions
{
    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / 32;        // 0-3 identifies the active warp
    const int lane_id = thread_id % 32;        // 0-31 thread ID within the warp
    
    // Arrange each warp's 32 threads into a 4x8 layout
    const int warp_row = lane_id / 8;          // 0-3, row within the warp
    const int warp_col = lane_id % 8;          // 0-7, column within the warp
    
    // 16 warp lanes loop 4 times, each processing a 64x64 block
    #pragma unroll
    for (int cycle = 0; cycle < 4; ++cycle) {
        // Calculate the warp's base position in the 64x64 tile for the current cycle
        int base_row = cycle * 64 + warp_id * 4 + warp_row;  // Each warp processes 4 rows
        int base_col = warp_col * 8;  // Each thread handles 8 columns (one float4)

        int global_row = global_start_m + base_row;
        int global_col = global_start_k + base_col;
            
        // Load 8 bfloat16 values from shared memory as one float4
        const __nv_bfloat16* smem_ptr = smem_output[base_row] + base_col;
        float4 vec = *reinterpret_cast<const float4*>(smem_ptr);
                
        // Write back to global memory
        __nv_bfloat16* global_ptr = &global_output[global_row * K_Global + global_col];
        *reinterpret_cast<float4*>(global_ptr) = vec;  
    }
}




template<typename TilingConfig>
__device__ __forceinline__ void DecompressMedianTileToSharedMemory(
    const uint8_t* __restrict__ SharedSignMantissa,
    const __nv_bfloat16* __restrict__ SharedFullValues,
    const uint64_t* __restrict__ SharedBitmap1_Warp,
    const uint64_t* __restrict__ SharedBitmap2_Warp,
    const uint64_t* __restrict__ SharedBitmap3_Warp,
    int high_freq_start,
    int full_start,
    int start_exp,
    __nv_bfloat16 (*smem_output)[64 + PADDING_SHARED_MEM_FOR_DECOMP],  // Pointer to the shared memory output buffer
    int warp_idx)                // Current warp index (0-3)
{
    const int laneId = threadIdx.x % WARP_SIZE;
    // Each thread processes two adjacent elements to optimize popcount usage
    const int lid_offset = laneId << 1;  // laneId * 2
    const int pos1 = lid_offset;
    const int pos2 = lid_offset + 1;
    const int local_m1 = pos1 / 8;
    const int local_k1 = pos1 % 8;
            
    // Process in column-major order matching the compression layout
    #pragma unroll
    for (int col_group = 0; col_group < 8; ++col_group) {       // Group of 4 small tiles per column
        #pragma unroll
        for (int row_group = 0; row_group < 2; ++row_group) {   // 2-row small tile group
            int small_tile_idx = col_group * 2 + row_group;  // Column-major index
            
            // Load the corresponding bitmap
            uint64_t bitmap1 = SharedBitmap1_Warp[small_tile_idx];
            uint64_t bitmap2 = SharedBitmap2_Warp[small_tile_idx];
            uint64_t bitmap3 = SharedBitmap3_Warp[small_tile_idx];
            uint64_t high_freq_indicator = bitmap1 | bitmap2 | bitmap3;
        
            // Precompute the code bits
            uint8_t code_pos1 = ((bitmap3 >> pos1) & 1ULL) << 2 | 
                               ((bitmap2 >> pos1) & 1ULL) << 1 | 
                               ((bitmap1 >> pos1) & 1ULL);
            uint8_t code_pos2 = ((bitmap3 >> pos2) & 1ULL) << 2 | 
                               ((bitmap2 >> pos2) & 1ULL) << 1 | 
                               ((bitmap1 >> pos2) & 1ULL);

            // Precompute exponent bits
            uint16_t exponent_bits_pos1 = zipserv::exponent_bits(start_exp, code_pos1);
            uint16_t exponent_bits_pos2 = zipserv::exponent_bits(start_exp, code_pos2);
            
            // Count high-frequency elements before pos1 (single popcount)
            uint64_t mask_before_pos1 = (1ULL << pos1) - 1;
            int high_freq_before_pos1 = __popcll(high_freq_indicator & mask_before_pos1);
            int low_freq_before_pos1 = pos1 - high_freq_before_pos1;
            
            // Check if position pos1 corresponds to a high-frequency element
            bool is_high_freq_pos1 = (high_freq_indicator & (1ULL << pos1)) != 0;
            
            // Decompress value at pos1
            __nv_bfloat16 val1;
            if (is_high_freq_pos1) {
                uint8_t combined = SharedSignMantissa[high_freq_start + high_freq_before_pos1];
                uint8_t sign = (combined >> 7) & 0x1;
                uint8_t mantissa = combined & 0x7F;
                uint16_t bf16_bits = ((sign & 0x1) << 15) | exponent_bits_pos1 | (mantissa & 0x7F);
                val1 = __ushort_as_bfloat16(bf16_bits);
            } else {
                val1 = SharedFullValues[full_start + low_freq_before_pos1];
            }
            
            // Process pos2 using the results already computed for pos1
            int high_freq_before_pos2;
            if (is_high_freq_pos1) {
                high_freq_before_pos2 = high_freq_before_pos1 + 1;
            } else {
                high_freq_before_pos2 = high_freq_before_pos1;
            }
            
            int low_freq_before_pos2 = pos2 - high_freq_before_pos2;
            bool is_high_freq_pos2 = (high_freq_indicator & (1ULL << pos2)) != 0;
            
            // Decompress value at pos2
            __nv_bfloat16 val2;
            if (is_high_freq_pos2) {
                uint8_t combined = SharedSignMantissa[high_freq_start + high_freq_before_pos2];
                uint8_t sign = (combined >> 7) & 0x1;
                uint8_t mantissa = combined & 0x7F;
                uint16_t bf16_bits = ((sign & 0x1) << 15) | exponent_bits_pos2 | (mantissa & 0x7F);
                val2 = __ushort_as_bfloat16(bf16_bits);
            } else {
                val2 = SharedFullValues[full_start + low_freq_before_pos2];
            }
            // Calculate position within shared memory
            // Shared memory layout: 64x64 block stored row-major
            // Each warp handles 16 rows, each small tile is 8x8
            int small_start_m = warp_idx * 16 + row_group * 8;  // Warp base + small tile row offset
            int small_start_k = col_group * 8;                 // Small tile column offset
            

            // Compute absolute position in shared memory
            int smem_m1 = small_start_m + local_m1;
            int smem_k1 = small_start_k + local_k1;
            // int smem_m2 = small_start_m + local_m2;
            // int smem_k2 = small_start_k + local_k2;
            __nv_bfloat162 bf16_pair = make_bfloat162(val1, val2);
            // Write to shared memory (64x64 layout)
            *reinterpret_cast<__nv_bfloat162*>(smem_output[smem_m1]+smem_k1) = bf16_pair;
            // smem_output[smem_m2 * 64 + smem_k2] = val2;
            
            // Update offsets using the result from thread 31 within the warp for synchronization
            int num_high_freq_lane_31;
            if (is_high_freq_pos2) {
                num_high_freq_lane_31 = high_freq_before_pos2 + 1;
            } else {
                num_high_freq_lane_31 = high_freq_before_pos2;
            }
            
            // Use shuffle to broadcast the thread 31 result
            int total_high_freq = __shfl_sync(0xffffffff, num_high_freq_lane_31, 31);
            int total_full = 64 - total_high_freq;
            // Advance the starting offset
            full_start += total_full;
            high_freq_start += total_high_freq;
        }
    }
}


// Main decompression kernel
template<typename TilingConfig>
__global__ void BF16TripleBitmap_Decompress_Kernel(
    const  uint8_t* __restrict__ SignMantissa,          // Sign bits + mantissa for high-frequency elements
    const  __nv_bfloat16* __restrict__ CompressedFull,  // Full BF16 values for low-frequency elements
    const  uint64_t* __restrict__ Bitmap1,              // First bitmap
    const  uint64_t* __restrict__ Bitmap2,              // Second bitmap
    const  uint64_t* __restrict__ Bitmap3,              // Third bitmap
    // const  int* __restrict__ TileOffsets,               // Small tile offsets
    const  int* __restrict__ TileOffsets_Median,        // Medium tile offsets
    const  int* __restrict__ TileOffsets_Global,        // Global tile offsets
    // const  __nv_bfloat16* __restrict__ TopExponents,    // Top 7 high-frequency exponents
    const int max_high_freq_count,        // Max high-frequency element count
    const int max_full_count,             // Max low-frequency element count
    const uint8_t start_exp,
    __nv_bfloat16* Output,                // Output matrix
    const int M_Global,                   // Global M dimension
    const int K_Global)                   // Global K dimension
{
    // Compute the global tile this block is processing
    const int global_tile_m = blockIdx.y;
    const int global_tile_k = blockIdx.x;
    const int global_tile_idx = global_tile_m * (K_Global / 64) + global_tile_k;
    
    // Compute the global tile's start position in the original matrix
    const int global_start_m = global_tile_m * 64;
    // const int global_start_m = global_tile_m * 128;
    // const int global_start_m = global_tile_m * 256;


    const int global_start_k = global_tile_k * 64;
    
    // Warp information
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // Shared memory allocation
    extern __shared__ __align__(128) __nv_bfloat16 smem_buffer[];
    
    __nv_bfloat16* smem_output = smem_buffer;


    // Bitmap shared memory (entire 64x64 global tile)
    const int bitmap_size = TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3;
    uint64_t* smem_Bitmap1 = reinterpret_cast<uint64_t*>(smem_output + 64*(64+PADDING_SHARED_MEM_FOR_DECOMP));
    // uint64_t* smem_Bitmap1 = reinterpret_cast<uint64_t*>(smem_output + 128*(64+PADDING_SHARED_MEM_FOR_DECOMP));
    // uint64_t* smem_Bitmap1 = reinterpret_cast<uint64_t*>(smem_output + 256*(64+PADDING_SHARED_MEM_FOR_DECOMP));

    uint64_t* smem_Bitmap2 = smem_Bitmap1 + bitmap_size;
    uint64_t* smem_Bitmap3 = smem_Bitmap2 + bitmap_size;
    
    // Compressed data shared memory
    uint8_t* smem_SignMantissa = reinterpret_cast<uint8_t*>(smem_Bitmap3 + bitmap_size);
    const size_t padding = (128 - (max_high_freq_count % 128)) % 128;
    // Decompression output buffer (64x64 BF16)
    __nv_bfloat16* smem_FullValues = reinterpret_cast<__nv_bfloat16*>(smem_SignMantissa + max_high_freq_count + padding);




    const int bitmap_global_offset = (global_start_m >> 3) * (K_Global >> 3) + global_start_k;
    // const int bitmap_global_offset = (global_start_m >> 3) * (K_Global >> 3) + (global_tile_k << 7);  // 128
    // const int bitmap_global_offset = (global_start_m >> 3) * (K_Global >> 3) + (global_tile_k << 8);  // 128



    // Load the current global tile's bitmap data into shared memory
    const uint64_t* Bitmap1TileGlobalPTR = Bitmap1 + bitmap_global_offset;
    const uint64_t* Bitmap2TileGlobalPTR = Bitmap2 + bitmap_global_offset;
    const uint64_t* Bitmap3TileGlobalPTR = Bitmap3 + bitmap_global_offset;
    
    // Get the compressed data offset for the current global tile
    const int* global_offset_ptr = TileOffsets_Global + global_tile_idx * 2;
    int global_high_freq_start = global_offset_ptr[0];
    int global_full_start = global_offset_ptr[1];
    int global_high_freq_count = global_offset_ptr[2] - global_high_freq_start;
    int global_full_count = global_offset_ptr[3] - global_full_start;
    
    // Load bitmaps into shared memory
    CopyTripleBitmapToShared<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
        smem_Bitmap1, smem_Bitmap2, smem_Bitmap3,
        Bitmap1TileGlobalPTR, Bitmap2TileGlobalPTR, Bitmap3TileGlobalPTR);
    // CopyTripleBitmapToShared_128<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
    //     smem_Bitmap1, smem_Bitmap2, smem_Bitmap3,
    //     Bitmap1TileGlobalPTR, Bitmap2TileGlobalPTR, Bitmap3TileGlobalPTR);
    // CopyTripleBitmapToShared_256<TilingConfig::TILE_BITMAP_M_V3, TilingConfig>(
    //     smem_Bitmap1, smem_Bitmap2, smem_Bitmap3,
    //     Bitmap1TileGlobalPTR, Bitmap2TileGlobalPTR, Bitmap3TileGlobalPTR);
    
    // Load compressed data into shared memory
    CopyCompressedDataToShared<TilingConfig>(
        smem_SignMantissa, smem_FullValues,
        SignMantissa + global_high_freq_start, CompressedFull + global_full_start,
        global_high_freq_count, global_full_count);
    cp_async_group_commit();
    cp_async_wait_group<0>();
    __syncthreads();
    
    // Get the median tile data offset for the current warp
    const int median_tile_idx = global_tile_idx * 4 + warpId;
    // const int median_tile_idx = global_tile_idx * 8 + warpId;
    // const int median_tile_idx = global_tile_idx * 16 + warpId;


    const int* median_offset_ptr = TileOffsets_Median + median_tile_idx * 2;
    int warp_high_freq_start = median_offset_ptr[0];
    int warp_full_start = median_offset_ptr[1];
    
    // // Current warp position in the shared memory bitmap
    // // Each median tile corresponds to 2 bitmap rows (16/8=2)
    uint64_t* smem_bitmap1_warp = smem_Bitmap1 + warpId * 2 * 8; // Each warp has 2 rows, 8 small tiles per row
    uint64_t* smem_bitmap2_warp = smem_Bitmap2 + warpId * 2 * 8;
    uint64_t* smem_bitmap3_warp = smem_Bitmap3 + warpId * 2 * 8;
    __nv_bfloat16(*smem_output_2d)[64 + PADDING_SHARED_MEM_FOR_DECOMP] =
        reinterpret_cast<__nv_bfloat16(*)[64 + PADDING_SHARED_MEM_FOR_DECOMP]>(smem_output);
    // // Decompress into shared memory
    DecompressMedianTileToSharedMemory<TilingConfig>(
        smem_SignMantissa, smem_FullValues,
        smem_bitmap1_warp, smem_bitmap2_warp, smem_bitmap3_warp,
        warp_high_freq_start, warp_full_start, start_exp,
        smem_output_2d, warpId);
    
    // // Wait for all warps to finish decompression
    __syncthreads();
    
    // Vectorized write-back to global memory
    VectorizedWriteToGlobalMemory(
        smem_output_2d, Output, global_start_m, global_start_k, M_Global, K_Global);
    // VectorizedWriteToGlobalMemory_128(
    //     smem_output_2d, Output, global_start_m, global_start_k, M_Global, K_Global);
    // VectorizedWriteToGlobalMemory_256(
    //     smem_output_2d, Output, global_start_m, global_start_k, M_Global, K_Global);
}





