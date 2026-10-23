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
#include "./MatMulUtilities.cuh"
#include "./Reduction_Kernel.cuh"
#include "./L_Kernel.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "CodecUtils.h"
#include "HostPacker.h"

cudaError_t LaunchKernelWithConfig_4Param(
    cudaStream_t stream,
    const uint8_t* SignMantissa, const __nv_bfloat16* CompressedFull,
    const uint64_t* Bitmap1, const uint64_t* Bitmap2, const uint64_t* Bitmap3,
    const int* TileOffsets_Median, const int* TileOffsets_Global,
    const int max_high_freq_count, const int max_full_count,
    const uint8_t start_exp, const __nv_bfloat16* B, __nv_bfloat16* OutputPTR,
    const int M_Global, const int N_Global, const int K_Global, int Split_K)
{
    using ConfigType = TilingConfigBF16TripleBitmap<4, 1, 1, 1>;
    
    const int SHMEM_SZ = max(
        (max_high_freq_count * sizeof(uint8_t)*2) + 
        (max_full_count * sizeof(__nv_bfloat16)*2) +
        (ConfigType::TILE_N * TILE_K * sizeof(__nv_bfloat16) * 2) + 
        (ConfigType::TILE_BITMAP_M_V3 * ConfigType::TILE_BITMAP_K_V3 * sizeof(uint64_t) * 6),
        (ConfigType::TILE_M + PADDING_SHARED_MEM_FOR_C) * ConfigType::TILE_N * sizeof(float));
    
    int dimN = (N_Global + ConfigType::TILE_N - 1) / ConfigType::TILE_N;
    int dimM = M_Global * Split_K / ConfigType::TILE_M;
    dim3 GridDim(dimN, dimM, 1);
    dim3 BlockDim(WARP_SIZE * ConfigType::BLOCK_WARPS, 1, 1);
    
    // === Key modification: Choose Fast or Safe version based on N_Global ===
    if (N_Global % ConfigType::TILE_N2 == 0) {
        // N is a multiple of TILE_N, use Fast version
        const cudaError_t attribute_error = cudaFuncSetAttribute(
            BF16TripleBitmap_MM_Kernel_Fast<ConfigType>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
        if (attribute_error != cudaSuccess) return attribute_error;
        BF16TripleBitmap_MM_Kernel_Fast<ConfigType><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
            SignMantissa, CompressedFull, Bitmap1, Bitmap2, Bitmap3,
            TileOffsets_Median, TileOffsets_Global,
            max_high_freq_count, max_full_count, start_exp, B, OutputPTR, 
            M_Global, N_Global, K_Global, Split_K);
    } else {
        // When N is not a multiple of TILE_N, use the Safe version
        const cudaError_t attribute_error = cudaFuncSetAttribute(
            BF16TripleBitmap_MM_Kernel_Safe<ConfigType>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
        if (attribute_error != cudaSuccess) return attribute_error;
        BF16TripleBitmap_MM_Kernel_Safe<ConfigType><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
            SignMantissa, CompressedFull, Bitmap1, Bitmap2, Bitmap3,
            TileOffsets_Median, TileOffsets_Global,
            max_high_freq_count, max_full_count, start_exp, B, OutputPTR, 
            M_Global, N_Global, K_Global, Split_K);
    }
    
    return cudaGetLastError();
}

template<int BLOCK_COL_WARPS>
cudaError_t LaunchKernelWithConfig_3Param(
    cudaStream_t stream,
    const uint8_t* SignMantissa, const __nv_bfloat16* CompressedFull,
    const uint64_t* Bitmap1, const uint64_t* Bitmap2, const uint64_t* Bitmap3,
    const int* TileOffsets_Median, const int* TileOffsets_Global,
    const int max_high_freq_count, const int max_full_count,
    const uint8_t start_exp, const __nv_bfloat16* B, __nv_bfloat16* OutputPTR,
    const int M_Global, const int N_Global, const int K_Global, int Split_K)
{
    using ConfigType = TilingConfigBF16TripleBitmap<4, 1, BLOCK_COL_WARPS>;
    
    const int SHMEM_SZ = max(
        (max_high_freq_count * sizeof(uint8_t)*2) + 
        (max_full_count * sizeof(__nv_bfloat16)*2) +
        (ConfigType::TILE_N * TILE_K * sizeof(__nv_bfloat16) * 2) + 
        (ConfigType::TILE_BITMAP_M_V3 * ConfigType::TILE_BITMAP_K_V3 * sizeof(uint64_t) * 6),
        (ConfigType::TILE_M + PADDING_SHARED_MEM_FOR_C) * ConfigType::TILE_N * sizeof(float));
    
    int dimN = (N_Global + ConfigType::TILE_N - 1) / ConfigType::TILE_N;
    int dimM = M_Global * Split_K / ConfigType::TILE_M;
    dim3 GridDim(dimN, dimM, 1);
    dim3 BlockDim(WARP_SIZE * ConfigType::BLOCK_WARPS, 1, 1);
    
    // === Key modification: Choose Fast or Safe version based on N_Global ===
    if (N_Global % ConfigType::TILE_N == 0) {
        // N is a multiple of TILE_N, use Fast version
        const cudaError_t attribute_error = cudaFuncSetAttribute(
            BF16TripleBitmap_MM_Kernel_Fast<ConfigType>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
        if (attribute_error != cudaSuccess) return attribute_error;
        BF16TripleBitmap_MM_Kernel_Fast<ConfigType><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
            SignMantissa, CompressedFull, Bitmap1, Bitmap2, Bitmap3,
            TileOffsets_Median, TileOffsets_Global,
            max_high_freq_count, max_full_count, start_exp, B, OutputPTR, 
            M_Global, N_Global, K_Global, Split_K);
    } else {
        // When N is not a multiple of TILE_N, use the Safe version
        const cudaError_t attribute_error = cudaFuncSetAttribute(
            BF16TripleBitmap_MM_Kernel_Safe<ConfigType>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
        if (attribute_error != cudaSuccess) return attribute_error;
        BF16TripleBitmap_MM_Kernel_Safe<ConfigType><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
            SignMantissa, CompressedFull, Bitmap1, Bitmap2, Bitmap3,
            TileOffsets_Median, TileOffsets_Global,
            max_high_freq_count, max_full_count, start_exp, B, OutputPTR, 
            M_Global, N_Global, K_Global, Split_K);
    }
    
    return cudaGetLastError();
}

cudaError_t BF16TripleBitmap_MM_API(
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
    const __nv_bfloat16* B,               
    __nv_bfloat16* C,                     
    const int M_Global,                   
    const int N_Global,                   
    const int K_Global,                   
    __nv_bfloat16* Reduction_Workspace,   
    int Split_K)                          
{
    if (!zipserv::valid_gemm(M_Global, N_Global, K_Global, Split_K) ||
        !zipserv::valid_counts(max_high_freq_count, max_full_count))
        return cudaErrorInvalidValue;
    if (N_Global == 0) return cudaSuccess;
    if (!Bitmap1 || !Bitmap2 || !Bitmap3 || !TileOffsets_Median || !TileOffsets_Global ||
        !B || !C || (max_high_freq_count && !SignMantissa) ||
        (max_full_count && !CompressedFull) || (Split_K > 1 && !Reduction_Workspace))
        return cudaErrorInvalidValue;
    __nv_bfloat16* OutputPTR;
    if (Split_K == 1)
        OutputPTR = C;
    else
        OutputPTR = Reduction_Workspace;
    
    // === Key modification: Select different Config types based on N_Global ===
    cudaError_t error;
    
    if (N_Global <= 8) {
        // === Special case: Use 4-parameter configuration ===
        error = LaunchKernelWithConfig_4Param(stream, SignMantissa, CompressedFull,
            Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
            max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
            M_Global, N_Global, K_Global, Split_K);
    }
    else if (N_Global > 128) {
        // Greater than 128, use fixed 3-parameter configuration BLOCK_COL_WARPS=8 (TILE_N=128)
        error = LaunchKernelWithConfig_3Param<8>(stream, SignMantissa, CompressedFull,
            Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
            max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
            M_Global, N_Global, K_Global, Split_K);
    }
    else {
        // 9 <= N_Global <= 128, use 3-parameter configuration, BLOCK_COL_WARPS = (N_Global + 15) / 16
        int block_col_warps = (N_Global + 15) / 16;
        
        switch (block_col_warps) {
            case 1:
                error = LaunchKernelWithConfig_3Param<1>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 2:
                error = LaunchKernelWithConfig_3Param<2>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 3:
                error = LaunchKernelWithConfig_3Param<3>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 4:
                error = LaunchKernelWithConfig_3Param<4>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 5:
                error = LaunchKernelWithConfig_3Param<5>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 6:
                error = LaunchKernelWithConfig_3Param<6>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 7:
                error = LaunchKernelWithConfig_3Param<7>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 8:
            default:
                error = LaunchKernelWithConfig_3Param<8>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
        }
    }
    
    if (error != cudaSuccess)
        return error;
    
    // If using Split-K, perform reduction
    if (Split_K > 1) {
        dim3 GridDim((int64_t(M_Global) * N_Global + 255) / 256, 1, 1);
        dim3 BlockDim(WARP_SIZE, 1, 1);
        SplitK_Reduction_BF16<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    }
    
    return cudaGetLastError();
}


// API function
cudaError_t BF16TripleBitmap_Decompress_API(
    cudaStream_t stream,
    const uint8_t* SignMantissa,
    const __nv_bfloat16* CompressedFull,
    const uint64_t* Bitmap1,
    const uint64_t* Bitmap2,
    const uint64_t* Bitmap3,
    /*const int* TileOffsets,*/
    const int* TileOffsets_Median,
    const int* TileOffsets_Global,
    /*const __nv_bfloat16* TopExponents,*/
    const int max_high_freq_count,
    const int max_full_count,
    const uint8_t start_exp,
    __nv_bfloat16* Output,
    const int M_Global,
    const int K_Global)
{
    // Validate input parameters
    if (!zipserv::valid_matrix(M_Global, K_Global) || M_Global / 64 > 65535 ||
        !zipserv::valid_counts(max_high_freq_count, max_full_count) ||
        !Bitmap1 || !Bitmap2 || !Bitmap3 || !TileOffsets_Median || !TileOffsets_Global ||
        !Output || (max_high_freq_count && !SignMantissa) ||
        (max_full_count && !CompressedFull)) {
        printf("Error: Matrix dimensions must be multiples of 64. Got M=%d, K=%d\n", M_Global, K_Global);
        return cudaErrorInvalidValue;
    }
    
    // Calculate grid dimensions
    int num_global_tiles_m = M_Global / 64;
    // int num_global_tiles_m = M_Global / 128;
    // int num_global_tiles_m = M_Global / 256;


    int num_global_tiles_k = K_Global / 64;
    
    dim3 GridDim(num_global_tiles_k, num_global_tiles_m, 1);
    dim3 BlockDim(WARP_SIZE * 4, 1, 1); // 4 warps per block
    // dim3 BlockDim(WARP_SIZE * 8, 1, 1); // 4 warps per block
    // dim3 BlockDim(WARP_SIZE * 16, 1, 1); // 4 warps per block


    
    // Calculate shared memory size
    using Config = TilingConfigBF16TripleBitmap<4, 1, 4>; // Reuse existing configuration
    const int bitmap_size = Config::TILE_BITMAP_M_V3 * Config::TILE_BITMAP_K_V3;
    
    int SHMEM_SZ = 
        (bitmap_size * sizeof(uint64_t) * 3) +           // Three bitmaps
        (max_high_freq_count * sizeof(uint8_t)) +        // sign_mantissa
        (max_full_count * sizeof(__nv_bfloat16)) +       // compressed_full
        (64 * (64+PADDING_SHARED_MEM_FOR_DECOMP) * sizeof(__nv_bfloat16)) + 
        256 * sizeof(uint8_t);               // Decompression output buffer
    // int SHMEM_SZ = 
    //     (bitmap_size * sizeof(uint64_t) * 3) +           // three bitmaps
    //     (max_high_freq_count * sizeof(uint8_t)) +        // sign_mantissa
    //     (max_full_count * sizeof(__nv_bfloat16)) +       // compressed_full
    //     (128 * (64+PADDING_SHARED_MEM_FOR_DECOMP) * sizeof(__nv_bfloat16)) + 
    //     256 * sizeof(uint8_t);               // decompression output buffer    
    // int SHMEM_SZ = 
    //     (bitmap_size * sizeof(uint64_t) * 3) +           // three bitmaps
    //     (max_high_freq_count * sizeof(uint8_t)) +        // sign_mantissa
    //     (max_full_count * sizeof(__nv_bfloat16)) +       // compressed_full
    //     (256 * (64+PADDING_SHARED_MEM_FOR_DECOMP) * sizeof(__nv_bfloat16)) + 
    //     256 * sizeof(uint8_t);               // decompression output buffer    
        // Set dynamic shared memory
    const cudaError_t attribute_error = cudaFuncSetAttribute(
        BF16TripleBitmap_Decompress_Kernel<Config>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    if (attribute_error != cudaSuccess) return attribute_error;
    
    // printf("Launching decompress kernel with grid (%d, %d), block (%d), SHMEM=%d KB\n", 
    //        num_global_tiles_k, num_global_tiles_m, WARP_SIZE * 8, SHMEM_SZ / 1024);
    
    // Launch kernel
    BF16TripleBitmap_Decompress_Kernel<Config><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        SignMantissa, CompressedFull, Bitmap1, Bitmap2, Bitmap3,
        /*TileOffsets,*/ TileOffsets_Median, TileOffsets_Global, /*TopExponents,*/
        max_high_freq_count, max_full_count, start_exp,
        Output, M_Global, K_Global);
    
    return cudaGetLastError();
}

// CPU function for BF16 matrix compression initialization
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
    uint8_t** sign_mantissa,   // Sign bit + mantissa for high frequency exponents
    __nv_bfloat16** compressed_full, // Complete BF16 for non-high frequency exponents
    uint64_t** bitmap1,        // First bitmap
    uint64_t** bitmap2,        // Second bitmap 
    uint64_t** bitmap3,        // Third bitmap
    int** TileOffsets,         // Small tile offsets
    int** TileOffsets_median,  // Medium tile offsets
    int** TileOffsets_global,  // Large tile offsets
    int& max_high_freq_count,  // Return max high frequency element count
    int& max_full_count)       // Return max non-high frequency element count
{
    if (!A_bf16 || !zipserv::valid_matrix(M, K) ||
        tile_M != 8 || tile_K != 8 || tile_M_median != 16 || tile_K_median != 64 ||
        tile_M_global != 64 || tile_K_global != 64 || !zipserv::valid_exponents(top_exponents) ||
        !sign_mantissa || !compressed_full || !bitmap1 || !bitmap2 || !bitmap3 ||
        !TileOffsets || !TileOffsets_median || !TileOffsets_global)
        return -1;
    try {
        auto packed = zipserv::pack_bf16(A_bf16, M, K, top_exponents[0], true);
        *sign_mantissa = packed.common.release();
        *compressed_full = reinterpret_cast<__nv_bfloat16*>(packed.rare.release());
        *bitmap1 = packed.b1.release();
        *bitmap2 = packed.b2.release();
        *bitmap3 = packed.b3.release();
        *TileOffsets = packed.small.release();
        *TileOffsets_median = packed.medium.release();
        *TileOffsets_global = packed.global.release();
        max_high_freq_count = packed.max_common;
        max_full_count = packed.max_rare;
        return packed.tiles;
    } catch (const std::exception&) {
        return -1;
    }
}
