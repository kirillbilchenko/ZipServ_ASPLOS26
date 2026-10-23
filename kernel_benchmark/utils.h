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
#include <iostream>
#include <vector>
#include <tuple>
#include <set>
#include <iomanip>
#include <algorithm> 
#include <fstream>
#include <random>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <map>
#include <string>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "../csrc/CodecUtils.h"
#include "../build/L_API.cuh"
// Performance Benchmark
#define WARM_UP_ITERATION 100
#define BENCHMARK_ITERATION 2000


void SavePerformanceData(const char* filename, 
                        const char* model_name, const char* layer_name,
                        int M, int K, int N, int SplitK, 
                        float duration_cublas_tc, float tflops_cublas_tc,
                        float duration_CompGEMM, float tflops_CompGEMM,
                        float duration_notc, float tflops_notc) 
{
    FILE* fp;
    // Check if file exists
    fp = fopen(filename, "r");
    bool fileExists = (fp != NULL);
    if (fp) fclose(fp);
    
    // Open file in append mode
    fp = fopen(filename, "a");
    if (!fp) {
        printf("Error opening file for writing!\n");
        return;
    }

    // Write header if new file
    if (!fileExists) {
        fprintf(fp, "Model,Layer,M,K,N,SplitK,Kernel,Duration(ms),TFLOPS\n");
    }

    // Write data for three kernels
    const char* kernels[] = {"cuBLAS", "cuBLAS_TC", "CompGEMM"};
    const float durations[] = {duration_notc, duration_cublas_tc, duration_CompGEMM};
    const float tflops[] = {tflops_notc, tflops_cublas_tc, tflops_CompGEMM};

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%s,%s,%d,%d,%d,%d,%s,%.6f,%.6f\n", 
                model_name, layer_name,
                M, K, N, SplitK, 
                kernels[i], durations[i], tflops[i]);
    }

    fclose(fp);
}
void checkCublasError(cublasStatus_t status, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Cublas Error at line %d, Error Code: %d\n", line, status);
        exit(EXIT_FAILURE);
    }
}

void checkLastCudaError(int line)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Last Cuda Error Detected at line: %d, Error: %s.\n", line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void PrintPerformance(const char* KernelName, float milliseconds, float tflops, double error)
{
    printf("%-10s \t -> \t\t Time/ms: %5.3f \t Performance/TFLOPs: %4.2f \t TotalError: %.2lf\n",
           KernelName,
           milliseconds,
           tflops,
           error);
}


void init_host_matrices_bf16(__nv_bfloat16* A_h, __nv_bfloat16* B_h, int M, int K, int N,
                           const int* custom_exponents = nullptr, unsigned seed = 12345) {
    // Default high-frequency exponent values
    // int default_exponents[7] = {123, 124, 125, 126, 127, 128, 129};
    // int default_exponents[7] = {123, 124, 125, 126, 127, 128, 130};
    int default_exponents[7] = {116, 117, 118, 119, 121, 120, 122};


    
    // Use default values if no custom exponents provided
    const int* target_exponents = custom_exponents ? custom_exponents : default_exponents;
    
    // Set decreasing probability distribution for exponents
    // Weights are {7, 6, 5, 4, 3, 2, 1}, sum = 28
    double weights[7] = {8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double total_weight = 30.0;  // 7+6+5+4+3+2+1
    
    // Initialize random number generator with fixed seed for reproducibility
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist_mantissa(0.0f, 1.0f);
    std::uniform_int_distribution<int> dist_sign(0, 1);
    std::uniform_real_distribution<double> dist_weighted(0.0, total_weight);
    
    // Exponent distribution probability: 97% use target exponents, 3% use random exponents
    std::uniform_real_distribution<float> dist_exp_choice(0.0f, 1.0f);
    std::uniform_int_distribution<int> dist_random_exp(110, 121);  // General BF16 exponent range
    
    // Initialize A - generate matrix with specific exponent distribution
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            uint8_t sign = dist_sign(gen);
            // uint8_t sign = 1;
            uint8_t mantissa = (uint8_t)(dist_mantissa(gen) * 127);
            uint8_t exponent;
            
            // 97% probability to use high-frequency exponents
            if (dist_exp_choice(gen) < 0.95f) {
                // Use weighted random selection
                double rand_val = dist_weighted(gen);
                double cumulative = 0.0;
                int idx = 0;
                
                // Find index corresponding to weight interval
                for (int w = 0; w < 7; w++) {
                    cumulative += weights[w];
                    if (rand_val < cumulative) {
                        idx = w;
                        break;
                    }
                }
                
                exponent = target_exponents[idx];
            } else {
                exponent = dist_random_exp(gen);
            }
            
            // Assemble into BF16 value
            uint16_t bf16_bits = ((sign & 0x1) << 15) | ((exponent & 0xFF) << 7) | (mantissa & 0x7F);
            A_h[i * K + j] = __ushort_as_bfloat16(bf16_bits);
        }
    }
    
    // Initialize B - use fixed value
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            // B_h[j * K + i] = __float2bfloat16(1.0);
            B_h[j * K + i] = __float2bfloat16(dis(gen));
        }
    }
}


// Calculate total error of BF16 matrix
double ComputeTotalError_BF16(const __nv_bfloat16* A, const __nv_bfloat16* B, int M, int N) {
    double total_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float a_val = __bfloat162float(A[i]);
        float b_val = __bfloat162float(B[i]);
        total_error += std::abs(a_val - b_val);
    }
    return total_error;
}



// Analyze exponent distribution of BF16 matrix
void analyzeExponentDistribution_BF16(__nv_bfloat16* matrix, int M, int K, int* top_exponents, int top_n = 7) {
    if (!matrix || !top_exponents || !zipserv::valid_matrix(M, K))
        throw std::invalid_argument("BF16 matrix must have positive 64-aligned dimensions and int32 offsets");
    uint64_t counts[256] = {};
    for (int i = 0; i < M * K; ++i)
        ++counts[(__bfloat16_as_ushort(matrix[i]) >> 7) & 255];
    const int base = zipserv::best_exponent_window(counts, top_n);
    for (int i = 0; i < top_n; ++i) top_exponents[i] = base + i;
}



int CompressBF16MatrixTripleBitmap_Host(
    __nv_bfloat16* A_bf16,
    int M, int K,
    int tile_M, int tile_M_median, int tile_M_global,
    int tile_K, int tile_K_median, int tile_K_global,
    const int* top_exponents,
    __nv_bfloat16** compressed_full,
    uint8_t** sign_mantissa,
    uint64_t** bitmap1, uint64_t** bitmap2, uint64_t** bitmap3,
    int** TileOffsets, int** TileOffsets_median, int** TileOffsets_global,
    int& max_high_freq_count, int& max_full_count)
{
    return InitBF16MatrixTripleBitmap(
        A_bf16, M, K, tile_M, tile_M_median, tile_M_global,
        tile_K, tile_K_median, tile_K_global, top_exponents,
        sign_mantissa, compressed_full, bitmap1, bitmap2, bitmap3,
        TileOffsets, TileOffsets_median, TileOffsets_global,
        max_high_freq_count, max_full_count);
}
// CPU version of BF16 initialization function
int InitBF16MatrixTripleBitmap_Host(
    __nv_bfloat16* A_bf16,
    int M, int K,
    int tile_M, int tile_M_median, int tile_M_global,
    int tile_K, int tile_K_median, int tile_K_global,
    __nv_bfloat16** top_exponents, 
    __nv_bfloat16** compressed_full,
    uint8_t** sign_mantissa,
    uint64_t** bitmap1, uint64_t** bitmap2, uint64_t** bitmap3,
    int** TileOffsets, int** TileOffsets_median, int** TileOffsets_global,
    int& max_high_freq_count, int& max_full_count, uint8_t& start_exp)
{
    // Analyze exponent distribution of BF16 matrix
    int top_exponent_values[7] = {0};
    analyzeExponentDistribution_BF16(A_bf16, M, K, top_exponent_values);
    start_exp = top_exponent_values[0] - 1;
    // Allocate top_exponents memory and convert to BF16
    *top_exponents = (__nv_bfloat16*)malloc(7 * sizeof(__nv_bfloat16));
    if (*top_exponents == nullptr) return -1;
    
    for (int i = 0; i < 7; i++) {
        // Create BF16 value with exponent part only
        uint16_t exp_bits = (top_exponent_values[i] & 0xFF) << 7;
        (*top_exponents)[i] = __ushort_as_bfloat16(exp_bits);
    }
    
    // Call compression function (needs reimplementation for direct BF16 operations)
    return CompressBF16MatrixTripleBitmap_Host(
        A_bf16, M, K, 
        tile_M, tile_M_median, tile_M_global,
        tile_K, tile_K_median, tile_K_global,
        top_exponent_values, compressed_full, sign_mantissa,
        bitmap1, bitmap2, bitmap3,
        TileOffsets, TileOffsets_median, TileOffsets_global,
        max_high_freq_count, max_full_count
    );
}



