// Modified 2026 by Kyrylo Bilchenko: correctness and MxWave backports.
#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <random>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include "L_API.cuh"
#include "./utils.h"

// Host-side popcount function to replace __popcll
inline int host_popcll(uint64_t val) {
    int count = 0;
    while (val) {
        count += val & 1;
        val >>= 1;
    }
    return count;
}

// Function to flush L2 cache using dynamic cache size detection
void flush_l2_cache() {
    static void* d_flush_buffer = nullptr;
    static size_t flush_buffer_size = 0;
    static bool initialized = false;
    
    if (!initialized) {
        // Get L2 cache size dynamically
        int device = 0;
        int l2_size = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device);
        
        // Use 2x L2 cache size to ensure complete flush
        flush_buffer_size = l2_size * 2;
        
        // Allocate flush buffer
        cudaMalloc(&d_flush_buffer, flush_buffer_size);
        if (d_flush_buffer == nullptr) {
            printf("Error: Failed to allocate L2 cache flush buffer\n");
            exit(-1);
        }
        
        printf("L2 Cache size: %d bytes, Flush buffer size: %zu bytes\n", l2_size, flush_buffer_size);
        initialized = true;
    }
    
    // Flush L2 cache by writing to buffer larger than L2 cache
    cudaMemsetAsync(d_flush_buffer, 0, flush_buffer_size);
    cudaDeviceSynchronize();
}

// Initialization helper that only populates matrix A
void init_host_matrices_bf16_A_only(__nv_bfloat16* A_h, int M, int K,
                                   const int* custom_exponents = nullptr, unsigned seed = 12345) {
    // Default high-frequency exponent values
    int default_exponents[7] = {116, 117, 118, 119, 121, 120, 122};
    
    // Use defaults if no custom exponents provided
    const int* target_exponents = custom_exponents ? custom_exponents : default_exponents;
    
    // Set decreasing probability distribution for exponent selection
    // Weights {7, 6, 5, 4, 3, 2, 1} sum to 28
    double weights[7] = {8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double total_weight = 30.0;  // 7+6+5+4+3+2+1
    
    // Initialize random number generator with fixed seed for reproducibility
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist_mantissa(0.0f, 1.0f);
    std::uniform_int_distribution<int> dist_sign(0, 1);
    std::uniform_real_distribution<double> dist_weighted(0.0, total_weight);
    
    // Exponent distribution: 95% target exponents, 5% random
    std::uniform_real_distribution<float> dist_exp_choice(0.0f, 1.0f);
    std::uniform_int_distribution<int> dist_random_exp(110, 121);  // Typical BF16 exponent range
    
    // Initialize A - generate matrix with specific exponent distribution
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            uint8_t sign = dist_sign(gen);
            uint8_t mantissa = (uint8_t)(dist_mantissa(gen) * 127);
            uint8_t exponent;
            
            // 95% probability to use high-frequency exponents
            if (dist_exp_choice(gen) < 0.95f) {
                // Use weighted random selection
                double rand_val = dist_weighted(gen);
                double cumulative = 0.0;
                int idx = 0;
                
                // Find the index corresponding to the weight interval
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
}

// Function to print BF16 matrix - for debugging only
void print_bf16_matrix(const char* name, __nv_bfloat16* matrix, int rows, int cols, int max_rows = 8, int max_cols = 16) {
    printf("\n===== %s [%dx%d] =====\n", name, rows, cols);
    
    int display_rows = std::min(rows, max_rows);
    int display_cols = std::min(cols, max_cols);
    
    for (int i = 0; i < display_rows; i++) {
        for (int j = 0; j < display_cols; j++) {
            printf("%8.4f ", __bfloat162float(matrix[i * cols + j]));
        }
        if (cols > max_cols) printf("...");
        printf("\n");
    }
    if (rows > max_rows) printf("...\n");
    printf("\n");
}

// Analyze exponent distribution in the matrix
void analyze_exponent_distribution(__nv_bfloat16* matrix, int size) {
    printf("\n========== Exponent Distribution Analysis ==========\n");
    
    // Count frequency of each exponent
    std::map<int, int> exp_count;
    int total_elements = 0;
    
    for (int i = 0; i < size; i++) {
        uint16_t bits = __bfloat16_as_ushort(matrix[i]);
        int exponent = (bits >> 7) & 0xFF;
        exp_count[exponent]++;
        total_elements++;
    }
    
    // Print high-frequency exponent distribution
    printf("High-frequency exponent (123-129) distribution:\n");
    printf("Exponent\tCount\t\tPercentage\n");
    int high_freq_total = 0;
    for (int exp = 123; exp <= 129; exp++) {
        int count = exp_count[exp];
        high_freq_total += count;
        printf("%d\t%d\t\t%.2f%%\n", exp, count, 100.0 * count / total_elements);
    }
    
    printf("\nHigh-frequency exponent total: %d (%.2f%%)\n", high_freq_total, 100.0 * high_freq_total / total_elements);
    printf("Other exponent total: %d (%.2f%%)\n", total_elements - high_freq_total, 
           100.0 * (total_elements - high_freq_total) / total_elements);
    
    // Display distribution of other exponents
    printf("\nOther exponent distribution (count > 0):\n");
    for (auto& pair : exp_count) {
        if (pair.first < 123 || pair.first > 129) {
            if (pair.second > 0) {
                printf("Exponent %d: %d times (%.2f%%)\n", 
                       pair.first, pair.second, 100.0 * pair.second / total_elements);
            }
        }
    }
}

// Compare two BF16 matrices
void compare_bf16_matrices(const char* name1, const char* name2, 
                          __nv_bfloat16* matrix1, __nv_bfloat16* matrix2, 
                          int rows, int cols) {
    printf("\n========== Comparing %s vs %s ==========\n", name1, name2);
    
    int total_elements = rows * cols;
    int identical_count = 0;
    int different_count = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    float total_abs_diff = 0.0f;
    
    // Display the first few differing elements in detail
    int diff_shown = 0;
    const int max_diff_show = 10;
    
    printf("Detailed differences (first %d elements):\n", max_diff_show);
    printf("Position\t%s\t\t%s\t\tAbsolute Diff\tRelative Diff\n", name1, name2);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            float val1 = __bfloat162float(matrix1[idx]);
            float val2 = __bfloat162float(matrix2[idx]);
            
            float abs_diff = std::abs(val1 - val2);
            float rel_diff = (val1 != 0.0f) ? abs_diff / std::abs(val1) : abs_diff;
            
            total_abs_diff += abs_diff;
            max_abs_diff = std::max(max_abs_diff, abs_diff);
            max_rel_diff = std::max(max_rel_diff, rel_diff);
            
            if (__bfloat16_as_ushort(matrix1[idx]) == __bfloat16_as_ushort(matrix2[idx])) {
                identical_count++;
            } else {
                different_count++;
                if (diff_shown < max_diff_show) {
                    uint16_t bits1 = __bfloat16_as_ushort(matrix1[idx]);
                    uint16_t bits2 = __bfloat16_as_ushort(matrix2[idx]);
                    printf("[%d,%d]\t\t%8.4f\t\t%8.4f\t\t%8.4f\t%8.4f (0x%04X vs 0x%04X)\n", 
                           i, j, val1, val2, abs_diff, rel_diff, bits1, bits2);
                    diff_shown++;
                }
            }
        }
    }
    
    float avg_abs_diff = total_abs_diff / total_elements;
    
    printf("\nStatistical results:\n");
    printf("  Total elements: %d\n", total_elements);
    printf("  Identical: %d (%.2f%%)\n", identical_count, 100.0f * identical_count / total_elements);
    printf("  Different: %d (%.2f%%)\n", different_count, 100.0f * different_count / total_elements);
    printf("  Max absolute difference: %8.6f\n", max_abs_diff);
    printf("  Max relative difference: %8.6f\n", max_rel_diff);
    printf("  Average absolute difference: %8.6f\n", avg_abs_diff);
    printf("  Total absolute error: %8.6f\n", total_abs_diff);
    
    // Determine test result
    if (identical_count == total_elements) {
        printf("  ✓ Perfect match! All elements are identical\n");
    } else if (max_abs_diff < 1e-5) {
        printf("  FAIL: bits differ (max numeric difference < 1e-5)\n");
    } else if (max_abs_diff < 1e-3) {
        printf("  FAIL: bits differ (max numeric difference < 1e-3)\n");
    } else {
        printf("  ✗ Large differences, potential issues\n");
    }
}

// Print compression statistics
void print_compression_stats(int M, int K, int high_freq_count, int full_count, 
                           int max_high_freq_count, int max_full_count, int num_global_tiles, uint8_t start_exp) {
    printf("\n========== Compression Statistics ==========\n");
    printf("Matrix dimensions: %dx%d\n", M, K);
    printf("Global tile count: %d\n", num_global_tiles);
    printf("High-frequency elements: %d (%.2f%%)\n", high_freq_count, 100.0f * high_freq_count / (M * K));
    printf("Non-high-frequency elements: %d (%.2f%%)\n", full_count, 100.0f * full_count / (M * K));
    printf("Max high-frequency per tile: %d\n", max_high_freq_count);
    printf("Max non-high-frequency per tile: %d\n", max_full_count);
    printf("High-frequency starting value: %u\n", start_exp);
    // Calculate compression ratio
    size_t original_size = M * K * sizeof(__nv_bfloat16);
    size_t compressed_size = 
        // (7 * sizeof(__nv_bfloat16)) +                    // High-freq exponents
        (high_freq_count * sizeof(uint8_t)) +            // High-freq elements
        (full_count * sizeof(__nv_bfloat16)) +           // Non-high-freq elements
        (num_global_tiles * 64 * sizeof(uint64_t) * 3) +
        (num_global_tiles * 4 * 2 * sizeof(int)) +       // Median offsets
        ((num_global_tiles + 1) * 2 * sizeof(int));      // Global offsets
    
    float compression_ratio = (float)original_size / compressed_size;
    printf("Compression ratio: %.2fx\n", compression_ratio);
}

void init_host_matrices_bf16_B_only(__nv_bfloat16* B, int K, int N, unsigned seed = 42)
{
    printf("Initializing matrix B (%dx%d)...\n", K, N);
    
    // Set random seed
    srand(seed);
    
    // Initialize matrix B (K x N, column-major storage)
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < K; i++) {
            // Generate random BF16 value
            float random_val = (float)rand() / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
            random_val *= 10.0f; // Scale to [-10, 10]
            B[i + j * K] = __float2bfloat16(random_val);
        }
    }
    
    printf("Matrix B initialization complete\n");
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        printf("Usage: ./decompress_test M K N\n");
        printf("Example: ./decompress_test 128 128 128\n");
        return -1;
    }
    
    int M_GLOBAL = atoi(argv[1]);
    int K_GLOBAL = atoi(argv[2]);
    int N_GLOBAL = atoi(argv[3]);
    // int N_GLOBAL = K_GLOBAL; // For matrix multiplication tests, set N equal to K
    
    // Verify matrix dimensions must be multiples of 64
    if (M_GLOBAL % 64 != 0 || K_GLOBAL % 64 != 0) {
        printf("Error: Matrix dimensions must be multiples of 64. Current: M=%d, K=%d\n", M_GLOBAL, K_GLOBAL);
        return -1;
    }
    
    printf("====== BF16 Triple Bitmap Decompression Correctness Test + Matrix Multiplication Performance Test ======\n");
    printf("Matrix dimensions: %dx%d, Matrix multiplication: %dx%dx%d\n", M_GLOBAL, K_GLOBAL, M_GLOBAL, K_GLOBAL, N_GLOBAL);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate host memory
    printf("\nAllocating host memory...\n");
    __nv_bfloat16* A_original = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * M_GLOBAL * K_GLOBAL);
    __nv_bfloat16* A_decompressed = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * M_GLOBAL * K_GLOBAL);
    __nv_bfloat16* B_h = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * K_GLOBAL * N_GLOBAL); // For matrix multiplication
    
    if (A_original == NULL || A_decompressed == NULL || B_h == NULL) {
        printf("Error: Host memory allocation failed!\n");
        exit(-1);
    }
    
    // Initialize test matrices using your helper
    printf("Initializing test matrices...\n");
    
    // Try different seed values if needed
    unsigned test_seed = 42;
    printf("Using random seed: %u\n", test_seed);
    
    // Initialize matrix A for decompression
    init_host_matrices_bf16_A_only(A_original, M_GLOBAL, K_GLOBAL, nullptr, test_seed);
    
    // Initialize matrix B for matrix multiplication
    init_host_matrices_bf16_B_only(B_h, K_GLOBAL, N_GLOBAL, test_seed);
    
    printf("Original matrix sample:\n");
    print_bf16_matrix("Original Matrix A", A_original, M_GLOBAL, K_GLOBAL, 64, 64);
    
    // Analyze exponent distribution
    analyze_exponent_distribution(A_original, M_GLOBAL * K_GLOBAL);
    
    // Allocate device memory
    printf("\nAllocating device memory...\n");
    __nv_bfloat16* A_gpu = NULL;
    __nv_bfloat16* A_decompressed_gpu = NULL;
    __nv_bfloat16* B_gpu = NULL;
    
    cudaMalloc(&A_gpu, sizeof(__nv_bfloat16) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(&A_decompressed_gpu, sizeof(__nv_bfloat16) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(&B_gpu, sizeof(__nv_bfloat16) * K_GLOBAL * N_GLOBAL);
    checkLastCudaError(__LINE__);
    
    // Copy original data to GPU
    cudaMemcpy(A_gpu, A_original, sizeof(__nv_bfloat16) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B_h, sizeof(__nv_bfloat16) * K_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);
    
    // ================== Compression Phase ==================
    printf("\n========== Compression Phase ==========\n");
    
    __nv_bfloat16* top_exponents_cpu = nullptr;
    __nv_bfloat16* compressed_full_cpu = nullptr;
    uint8_t* sign_mantissa_cpu = nullptr;
    uint64_t* bitmap1_cpu = nullptr;
    uint64_t* bitmap2_cpu = nullptr;
    uint64_t* bitmap3_cpu = nullptr;
    int* TileOffsets_cpu = nullptr;
    int* TileOffsets_median_cpu = nullptr;
    int* TileOffsets_global_cpu = nullptr;
    int max_high_freq_count = 0;
    int max_full_count = 0;
    uint8_t start_exp;
    
    // Call compression function
    int num_global_tiles = InitBF16MatrixTripleBitmap_Host(
        A_original, M_GLOBAL, K_GLOBAL, 
        8, 16, 64, 8, 64, 64,  // tile configuration
        // 8, 16, 128, 8, 64, 64,  // tile configuration
        // 8, 16, 256, 8, 64, 64,  // tile configuration

        &top_exponents_cpu, &compressed_full_cpu, &sign_mantissa_cpu,
        &bitmap1_cpu, &bitmap2_cpu, &bitmap3_cpu,
        &TileOffsets_cpu, &TileOffsets_median_cpu, &TileOffsets_global_cpu,
        max_high_freq_count, max_full_count, start_exp);
    
    if (num_global_tiles <= 0) {
        printf("Error: BF16 triple bitmap compression failed\n");
        exit(-1);
    }
    
    // Calculate compressed data size
    int high_freq_count = TileOffsets_global_cpu[num_global_tiles * 2];
    int full_count = TileOffsets_global_cpu[num_global_tiles * 2 + 1];
    
    print_compression_stats(M_GLOBAL, K_GLOBAL, high_freq_count, full_count, 
                           max_high_freq_count, max_full_count, num_global_tiles, start_exp);
    
    // Print high-frequency exponents after compression
    printf("\nHigh-frequency exponents after compression:\n");
    for (int i = 0; i < 7; i++) {
        uint16_t bits = __bfloat16_as_ushort(top_exponents_cpu[i]);
        int exponent = (bits >> 7) & 0xFF;
        printf("  Exponent[%d] = %d\n", i, exponent);
    }
    
    // ================== Prepare GPU Compressed Data ==================
    printf("\nPreparing GPU compressed data...\n");
    
    // Calculate tile counts
    int num_tiles = (M_GLOBAL / 8) * (K_GLOBAL / 8);
    int num_median_tiles = (M_GLOBAL / 16) * (K_GLOBAL / 64);


    
    // Allocate GPU memory for compressed data
    __nv_bfloat16* top_exponents_gpu = nullptr;
    __nv_bfloat16* compressed_full_gpu = nullptr;
    uint8_t* sign_mantissa_gpu = nullptr;
    uint64_t* bitmap1_gpu = nullptr;
    uint64_t* bitmap2_gpu = nullptr;
    uint64_t* bitmap3_gpu = nullptr;
    int* TileOffsets_gpu = nullptr;
    int* TileOffsets_median_gpu = nullptr;
    int* TileOffsets_global_gpu = nullptr;
    
    cudaMalloc(&top_exponents_gpu, 7 * sizeof(__nv_bfloat16));
    cudaMalloc(&compressed_full_gpu, full_count * sizeof(__nv_bfloat16));
    cudaMalloc(&sign_mantissa_gpu, high_freq_count * sizeof(uint8_t));
    cudaMalloc(&bitmap1_gpu, num_tiles * sizeof(uint64_t));
    cudaMalloc(&bitmap2_gpu, num_tiles * sizeof(uint64_t));
    cudaMalloc(&bitmap3_gpu, num_tiles * sizeof(uint64_t));
    cudaMalloc(&TileOffsets_gpu, num_tiles * 2 * sizeof(int));
    cudaMalloc(&TileOffsets_median_gpu, num_median_tiles * 2 * sizeof(int));
    cudaMalloc(&TileOffsets_global_gpu, (num_global_tiles + 1) * 2 * sizeof(int));
    checkLastCudaError(__LINE__);
    
    // Copy compressed data to GPU
    cudaMemcpy(top_exponents_gpu, top_exponents_cpu, 7 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(compressed_full_gpu, compressed_full_cpu, full_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(sign_mantissa_gpu, sign_mantissa_cpu, high_freq_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap1_gpu, bitmap1_cpu, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap2_gpu, bitmap2_cpu, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap3_gpu, bitmap3_cpu, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(TileOffsets_gpu, TileOffsets_cpu, num_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(TileOffsets_median_gpu, TileOffsets_median_cpu, num_median_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(TileOffsets_global_gpu, TileOffsets_global_cpu, (num_global_tiles + 1) * 2 * sizeof(int), cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);
    
    // ================== Decompression Phase ==================
    printf("\n========== Decompression Phase ==========\n");
    
    // Clear decompression output matrix
    cudaMemset(A_decompressed_gpu, 0, sizeof(__nv_bfloat16) * M_GLOBAL * K_GLOBAL);
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cudaError_t decompress_error = BF16TripleBitmap_Decompress_API(
            0,
            sign_mantissa_gpu,
            compressed_full_gpu,
            bitmap1_gpu,
            bitmap2_gpu,
            bitmap3_gpu,
            TileOffsets_median_gpu,
            TileOffsets_global_gpu,
            max_high_freq_count,
            max_full_count,
            start_exp,
            A_decompressed_gpu,
            M_GLOBAL,
            K_GLOBAL);
    }
    
    // Flush L2 cache before decompression benchmark
    flush_l2_cache();
    
    cudaError_t decompress_error;    
    // Measure decompression time
    printf("Measuring decompression time...\n");
    float total_milliseconds_decompress = 0.0f;
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        // Flush L2 cache before each iteration to simulate real-world cold cache scenario
        flush_l2_cache();
        
        // Measure only the decompression operation time, excluding cache flush overhead
        cudaEventRecord(start);
        decompress_error = BF16TripleBitmap_Decompress_API(
            0,
            sign_mantissa_gpu,
            compressed_full_gpu,
            bitmap1_gpu,
            bitmap2_gpu,
            bitmap3_gpu,
            TileOffsets_median_gpu,
            TileOffsets_global_gpu,
            max_high_freq_count,
            max_full_count,
            start_exp,
            A_decompressed_gpu,
            M_GLOBAL,
            K_GLOBAL);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float iter_time = 0.0f;
        cudaEventElapsedTime(&iter_time, start, stop);
        total_milliseconds_decompress += iter_time;
    }
    
    float milliseconds_decompress = total_milliseconds_decompress / BENCHMARK_ITERATION;
    if (decompress_error != cudaSuccess) {
            printf("Error: Decompression failed - %s\n", cudaGetErrorString(decompress_error));
            exit(-1);
    }
    printf("Decompression complete!\n");
    printf("Average decompression time: %.4f ms\n", milliseconds_decompress);
    
    // ================== CuBLAS Matrix Multiplication Test ==================
    printf("\n========== CuBLAS Matrix Multiplication Performance Test ==========\n");
    
    // Allocate output matrix
    __nv_bfloat16* C_cublas_no_tc = NULL;
    __nv_bfloat16* C_cublas_tc = NULL;
    cudaMalloc(&C_cublas_no_tc, sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL);
    cudaMalloc(&C_cublas_tc, sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL);
    checkLastCudaError(__LINE__);
    
    // Create CuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, 0);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int m = M_GLOBAL, n = N_GLOBAL, k = K_GLOBAL;
    cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0);
    
    // Test CuBLAS without Tensor Core
    printf("Testing CuBLAS without Tensor Core...\n");
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    
    // Warmup
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublasGemmEx(handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     m, n, k,
                     &alpha,
                     A_decompressed_gpu, CUDA_R_16BF, k,
                     B_gpu, CUDA_R_16BF, k,
                     &beta,
                     C_cublas_no_tc, CUDA_R_16BF, m,
                     CUDA_R_32F,
                     CuBlasALG);
    }
    
    // Flush L2 cache before CuBlas benchmark (without Tensor Core)
    flush_l2_cache();
    
    // Benchmark
    float total_milliseconds_cublas_no_tc = 0.0f;
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        // Flush L2 cache before each iteration to simulate real-world cold cache scenario
        flush_l2_cache();
        
        // Measure only the GEMM operation time, excluding cache flush overhead
        cudaEventRecord(start);
        cublasGemmEx(handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     m, n, k,
                     &alpha,
                     A_decompressed_gpu, CUDA_R_16BF, k,
                     B_gpu, CUDA_R_16BF, k,
                     &beta,
                     C_cublas_no_tc, CUDA_R_16BF, m,
                     CUDA_R_32F,
                     CuBlasALG);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float iter_time = 0.0f;
        cudaEventElapsedTime(&iter_time, start, stop);
        total_milliseconds_cublas_no_tc += iter_time;
    }
    
    float milliseconds_cublas_no_tc = total_milliseconds_cublas_no_tc / BENCHMARK_ITERATION;
    float tflops_cublas_no_tc = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) 
                                                   / (milliseconds_cublas_no_tc / 1000.)) / 1e12;
    
    // Test CuBLAS with Tensor Core
    printf("Testing CuBLAS with Tensor Core...\n");
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    
    // Warmup
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublasGemmEx(handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     m, n, k,
                     &alpha,
                     A_decompressed_gpu, CUDA_R_16BF, k,
                     B_gpu, CUDA_R_16BF, k,
                     &beta,
                     C_cublas_tc, CUDA_R_16BF, m,
                     CUDA_R_32F,
                     CuBlasALG);
    }
    
    // Flush L2 cache before CuBlas benchmark (with Tensor Core)
    flush_l2_cache();
    
    // Benchmark
    float total_milliseconds_cublas_tc = 0.0f;
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        // Flush L2 cache before each iteration to simulate real-world cold cache scenario
        flush_l2_cache();
        
        // Measure only the GEMM operation time, excluding cache flush overhead
        cudaEventRecord(start);
        cublasGemmEx(handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     m, n, k,
                     &alpha,
                     A_decompressed_gpu, CUDA_R_16BF, k,
                     B_gpu, CUDA_R_16BF, k,
                     &beta,
                     C_cublas_tc, CUDA_R_16BF, m,
                     CUDA_R_32F,
                     CuBlasALG);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float iter_time = 0.0f;
        cudaEventElapsedTime(&iter_time, start, stop);
        total_milliseconds_cublas_tc += iter_time;
    }
    
    float milliseconds_cublas_tc = total_milliseconds_cublas_tc / BENCHMARK_ITERATION;
    float tflops_cublas_tc = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) 
                                                / (milliseconds_cublas_tc / 1000.)) / 1e12;
    
    // ================== Verification Phase ==================
    printf("\n========== Verification Phase ==========\n");
    
    // Copy decompression result back to host
    cudaMemcpy(A_decompressed, A_decompressed_gpu, sizeof(__nv_bfloat16) * M_GLOBAL * K_GLOBAL, cudaMemcpyDeviceToHost);
    checkLastCudaError(__LINE__);
    
    printf("Decompressed matrix sample:\n");
    print_bf16_matrix("Decompressed Matrix A", A_decompressed, M_GLOBAL, K_GLOBAL, 8, 16);
    
    // Analyze exponent distribution after decompression
    analyze_exponent_distribution(A_decompressed, M_GLOBAL * K_GLOBAL);
    
    // Compare original and decompressed matrices
    compare_bf16_matrices("Original Matrix", "Decompressed Matrix", A_original, A_decompressed, M_GLOBAL, K_GLOBAL);
    
    // Copy CuBLAS results and compare
    __nv_bfloat16* C_cublas_no_tc_h = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL);
    __nv_bfloat16* C_cublas_tc_h = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL);
    
    cudaMemcpy(C_cublas_no_tc_h, C_cublas_no_tc, sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_cublas_tc_h, C_cublas_tc, sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);
    
    // Compare CuBLAS TC and non-TC results
    printf("\n========== CuBLAS Result Comparison ==========\n");
    double totalError_cublas = ComputeTotalError_BF16(C_cublas_no_tc_h, C_cublas_tc_h, M_GLOBAL, N_GLOBAL);
    printf("CuBLAS TC vs non-TC total error: %g\n", totalError_cublas);
    
    // ================== Performance Results Summary ==================
    printf("\n========== Performance Results Summary ==========\n");
    printf("Decompression time: %.4f ms\n", milliseconds_decompress);
    printf("CuBLAS non-TC: %.4f ms, %.2f TFLOPS\n", milliseconds_cublas_no_tc, tflops_cublas_no_tc);
    printf("CuBLAS TC:   %.4f ms, %.2f TFLOPS\n", milliseconds_cublas_tc, tflops_cublas_tc);
    
    // Calculate time comparisons
    float total_decompress_plus_matmul_no_tc = milliseconds_decompress + milliseconds_cublas_no_tc;
    float total_decompress_plus_matmul_tc = milliseconds_decompress + milliseconds_cublas_tc;
    
    printf("\nDecompression + Matrix Multiplication Total Time:\n");
    printf("Decompression + CuBLAS non-TC: %.4f ms\n", total_decompress_plus_matmul_no_tc);
    printf("Decompression + CuBLAS TC:  %.4f ms\n", total_decompress_plus_matmul_tc);
    
    printf("\nTime Distribution Analysis:\n");
    printf("Decompression ratio (vs non-TC): %.1f%% (Decomp: %.4f ms, MatMul: %.4f ms)\n", 
           100.0f * milliseconds_decompress / total_decompress_plus_matmul_no_tc,
           milliseconds_decompress, milliseconds_cublas_no_tc);
    printf("Decompression ratio (vs TC):   %.1f%% (Decomp: %.4f ms, MatMul: %.4f ms)\n", 
           100.0f * milliseconds_decompress / total_decompress_plus_matmul_tc,
           milliseconds_decompress, milliseconds_cublas_tc);
    
    const bool exact_decompression = std::memcmp(
        A_original, A_decompressed, size_t(M_GLOBAL) * K_GLOBAL * sizeof(__nv_bfloat16)) == 0;

    // ================== Cleanup Resources ==================
    printf("\nCleaning up resources...\n");
    
    // Free host memory
    free(A_original);
    free(A_decompressed);
    free(B_h);
    free(C_cublas_no_tc_h);
    free(C_cublas_tc_h);
    free(top_exponents_cpu);
    free(compressed_full_cpu);
    free(sign_mantissa_cpu);
    free(bitmap1_cpu);
    free(bitmap2_cpu);
    free(bitmap3_cpu);
    free(TileOffsets_cpu);
    free(TileOffsets_median_cpu);
    free(TileOffsets_global_cpu);
    
    // Free GPU memory
    cudaFree(A_gpu);
    cudaFree(A_decompressed_gpu);
    cudaFree(B_gpu);
    cudaFree(C_cublas_no_tc);
    cudaFree(C_cublas_tc);
    cudaFree(top_exponents_gpu);
    cudaFree(compressed_full_gpu);
    cudaFree(sign_mantissa_gpu);
    cudaFree(bitmap1_gpu);
    cudaFree(bitmap2_gpu);
    cudaFree(bitmap3_gpu);
    cudaFree(TileOffsets_gpu);
    cudaFree(TileOffsets_median_gpu);
    cudaFree(TileOffsets_global_gpu);
    
    cudaStreamDestroy(stream);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n========== Test Complete ==========\n");
    printf("Bit-exact decompression: %s\n", exact_decompression ? "PASS" : "FAIL");
    printf("Performance test: Comparison analysis of decompression and matrix multiplication time completed\n");
    
    return exact_decompression ? 0 : 1;
}




