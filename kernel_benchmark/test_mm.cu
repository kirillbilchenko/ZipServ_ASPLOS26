// Modified 2026 by Kyrylo Bilchenko: correctness and MxWave backports.
#include <assert.h>
#include <cublas_v2.h>
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

// Function to print BF16 matrix - for debugging only
void print_bf16_matrix(const char* name, __nv_bfloat16* matrix, int rows, int cols, int max_rows = 128, int max_cols = 32) {
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
    
    // Print some BF16 internal details
    printf("BF16 details example (first 4 elements):\n");
    for (int i = 0; i < std::min(4, rows * cols); i++) {
        uint16_t bits = __bfloat16_as_ushort(matrix[i]);
        uint8_t sign = (bits >> 15) & 0x1;
        uint8_t exponent = (bits >> 7) & 0xFF;
        uint8_t mantissa = bits & 0x7F;
        
        printf("Element[%d] = %8.4f (sign=%d, exp=%3d, mantissa=%3d, raw=0x%04X)\n", 
               i, __bfloat162float(matrix[i]), sign, exponent, mantissa, bits);
    }
    printf("\n");
}
void print_bf16_matrix_col(const char* name, __nv_bfloat16* matrix, int rows, int cols, int max_rows = 128, int max_cols = 32) {
    printf("\n===== %s [%dx%d] =====\n", name, rows, cols);
    
    int display_rows = std::min(rows, max_rows);
    int display_cols = std::min(cols, max_cols);
    
    for (int i = 0; i < display_rows; i++) {
        for (int j = 0; j < display_cols; j++) {
            printf("%8.4f ", __bfloat162float(matrix[j * rows + i]));
        }
        if (cols > max_cols) printf("...");
        printf("\n");
    }
    if (rows > max_rows) printf("...\n");
    
    // Print some BF16 internal details
    printf("BF16 details example (first 4 elements):\n");
    for (int i = 0; i < std::min(4, rows * cols); i++) {
        uint16_t bits = __bfloat16_as_ushort(matrix[i]);
        uint8_t sign = (bits >> 15) & 0x1;
        uint8_t exponent = (bits >> 7) & 0xFF;
        uint8_t mantissa = bits & 0x7F;
        
        printf("Element[%d] = %8.4f (sign=%d, exp=%3d, mantissa=%3d, raw=0x%04X)\n", 
               i, __bfloat162float(matrix[i]), sign, exponent, mantissa, bits);
    }
    printf("\n");
}

// Function to print bitmap data
void print_bitmap_data(uint64_t* bitmap1, uint64_t* bitmap2, uint64_t* bitmap3, 
                       int* offsets, int count, int max_tiles = 2) {
    printf("\n===== Bitmap Data Example (first %d small tiles) =====\n", std::min(count, max_tiles));
    
    for (int i = 0; i < std::min(count, max_tiles); i++) {
        printf("Tile[%d]: High-frequency count=%d, Non-high-frequency count=%d\n", i, offsets[i*2], offsets[i*2+1]);
        printf("  bitmap1: 0x%016lX (%lu)\n", bitmap1[i], bitmap1[i]);
        printf("  bitmap2: 0x%016lX (%lu)\n", bitmap2[i], bitmap2[i]);
        printf("  bitmap3: 0x%016lX (%lu)\n", bitmap3[i], bitmap3[i]);
        
        // Count number of 1 bits in bitmap
        int ones_bitmap1 = host_popcll(bitmap1[i]);
        int ones_bitmap2 = host_popcll(bitmap2[i]);
        int ones_bitmap3 = host_popcll(bitmap3[i]);
        
        printf("  1-bit count: bitmap1=%d, bitmap2=%d, bitmap3=%d\n", ones_bitmap1, ones_bitmap2, ones_bitmap3);
        
        // Display bitmap binary representation (first 32 bits)
        printf("  Binary(first 32 bits):\n  bitmap1: ");
        for (int b = 31; b >= 0; b--) {
            printf("%d", (bitmap1[i] >> b) & 1);
            if (b % 8 == 0) printf(" ");
        }
        printf("...\n  bitmap2: ");
        for (int b = 31; b >= 0; b--) {
            printf("%d", (bitmap2[i] >> b) & 1);
            if (b % 8 == 0) printf(" ");
        }
        printf("...\n  bitmap3: ");
        for (int b = 31; b >= 0; b--) {
            printf("%d", (bitmap3[i] >> b) & 1);
            if (b % 8 == 0) printf(" ");
        }
        printf("...\n\n");
    }
}

// Print compressed data example
void print_compressed_data(const __nv_bfloat16* top_exponents, const uint8_t* sign_mantissa, 
                           const __nv_bfloat16* full_values, int high_freq_count, int full_count) {
    printf("\n===== Compressed Data Example =====\n");
    
    // Print high-frequency exponents
    printf("High-frequency exponent values (7 entries):\n");
    for (int i = 0; i < 7; i++) {
        uint16_t bits = __bfloat16_as_ushort(top_exponents[i]);
        uint8_t exponent = (bits >> 7) & 0xFF;
        printf("  Exponent[%d] = %d (0x%04X)\n", i, exponent, bits);
    }
    
    // Print high-frequency element examples
    int high_freq_display = std::min(high_freq_count, 20);
    printf("\nHigh-frequency element examples (first %d):\n", high_freq_display);
    for (int i = 0; i < high_freq_display; i++) {
        uint8_t combined = sign_mantissa[i];
        uint8_t sign = (combined >> 7) & 0x1;
        uint8_t mantissa = combined & 0x7F;
        printf("  Element[%d]: sign=%d, mantissa=0x%02X (%d)\n", i, sign, mantissa, mantissa);
    }
    
    // Print non-high-frequency element examples
    int full_display = std::min(full_count, 20);
    printf("\nNon-high-frequency element examples (first %d):\n", full_display);
    for (int i = 0; i < full_display; i++) {
        uint16_t bits = __bfloat16_as_ushort(full_values[i]);
        uint8_t sign = (bits >> 15) & 0x1;
        uint8_t exponent = (bits >> 7) & 0xFF;
        uint8_t mantissa = bits & 0x7F;
        printf("  Element[%d] = %8.4f (sign=%d, exp=%d, mantissa=0x%02X)\n", 
               i, __bfloat162float(full_values[i]), sign, exponent, mantissa);
    }
    printf("\n");
}
// Save matrix comparison results to file
void save_matrix_comparison(const char* filename, const __nv_bfloat16* cublas_result, 
                            const __nv_bfloat16* our_result, int M, int N) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        printf("Cannot create comparison result file: %s\n", filename);
        return;
    }
    
    // Write header
    outfile << "Row,Col,CuBLAS,Our_Result,Difference\n";
    
    // Check at most 100x100 submatrix
    int display_rows = std::min(M, 100);
    int display_cols = std::min(N, 100);
    
    // Write data
    for (int i = 0; i < display_rows; i++) {
        for (int j = 0; j < display_cols; j++) {
            float cublas_val = __bfloat162float(cublas_result[i + j * M]);
            float our_val = __bfloat162float(our_result[i + j * M]);
            float diff = our_val - cublas_val;
            
            outfile << i << "," << j << "," 
                    << std::fixed << std::setprecision(6) << cublas_val << "," 
                    << our_val << "," << diff << "\n";
        }
    }
    
    outfile.close();
    printf("Matrix comparison results saved to: %s\n", filename);
}
// Save bitmap vs original data comparison
void save_bitmap_analysis(const char* filename, const __nv_bfloat16* A_h, 
                         const uint64_t* bitmap1, const uint64_t* bitmap2, 
                         const uint64_t* bitmap3, int M, int K, int tile_M, int tile_K) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        printf("Cannot create bitmap analysis file: %s\n", filename);
        return;
    }
    
    // Write header
    outfile << "TileRow,TileCol,Row,Col,Value,Sign,Exponent,Mantissa,Bitmap1,Bitmap2,Bitmap3,Code\n";
    
    // Analyze selected small tiles
    int tiles_to_analyze = std::min(16, (M / tile_M) * (K / tile_K));
    
    for (int tile_idx = 0; tile_idx < tiles_to_analyze; tile_idx++) {
        int tile_row = (tile_idx / (K / tile_K)) * tile_M;
        int tile_col = (tile_idx % (K / tile_K)) * tile_K;
        
        for (int i = 0; i < tile_M; i++) {
            for (int j = 0; j < tile_K; j++) {
                int row = tile_row + i;
                int col = tile_col + j;
                int pos = i * tile_K + j;
                
                if (row < M && col < K) {
                    __nv_bfloat16 val = A_h[row * K + col];
                    uint16_t bits = __bfloat16_as_ushort(val);
                    uint8_t sign = (bits >> 15) & 0x1;
                    uint8_t exponent = (bits >> 7) & 0xFF;
                    uint8_t mantissa = bits & 0x7F;
                    
                    uint64_t bit1 = (bitmap1[tile_idx] >> pos) & 1ULL;
                    uint64_t bit2 = (bitmap2[tile_idx] >> pos) & 1ULL;
                    uint64_t bit3 = (bitmap3[tile_idx] >> pos) & 1ULL;
                    uint8_t code = (bit3 << 2) | (bit2 << 1) | bit1;
                    
                    outfile << tile_idx / (K / tile_K) << "," 
                            << tile_idx % (K / tile_K) << "," 
                            << row << "," << col << "," 
                            << std::fixed << std::setprecision(6) << __bfloat162float(val) << "," 
                            << (int)sign << "," << (int)exponent << "," << (int)mantissa << "," 
                            << bit1 << "," << bit2 << "," << bit3 << "," << (int)code << "\n";
                }
            }
        }
    }
    
    outfile.close();
    printf("Bitmap analysis saved to: %s\n", filename);
}



int main(int argc, char** argv)
{
    // if (argc != 5) {
    //     printf("Wrong Inputs! Correct input format: ./bf16bitmap_test M K N SplitK\n");
    //     return -1;
    // }
    // int M_GLOBAL = atoi(argv[1]);
    // int K_GLOBAL = atoi(argv[2]);
    // int N_GLOBAL = atoi(argv[3]);
    // int SPLIT_K = atoi(argv[4]);

    
    // Default values
    std::string model_name = "unknown";
    std::string layer_name = "unknown";
    
    // Parameter parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_name = argv[++i];
        } else if (arg == "--layer" && i + 1 < argc) {
            layer_name = argv[++i];
        }
    }

    if (argc < 5) {  // Parameter count check updated to minimum 5
        printf("Wrong Inputs! Correct input format: ./bf16bitmap_test M K N SplitK [--model MODEL_NAME --layer LAYER_NAME]\n");
        return -1;
    }
    
    int M_GLOBAL = atoi(argv[1]);
    int K_GLOBAL = atoi(argv[2]);
    int N_GLOBAL = atoi(argv[3]);
    int SPLIT_K = atoi(argv[4]);
    

    // Set debug level (0-4)
    int debug_level = 0;  // Higher value prints more information
    
    printf("====== BF16 Triple Bitmap Compressed Matrix Multiplication Test ======\n");
    printf("Dimensions: M=%d, K=%d, N=%d, SPLIT_K=%d\n", M_GLOBAL, K_GLOBAL, N_GLOBAL, SPLIT_K);
    printf("Debug level: %d\n\n", debug_level);
    
    cublasStatus_t cublas_status;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Host memory
    __nv_bfloat16* A_h            = NULL;  // row major
    __nv_bfloat16* B_h            = NULL;  // col major
    __nv_bfloat16* B_Transposed_h = NULL;  // row major
    
    // Device memory
    __nv_bfloat16* A            = NULL;
    __nv_bfloat16* B            = NULL;
    __nv_bfloat16* B_Transposed = NULL;
    
    printf("Allocating host memory...\n");
    A_h            = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * M_GLOBAL * K_GLOBAL);
    B_h            = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * K_GLOBAL * N_GLOBAL);
    B_Transposed_h = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * K_GLOBAL * N_GLOBAL);
    if (A_h == NULL || B_h == NULL || B_Transposed_h == NULL) {
        printf("Error in CPU Malloc!\n");
        exit(-1);
    }
    
    printf("Allocating device memory...\n");
    cudaMalloc(reinterpret_cast<void**>(&A), sizeof(__nv_bfloat16) * M_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B), sizeof(__nv_bfloat16) * N_GLOBAL * K_GLOBAL);
    cudaMalloc(reinterpret_cast<void**>(&B_Transposed), sizeof(__nv_bfloat16) * N_GLOBAL * K_GLOBAL);
    checkLastCudaError(__LINE__);
    if (A == NULL || B == NULL || B_Transposed == NULL) {
        printf("Error in cudaMalloc!\n");
        exit(-1);
    }
    
    // Initialize host matrices
    printf("Initializing test matrices...\n");
    // init_host_matrices_bf16_ones(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL);
    init_host_matrices_bf16(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL);

    // Uncomment below for other initialization modes
    // init_host_matrices_bf16_pattern(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL);
    // init_host_matrices_bf16_fixed(A_h, B_h, M_GLOBAL, K_GLOBAL, N_GLOBAL, 1.0f, 1.0f);
    
    for (int i = 0; i < K_GLOBAL; i++)
        for (int j = 0; j < N_GLOBAL; j++)
            B_Transposed_h[i * N_GLOBAL + j] = B_h[i + j * K_GLOBAL];
    
    // Print partial input matrix data
    if (debug_level >= 1) {
        print_bf16_matrix("Matrix A (input)", A_h, M_GLOBAL, K_GLOBAL);
        // print_bf16_matrix("Matrix B (input)", B_h, K_GLOBAL, N_GLOBAL);
    }
    
    printf("Copying data to GPU...\n");
    cudaMemcpy(A, A_h, sizeof(__nv_bfloat16) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(__nv_bfloat16) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(B_Transposed, B_Transposed_h, sizeof(__nv_bfloat16) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);
    
    //================ CUBLAS Benchmark ================
    printf("\n========== Running CuBLAS Benchmark ==========\n");
    
    __nv_bfloat16* D_cublas_no_tc = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_cublas_no_tc), sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL);
    if (D_cublas_no_tc == NULL) {
        printf("Error: cudaMalloc failed, line: %d\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_cublas_no_tc, 0, sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL);
    
    __nv_bfloat16* D_cublas_tc = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_cublas_tc), sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL);
    if (D_cublas_tc == NULL) {
        printf("Error: cudaMalloc failed, line: %d\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_cublas_tc, 0, sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, 0);
    
    // Tensor core disabled
    printf("Running CuBLAS without Tensor Core...\n");
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    cudaDeviceSynchronize();
    int              m = M_GLOBAL, n = N_GLOBAL, k = K_GLOBAL;
    const float      alpha     = 1.0;
    const float      beta      = 0.0;
    cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0);
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     m,
                                     n,
                                     k,
                                     &alpha,
                                     A,
                                     CUDA_R_16BF,
                                     k,
                                     B,
                                     CUDA_R_16BF,
                                     k,
                                     &beta,
                                     D_cublas_no_tc,
                                     CUDA_R_16BF,
                                     m,
                                     CUDA_R_32F,
                                     CuBlasALG);
        checkCublasError(cublas_status, __LINE__);
    }
    
    // Flush L2 cache before CuBlas benchmark (without Tensor Core)
    flush_l2_cache();
    
    float total_milliseconds_cublas_no_tc = 0.0f;
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        // Flush L2 cache before each iteration to simulate real-world cold cache scenario
        flush_l2_cache();
        
        // Measure only the GEMM operation time, excluding cache flush overhead
        cudaEventRecord(start);
        cublasGemmEx(handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     m,
                     n,
                     k,
                     &alpha,
                     A,
                     CUDA_R_16BF,
                     k,
                     B,
                     CUDA_R_16BF,
                     k,
                     &beta,
                     D_cublas_no_tc,
                     CUDA_R_16BF,
                     m,
                     CUDA_R_32F,
                     CuBlasALG);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float iter_time = 0.0f;
        cudaEventElapsedTime(&iter_time, start, stop);
        total_milliseconds_cublas_no_tc += iter_time;
    }
    //
    float milliseconds_cublas_no_tc = total_milliseconds_cublas_no_tc / BENCHMARK_ITERATION;
    float tflops_cublas_no_tc =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds_cublas_no_tc / 1000.))
        / 1e12;
        
    // Tensor core enabled
    printf("Running CuBLAS with Tensor Core...\n");
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    cudaDeviceSynchronize();
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     m,
                                     n,
                                     k,
                                     &alpha,
                                     A,
                                     CUDA_R_16BF,
                                     k,
                                     B,
                                     CUDA_R_16BF,
                                     k,
                                     &beta,
                                     D_cublas_tc,
                                     CUDA_R_16BF,
                                     m,
                                     CUDA_R_32F,
                                     CuBlasALG);
        checkCublasError(cublas_status, __LINE__);
    }
    
    // Flush L2 cache before CuBlas benchmark (with Tensor Core)
    flush_l2_cache();
    
    float total_milliseconds_cublas_tc = 0.0f;
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        // Flush L2 cache before each iteration to simulate real-world cold cache scenario
        flush_l2_cache();
        
        // Measure only the GEMM operation time, excluding cache flush overhead
        cudaEventRecord(start);
        cublasGemmEx(handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     m,
                     n,
                     k,
                     &alpha,
                     A,
                     CUDA_R_16BF,
                     k,
                     B,
                     CUDA_R_16BF,
                     k,
                     &beta,
                     D_cublas_tc,
                     CUDA_R_16BF,
                     m,
                     CUDA_R_32F,
                     CuBlasALG);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float iter_time = 0.0f;
        cudaEventElapsedTime(&iter_time, start, stop);
        total_milliseconds_cublas_tc += iter_time;
    }
    //
    float milliseconds_cublas_tc = total_milliseconds_cublas_tc / BENCHMARK_ITERATION;
    float tflops_cublas_tc = static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2)
                                                 / (milliseconds_cublas_tc / 1000.))
                             / 1e12;
    
    // Copy results to host for verification
    __nv_bfloat16* D_cublas_no_tc_h = NULL;  // col major
    __nv_bfloat16* D_cublas_tc_h = NULL;     // col major
    
    D_cublas_no_tc_h = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL);
    D_cublas_tc_h = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL);
    
    if (D_cublas_no_tc_h == NULL || D_cublas_tc_h == NULL) {
        printf("Error: CPU memory allocation failed, line: %d\n", __LINE__);
        exit(-1);
    }
    
    cudaMemcpy(D_cublas_no_tc_h, D_cublas_no_tc, sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);
    cudaMemcpy(D_cublas_tc_h, D_cublas_tc, sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);
    
    // Print CuBLAS results
    if (debug_level >= 2) {
        print_bf16_matrix_col("CuBLAS non-TC result matrix", D_cublas_no_tc_h, M_GLOBAL, N_GLOBAL);
        print_bf16_matrix_col("CuBLAS TC result matrix", D_cublas_tc_h, M_GLOBAL, N_GLOBAL);
    }
    
    // Compare CuBLAS non-TC and TC result differences
    printf("\n========== CuBLAS Result Comparison (Tensor Core vs Non-Tensor Core) ==========\n");
    double totalError_cublas_tc_vs_no_tc = ComputeTotalError_BF16(D_cublas_no_tc_h, D_cublas_tc_h, M_GLOBAL, N_GLOBAL);
    
    // Calculate detailed error metrics
    double max_rel_error_cublas = 0.0;
    double avg_rel_error_cublas = 0.0;
    int error_count_cublas = 0;
    
    for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
        float no_tc_val = __bfloat162float(D_cublas_no_tc_h[i]);
        float tc_val = __bfloat162float(D_cublas_tc_h[i]);
        
        if (no_tc_val != 0.0f) {
            double rel_error = std::abs((tc_val - no_tc_val) / no_tc_val);
            avg_rel_error_cublas += rel_error;
            max_rel_error_cublas = std::max(max_rel_error_cublas, rel_error);
            
            if (rel_error > 1e-4) {
                error_count_cublas++;
            }
        } else if (tc_val != 0.0f) {
            error_count_cublas++;
        }
    }
    
    avg_rel_error_cublas /= M_GLOBAL * N_GLOBAL;
    
    printf("CuBLAS TC vs non-TC comparison results:\n");
    printf("  Total absolute error: %g\n", totalError_cublas_tc_vs_no_tc);
    printf("  Max relative error: %g\n", max_rel_error_cublas);
    printf("  Average relative error: %g\n", avg_rel_error_cublas);
    printf("  Significant error element count: %d (%.2f%%)\n", error_count_cublas, 100.0f * error_count_cublas / (M_GLOBAL * N_GLOBAL));
    
    // Save comparison results to file
    if (debug_level >= 3) {
        save_matrix_comparison("cublas_tc_vs_no_tc_comparison.csv", D_cublas_no_tc_h, D_cublas_tc_h, M_GLOBAL, N_GLOBAL);
    }
    
    // Print error samples if discrepancies are large
    if (error_count_cublas > 0 && debug_level >= 1) {
        printf("\nCuBLAS TC vs non-TC error samples (first 10):\n");
        printf("Index\tCoord\t\tNon-TC\t\tTC\t\tDifference\tRelative Error\n");
        
        int shown = 0;
        for (int i = 0; i < M_GLOBAL; i++) {
            for (int j = 0; j < N_GLOBAL; j++) {
                int idx = i + j * M_GLOBAL;  // Column-major
                float no_tc_val = __bfloat162float(D_cublas_no_tc_h[idx]);
                float tc_val = __bfloat162float(D_cublas_tc_h[idx]);
                float diff = tc_val - no_tc_val;
                float rel_err = no_tc_val != 0.0f ? std::abs(diff / no_tc_val) : std::abs(tc_val);
                
                if (rel_err > 1e-4 && shown < 100) {
                    printf("%d\t[%d,%d]\t\t%f\t%f\t%f\t%f\n", 
                           idx, i, j, no_tc_val, tc_val, diff, rel_err);
                    shown++;
                }
            }
        }
    }
    
    auto Split_K = SPLIT_K;
    
    //================ BF16 Triple Bitmap Compressed Matrix Multiplication Test ================
    printf("\n========== Running BF16 Triple Bitmap Compressed Matrix Multiplication Test ==========\n");
    __nv_bfloat16* D_BF16TripleBitmap = NULL;
    cudaMalloc(reinterpret_cast<void**>(&D_BF16TripleBitmap), sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL);
    if (D_BF16TripleBitmap == NULL) {
        printf("Error: cudaMalloc failed, line: %d\n", __LINE__);
        exit(-1);
    }
    cudaMemset(D_BF16TripleBitmap, 0, sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL);
    
    printf("Compressing matrix...\n");
    // Define compressed data pointers
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
    uint8_t start_exp = 0;
    
    // Compress matrix A
    int num_global_tiles = InitBF16MatrixTripleBitmap_Host(
        A_h, M_GLOBAL, K_GLOBAL, 
        8, 16, 64, 8, 64, 64,
        &top_exponents_cpu, &compressed_full_cpu, &sign_mantissa_cpu,
        &bitmap1_cpu, &bitmap2_cpu, &bitmap3_cpu,
        &TileOffsets_cpu, &TileOffsets_median_cpu, &TileOffsets_global_cpu,
        max_high_freq_count, max_full_count, start_exp);
    
    if (num_global_tiles <= 0) {
        printf("Error: BF16 triple bitmap compression initialization failed\n");
        exit(-1);
    }
    
    int tile_m = 8;
    int tile_k = 8;
    int tile_m_global = 64;
    int tile_k_global = 64;
    
    int num_tiles_m = M_GLOBAL / tile_m;
    int num_tiles_k = K_GLOBAL / tile_k;
    int num_tiles = num_tiles_m * num_tiles_k;
    
    int num_median_tiles_m = M_GLOBAL / 16;
    int num_median_tiles_k = K_GLOBAL / 64;
    int num_median_tiles = num_median_tiles_m * num_median_tiles_k;
    
    // Get total element count after compression
    int high_freq_count = TileOffsets_global_cpu[num_global_tiles * 2];
    int full_count = TileOffsets_global_cpu[num_global_tiles * 2 + 1];
    
    printf("Compression statistics:\n");
    printf("  Global tile count: %d\n", num_global_tiles);
    printf("  Medium tile count: %d\n", num_median_tiles);
    printf("  Small tile count: %d\n", num_tiles);
    printf("  High-freq elements: %d (%.2f%%)\n", high_freq_count, 100.0f * high_freq_count / (M_GLOBAL * K_GLOBAL));
    printf("  Non-high-freq elements: %d (%.2f%%)\n", full_count, 100.0f * full_count / (M_GLOBAL * K_GLOBAL));
    printf("  Max high-freq elements per tile: %d\n", max_high_freq_count);
    printf("  Max non-high-freq elements per tile: %d\n", max_full_count);
    printf("  High-freq starting value: %u\n", start_exp);
    
    // Original size and compressed size
    size_t original_size = M_GLOBAL * K_GLOBAL * sizeof(__nv_bfloat16);
    size_t compressed_size = 
        // (7 * sizeof(__nv_bfloat16)) +                    // High-freq exponents
        (high_freq_count * sizeof(uint8_t)) +            // High-freq elements (sign+mantissa)
        (full_count * sizeof(__nv_bfloat16)) +           // Non-high-freq elements
        (num_tiles * sizeof(uint64_t) * 3) +             // Three bitmaps
        // (num_tiles * 2 * sizeof(int)) +                  // Small tile offsets
        (num_median_tiles * 2 * sizeof(int)) +           // Medium tile offsets
        ((num_global_tiles + 1) * 2 * sizeof(int));      // Global tile offsets
    
    float compression_ratio = (float)original_size / compressed_size;
    printf("  Original size: %zu bytes\n", original_size);
    printf("  Compressed size: %zu bytes\n", compressed_size);
    printf("  Compression ratio: %.2fx\n", compression_ratio);
    
    // Print more compressed data details
    if (debug_level >= 3) {
        print_bitmap_data(bitmap1_cpu, bitmap2_cpu, bitmap3_cpu, TileOffsets_cpu, num_tiles);
        print_compressed_data(top_exponents_cpu, sign_mantissa_cpu, compressed_full_cpu, high_freq_count, full_count);
        
        // Save bitmap analysis to file
        save_bitmap_analysis("bitmap_analysis.csv", A_h, bitmap1_cpu, bitmap2_cpu, bitmap3_cpu, 
                             M_GLOBAL, K_GLOBAL, tile_m, tile_k);
    }
    
    printf("Allocating GPU memory for compressed data...\n");
    // Allocate device memory for compressed data
    __nv_bfloat16* top_exponents_gpu = nullptr;
    __nv_bfloat16* compressed_full_gpu = nullptr;
    uint8_t* sign_mantissa_gpu = nullptr;
    uint64_t* bitmap1_gpu = nullptr;
    uint64_t* bitmap2_gpu = nullptr;
    uint64_t* bitmap3_gpu = nullptr;
    int* TileOffsets_gpu = nullptr;
    int* TileOffsets_median_gpu = nullptr;
    int* TileOffsets_global_gpu = nullptr;
    
    cudaMalloc(&top_exponents_gpu, 7 * sizeof(__nv_bfloat16)); // 7 high-freq exponents
    cudaMalloc(&compressed_full_gpu, full_count * sizeof(__nv_bfloat16));
    cudaMalloc(&sign_mantissa_gpu, high_freq_count * sizeof(uint8_t));
    cudaMalloc(&bitmap1_gpu, num_tiles * sizeof(uint64_t));
    cudaMalloc(&bitmap2_gpu, num_tiles * sizeof(uint64_t));
    cudaMalloc(&bitmap3_gpu, num_tiles * sizeof(uint64_t));
    cudaMalloc(&TileOffsets_gpu, num_tiles * 2 * sizeof(int));
    cudaMalloc(&TileOffsets_median_gpu, num_median_tiles * 2 * sizeof(int));
    cudaMalloc(&TileOffsets_global_gpu, (num_global_tiles + 1) * 2 * sizeof(int));
    
    int* max_high_freq_gpu = nullptr;
    int* max_full_gpu = nullptr;
    cudaMalloc(&max_high_freq_gpu, sizeof(int));
    cudaMalloc(&max_full_gpu, sizeof(int));
    
    printf("Copying compressed data to GPU...\n");
    // Copy compressed data to device
    cudaMemcpy(top_exponents_gpu, top_exponents_cpu, 7 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(compressed_full_gpu, compressed_full_cpu, full_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(sign_mantissa_gpu, sign_mantissa_cpu, high_freq_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap1_gpu, bitmap1_cpu, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap2_gpu, bitmap2_cpu, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(bitmap3_gpu, bitmap3_cpu, num_tiles * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(TileOffsets_gpu, TileOffsets_cpu, num_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(TileOffsets_median_gpu, TileOffsets_median_cpu, num_median_tiles * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(TileOffsets_global_gpu, TileOffsets_global_cpu, (num_global_tiles + 1) * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(max_high_freq_gpu, &max_high_freq_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(max_full_gpu, &max_full_count, sizeof(int), cudaMemcpyHostToDevice);
    
    // Free host compressed data memory
    free(top_exponents_cpu);
    free(compressed_full_cpu);
    free(sign_mantissa_cpu);
    free(bitmap1_cpu);
    free(bitmap2_cpu);
    free(bitmap3_cpu);
    free(TileOffsets_cpu);
    free(TileOffsets_median_cpu);
    free(TileOffsets_global_cpu);
    
    printf("Compressed data ready, starting BF16 triple bitmap compression kernel...\n");
    Split_K = SPLIT_K;
    printf("Split_K = %d\n", Split_K);
    __nv_bfloat16* Reduction_Workspace_BF16TripleBitmap = NULL;
    cudaMalloc(reinterpret_cast<void**>(&Reduction_Workspace_BF16TripleBitmap), 
               sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL * Split_K);
    if (Reduction_Workspace_BF16TripleBitmap == NULL) {
        printf("Error: cudaMalloc failed\n");
        exit(-1);
    }
    
    printf("Running warmup...\n");
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        // printf("  Warmup iteration %d/%d\n", i+1, WARM_UP_ITERATION);
        BF16TripleBitmap_MM_API(0,
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
                        B,
                        D_BF16TripleBitmap,
                        M_GLOBAL,
                        N_GLOBAL,
                        K_GLOBAL,
                        Reduction_Workspace_BF16TripleBitmap,
                        Split_K);
        cudaDeviceSynchronize();
        checkLastCudaError(__LINE__);
    }
    
    // Flush L2 cache before BF16TripleBitmap benchmark
    flush_l2_cache();
    
    printf("Running benchmark...\n");
    float total_milliseconds_BF16TripleBitmap = 0.0f;
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        // Flush L2 cache before each iteration to simulate real-world cold cache scenario
        flush_l2_cache();
        
        // Measure only the MM operation time, excluding cache flush overhead
        cudaEventRecord(start);
        BF16TripleBitmap_MM_API(0,
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
                        B,
                        D_BF16TripleBitmap,
                        M_GLOBAL,
                        N_GLOBAL,
                        K_GLOBAL,
                        Reduction_Workspace_BF16TripleBitmap,
                        Split_K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        checkLastCudaError(__LINE__);
        
        float iter_time = 0.0f;
        cudaEventElapsedTime(&iter_time, start, stop);
        total_milliseconds_BF16TripleBitmap += iter_time;
    }
    
    float milliseconds_BF16TripleBitmap = total_milliseconds_BF16TripleBitmap / BENCHMARK_ITERATION;
    float tflops_BF16TripleBitmap =
        static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / 
                           (milliseconds_BF16TripleBitmap / 1000.)) / 1e12;
    __nv_bfloat16* D_BF16TripleBitmap_h = NULL;  // col major
    D_BF16TripleBitmap_h = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL);
    cudaMemcpy(D_BF16TripleBitmap_h, D_BF16TripleBitmap, 
              sizeof(__nv_bfloat16) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);  // Col Major
    
    // Print result matrix
    if (debug_level >= 2) {
        print_bf16_matrix_col("BF16 triple bitmap compression result matrix", D_BF16TripleBitmap_h, M_GLOBAL, N_GLOBAL);
    }
    
    // Calculate verification error - compare with both non-TC and TC versions
    printf("\n========== Result Verification ==========\n");
    
    // Compare with non-TC CuBLAS
    double totalError_BF16TripleBitmap_vs_no_tc = 0.0;
    totalError_BF16TripleBitmap_vs_no_tc = ComputeTotalError_BF16(D_cublas_no_tc_h, D_BF16TripleBitmap_h, M_GLOBAL, N_GLOBAL);
    
    // Compare with TC CuBLAS
    double totalError_BF16TripleBitmap_vs_tc = 0.0;
    totalError_BF16TripleBitmap_vs_tc = ComputeTotalError_BF16(D_cublas_tc_h, D_BF16TripleBitmap_h, M_GLOBAL, N_GLOBAL);
    
    // Save matrix comparison results
    if (debug_level >= 3) {
        save_matrix_comparison("matrix_comparison_no_tc.csv", D_cublas_no_tc_h, D_BF16TripleBitmap_h, M_GLOBAL, N_GLOBAL);
        save_matrix_comparison("matrix_comparison_tc.csv", D_cublas_tc_h, D_BF16TripleBitmap_h, M_GLOBAL, N_GLOBAL);
    }
    
    // Calculate relative error (compare with non-TC CuBLAS)
    double max_rel_error_no_tc = 0.0;
    double avg_rel_error_no_tc = 0.0;
    int error_count_no_tc = 0;
    
    for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
        float cublas_val = __bfloat162float(D_cublas_no_tc_h[i]);
        float our_val = __bfloat162float(D_BF16TripleBitmap_h[i]);
        
        if (cublas_val != 0.0f) {
            double rel_error = std::abs((our_val - cublas_val) / cublas_val);
            avg_rel_error_no_tc += rel_error;
            max_rel_error_no_tc = std::max(max_rel_error_no_tc, rel_error);
            
            if (rel_error > 1e-4) {
                error_count_no_tc++;
            }
        } else if (our_val != 0.0f) {
            error_count_no_tc++;
        }
    }
    
    avg_rel_error_no_tc /= M_GLOBAL * N_GLOBAL;
    
    // Calculate relative error (compare with TC CuBLAS)
    double max_rel_error_tc = 0.0;
    double avg_rel_error_tc = 0.0;
    int error_count_tc = 0;
    
    for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
        float cublas_val = __bfloat162float(D_cublas_tc_h[i]);
        float our_val = __bfloat162float(D_BF16TripleBitmap_h[i]);
        
        if (cublas_val != 0.0f) {
            double rel_error = std::abs((our_val - cublas_val) / cublas_val);
            avg_rel_error_tc += rel_error;
            max_rel_error_tc = std::max(max_rel_error_tc, rel_error);
            
            if (rel_error > 1e-4) {
                error_count_tc++;
            }
        } else if (our_val != 0.0f) {
            error_count_tc++;
        }
    }
    
    avg_rel_error_tc /= M_GLOBAL * N_GLOBAL;
    
    printf("Verification results:\n");
    printf("Triple bitmap vs non-TC CuBLAS:\n");
    printf("  Total absolute error: %g\n", totalError_BF16TripleBitmap_vs_no_tc);
    printf("  Max relative error: %g\n", max_rel_error_no_tc);
    printf("  Average relative error: %g\n", avg_rel_error_no_tc);
    printf("  Significant error element count: %d (%.2f%%)\n", error_count_no_tc, 100.0f * error_count_no_tc / (M_GLOBAL * N_GLOBAL));
    
    printf("\nTriple bitmap vs TC CuBLAS:\n");
    printf("  Total absolute error: %g\n", totalError_BF16TripleBitmap_vs_tc);
    printf("  Max relative error: %g\n", max_rel_error_tc);
    printf("  Average relative error: %g\n", avg_rel_error_tc);
    printf("  Significant error element count: %d (%.2f%%)\n", error_count_tc, 100.0f * error_count_tc / (M_GLOBAL * N_GLOBAL));
    
    // Print error samples if discrepancies are large (compare triple bitmap with TC CuBLAS)
    if (error_count_no_tc > 0 && debug_level >= 1) {
        printf("\nError samples (triple bitmap vs TC CuBLAS) (first 10):\n");
        printf("Index\tCoord\t\tCuBLAS non-TC\tTriple bitmap\tDifference\tRelative Error\n");
        
        int shown = 0;
        for (int i = 0; i < M_GLOBAL; i++) {
            for (int j = 0; j < N_GLOBAL; j++) {
                int idx = i + j * M_GLOBAL;  // Column-major
                float cublas_val = __bfloat162float(D_cublas_tc_h[idx]);
                float our_val = __bfloat162float(D_BF16TripleBitmap_h[idx]);
                float diff = our_val - cublas_val;
                float rel_err = cublas_val != 0.0f ? std::abs(diff / cublas_val) : std::abs(our_val);
                
                if (rel_err > 1e-4 && shown < 100) {
                    printf("%d\t[%d,%d]\t\t%f\t%f\t%f\t%f\n", 
                           idx, i, j, cublas_val, our_val, diff, rel_err);
                    shown++;
                }
            }
        }
    }
    
    // Free GPU memory
    cudaFree(D_BF16TripleBitmap);
    cudaFree(D_cublas_no_tc);
    cudaFree(D_cublas_tc);
    cudaFree(top_exponents_gpu);
    cudaFree(compressed_full_gpu);
    cudaFree(sign_mantissa_gpu);
    cudaFree(bitmap1_gpu);
    cudaFree(bitmap2_gpu);
    cudaFree(bitmap3_gpu);
    cudaFree(TileOffsets_gpu);
    cudaFree(TileOffsets_median_gpu);
    cudaFree(TileOffsets_global_gpu);
    cudaFree(max_high_freq_gpu);
    cudaFree(max_full_gpu);
    cudaFree(Reduction_Workspace_BF16TripleBitmap);
    
    // Print performance results
    printf("\n========== Performance Results ==========\n");
    PrintPerformance("BF16_triple_bitmap", milliseconds_BF16TripleBitmap, tflops_BF16TripleBitmap, totalError_BF16TripleBitmap_vs_tc);
    PrintPerformance("CuBLAS_TC", milliseconds_cublas_tc, tflops_cublas_tc, 0.0);
    PrintPerformance("CuBLAS_non-TC", milliseconds_cublas_no_tc, tflops_cublas_no_tc, totalError_cublas_tc_vs_no_tc);
    
    // Free remaining host memory
    free(D_cublas_no_tc_h);
    free(D_cublas_tc_h);
    free(D_BF16TripleBitmap_h);
    free(A_h);
    free(B_h);
    free(B_Transposed_h);
    cudaFree(A);
    cudaFree(B);
    cudaFree(B_Transposed);
    
    // Save benchmark results to CSV
    SavePerformanceData("bf16_triplebm_res.csv",
        model_name.c_str(), layer_name.c_str(),
        M_GLOBAL, K_GLOBAL, N_GLOBAL, 
        SPLIT_K,
        milliseconds_cublas_tc, tflops_cublas_tc,
        milliseconds_BF16TripleBitmap, tflops_BF16TripleBitmap, 
        milliseconds_cublas_no_tc, tflops_cublas_no_tc);
    printf("\n========== Test Complete ==========\n");
    return 0;
}