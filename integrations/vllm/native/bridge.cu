// Copyright 2026 Kyrylo Bilchenko
// SPDX-License-Identifier: Apache-2.0
// Adapted from MxWave. GPU entry points share the existing ABI and error channel
// with host_bridge.cpp; device allocations belong to the Python caller.
#include "host_bridge.h"
#include "../../../build/L_API.cuh"

extern "C" int zipserv_mm(void* stream, const void* sm, const void* rare,
    const void* b1, const void* b2, const void* b3, const void* medium, const void* global,
    int max_common, int max_rare, int start_exp, const void* x, void* y,
    int m, int n, int k, void* workspace, int split_k) {
    auto result = BF16TripleBitmap_MM_API(static_cast<cudaStream_t>(stream),
        static_cast<const uint8_t*>(sm), static_cast<const __nv_bfloat16*>(rare),
        static_cast<const uint64_t*>(b1), static_cast<const uint64_t*>(b2),
        static_cast<const uint64_t*>(b3), static_cast<const int*>(medium),
        static_cast<const int*>(global), max_common, max_rare, uint8_t(start_exp),
        static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(y), m, n, k,
        static_cast<__nv_bfloat16*>(workspace), split_k);
    if (result != cudaSuccess) zipserv_last_error = cudaGetErrorString(result);
    return int(result);
}

extern "C" int zipserv_decode(void* stream, const void* sm, const void* rare,
    const void* b1, const void* b2, const void* b3, const void* medium, const void* global,
    int max_common, int max_rare, int start_exp, void* output, int m, int k) {
    auto result = BF16TripleBitmap_Decompress_API(static_cast<cudaStream_t>(stream),
        static_cast<const uint8_t*>(sm), static_cast<const __nv_bfloat16*>(rare),
        static_cast<const uint64_t*>(b1), static_cast<const uint64_t*>(b2),
        static_cast<const uint64_t*>(b3), static_cast<const int*>(medium),
        static_cast<const int*>(global), max_common, max_rare, uint8_t(start_exp),
        static_cast<__nv_bfloat16*>(output), m, k);
    if (result != cudaSuccess) zipserv_last_error = cudaGetErrorString(result);
    return int(result);
}

// Optional extension to ABI 2. The old symbol keeps its BF16 workspace contract;
// this separate symbol always interprets nonempty scratch as FP32.
extern "C" int zipserv_mm_fp32(void* stream, const void* sm, const void* rare,
    const void* b1, const void* b2, const void* b3, const void* medium, const void* global,
    int max_common, int max_rare, int start_exp, const void* x, void* y,
    int m, int n, int k, void* workspace, int split_k) {
    auto result = BF16TripleBitmap_MM_API_FP32(static_cast<cudaStream_t>(stream),
        static_cast<const uint8_t*>(sm), static_cast<const __nv_bfloat16*>(rare),
        static_cast<const uint64_t*>(b1), static_cast<const uint64_t*>(b2),
        static_cast<const uint64_t*>(b3), static_cast<const int*>(medium),
        static_cast<const int*>(global), max_common, max_rare, uint8_t(start_exp),
        static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(y), m, n, k,
        static_cast<float*>(workspace), split_k);
    if (result != cudaSuccess) zipserv_last_error = cudaGetErrorString(result);
    return int(result);
}
