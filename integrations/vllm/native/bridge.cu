// Copyright 2026 Kyrylo Bilchenko
// SPDX-License-Identifier: Apache-2.0
// Adapted from MxWave; modified to use the local public compressor and ABI v2.
// MxWave's independent C ABI bridge to an external Apache-2.0 ZipServ checkout.
// No torch C++ ABI dependency; device allocations belong to the Python caller.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include "../../../build/L_API.cuh"
#include "../../../csrc/HostPacker.h"

// Layout version 1. All lengths are bytes, all seven buffers have independent ownership.
struct Packed {
    void* buffers[7];
    uint64_t lengths[7];
    int32_t m, k, max_common, max_rare, start_exp;
};
static thread_local std::string last_error;

extern "C" const char* zipserv_error() { return last_error.c_str(); }
extern "C" int zipserv_abi_version() { return 2; }
extern "C" void zipserv_free(Packed* out) {
    for (auto& pointer : out->buffers) { std::free(pointer); pointer = nullptr; }
}

extern "C" int zipserv_compress(const void* weights, int m, int k, Packed* out) {
    if (!out) { last_error = "Missing output descriptor"; return -1; }
    std::memset(out, 0, sizeof(*out));
    // The upstream compressor uses signed int offsets and products.
    if (!weights || m <= 0 || k <= 0 || m % 64 || k % 64 ||
        uint64_t(m) * k > uint64_t(std::numeric_limits<int>::max()) - 4096) {
        last_error = "BF16 matrix must have positive 64-aligned dimensions and int32 offsets";
        return -1;
    }
    try {
        auto packed = zipserv::pack_bf16(weights, m, k);
        const auto lengths = packed.lengths();
        const int indices[] = {0, 1, 2, 3, 4, 6, 7};
        for (int i = 0; i < 7; ++i) out->lengths[i] = lengths[indices[i]];
        out->m = m; out->k = k; out->max_common = packed.max_common;
        out->max_rare = packed.max_rare; out->start_exp = uint8_t(packed.base - 1);
        void* buffers[] = {packed.common.release(), packed.rare.release(),
            packed.b1.release(), packed.b2.release(), packed.b3.release(),
            packed.medium.release(), packed.global.release()};
        std::memcpy(out->buffers, buffers, sizeof(buffers));
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

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
    if (result != cudaSuccess) last_error = cudaGetErrorString(result);
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
    if (result != cudaSuccess) last_error = cudaGetErrorString(result);
    return int(result);
}
