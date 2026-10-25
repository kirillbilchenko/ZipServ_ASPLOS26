// Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
// Experimental exact byte decoder, GEMV and synchronous Tensor Core probes.
// The host must validate the complete immutable container before launching.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cstdint>
#include <cstddef>

template<int Group>
__device__ __forceinline__ uint32_t fp8_range_word(const unsigned char* data,
                                                 size_t word, int tile_bytes) {
    const unsigned shift = tile_bytes == 2048 ? 9 : 8;
    const uint64_t entry = reinterpret_cast<const uint64_t*>(data + 64)[word >> shift];
    const auto* payload = data + (entry & ~uint64_t(1));
    const unsigned offset = word & ((1u << shift) - 1);
    if (entry & 1) return reinterpret_cast<const uint32_t*>(payload)[offset];
    const unsigned group = offset / (Group / 4), local = offset % (Group / 4);
    const auto* masks = reinterpret_cast<const uint64_t*>(payload);
    const uint64_t mask = masks[group / 64];
    const unsigned bit = group & 63;
    unsigned raw_before = __popcll(mask & ((uint64_t(1) << bit) - 1));
    if (group >= 64) raw_before += __popcll(masks[0]);
    const unsigned prefix = ((tile_bytes / Group + 63) / 64) * 8;
    const auto* block = payload + prefix + group * (Group * 7 / 8) + raw_before * (Group / 8);
    if ((mask >> bit) & 1) {
        // Sixteen-value compressed blocks are 14 bytes; subsequent raw blocks
        // can be only two-byte aligned. Never issue an unaligned U32 read.
        const auto* p = reinterpret_cast<const uint16_t*>(block) + local * 2;
        return unsigned(p[0]) | (unsigned(p[1]) << 16);
    }
    const unsigned residual = reinterpret_cast<const uint16_t*>(block)[local];
    const auto* exponents = block + Group / 2;
    const unsigned first_bit = local * 12;
    unsigned fields = (unsigned(exponents[first_bit / 8])
        | (unsigned(exponents[first_bit / 8 + 1]) << 8)) >> (first_bit % 8);
    const unsigned base = data[4];
    unsigned result = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned low = (residual >> (j * 4)) & 15;
        const unsigned exp = ((fields >> (j * 3)) & 7) + base;
        result |= ((low & 7) | ((low & 8) << 4) | (exp << 3)) << (j * 8);
    }
    return result;
}

template<int Bits>
__device__ __forceinline__ uint32_t fp8_word(const unsigned char* data,
                                           size_t word, int tile_bytes) {
    if constexpr (Bits == 16 || Bits == 32) {
        return fp8_range_word<Bits>(data, word, tile_bytes);
    } else {
        // Only the two validated power-of-two tile sizes are supported. Avoid
        // runtime 64-bit division/modulo in every decoded four-byte word.
        const unsigned shift = tile_bytes == 2048 ? 9 : 8;
        const size_t tile = word >> shift;
        const unsigned offset = word & ((1u << shift) - 1);
        const uint64_t entry = reinterpret_cast<const uint64_t*>(data + 64)[tile];
        const auto* payload = data + (entry & ~uint64_t(1));
        if (entry & 1) return reinterpret_cast<const uint32_t*>(payload)[offset];
        constexpr unsigned Escape = (1u << Bits) - 1;
        const unsigned prefix_bytes = (tile_bytes / 128) * 2;
        const auto* selectors = payload + prefix_bytes + tile_bytes / 2;
        const auto* exceptions = selectors + tile_bytes * Bits / 8;
        const unsigned bit = offset * 4 * Bits;
        unsigned tags = selectors[bit / 8];
        if constexpr (Bits == 3) tags |= unsigned(selectors[bit / 8 + 1]) << 8;
        tags >>= bit % 8;
        const unsigned residual = reinterpret_cast<const uint16_t*>(payload + prefix_bytes)[offset];
        const uint64_t palette = *reinterpret_cast<const uint64_t*>(data + 24);
        unsigned masks[4], fields[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            fields[j] = (tags >> (j * Bits)) & Escape;
            masks[j] = __ballot_sync(0xffffffffu, fields[j] == Escape);
        }
        const unsigned lane = threadIdx.x & 31;
        const unsigned previous = (1u << lane) - 1;
        unsigned exception = reinterpret_cast<const uint16_t*>(payload)[offset / 32];
        #pragma unroll
        for (int j = 0; j < 4; ++j) exception += __popc(masks[j] & previous);
        unsigned result = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned exponent;
            if (fields[j] == Escape) {
                exponent = (exceptions[exception / 2] >> ((exception & 1) * 4)) & 15;
                ++exception;
            } else {
                exponent = (palette >> (8 * fields[j])) & 15;
            }
            const unsigned low = (residual >> (4 * j)) & 15;
            result |= ((low & 7) | ((low & 8) << 4) | (exponent << 3)) << (8 * j);
        }
        return result;
    }
}

template<int Bits>
__global__ void fp8_decode(const unsigned char* data, uint32_t* output,
                           int tile_bytes) {
    const size_t first = size_t(blockIdx.x) * (tile_bytes / 4);
    for (int i = threadIdx.x; i < tile_bytes / 4; i += blockDim.x)
        output[first + i] = fp8_word<Bits>(data, first + i, tile_bytes);
}

template<int Bits, bool Compressed>
__global__ void fp8_gemv(const unsigned char* data, const unsigned char* scales,
                        const __nv_bfloat16* x, __nv_bfloat16* y,
                        int n, int k, int tile_bytes) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    float sum = 0;
    for (int col = threadIdx.x * 4; col < k; col += blockDim.x * 4) {
        const size_t index = (size_t(row) * k + col) / 4;
        const unsigned codes = Compressed ? fp8_word<Bits>(data, index, tile_bytes)
            : reinterpret_cast<const uint32_t*>(data)[index];
        const unsigned scale = scales[size_t(row) * (k / 32) + col / 32];
        const float multiplier = __uint_as_float(scale ? scale << 23 : 0x00400000u);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __half value = __nv_cvt_fp8_to_halfraw((codes >> (8 * j)) & 255, __NV_E4M3);
            sum = fmaf(__half2float(value) * multiplier,
                       __bfloat162float(x[size_t(token) * k + col + j]), sum);
        }
    }
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, delta);
    __shared__ float partial[8];
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < 8 ? partial[threadIdx.x] : 0;
        #pragma unroll
        for (int delta = 16; delta; delta >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, delta);
        if (threadIdx.x == 0) y[size_t(token) * n + row] = __float2bfloat16_rn(sum);
    }
}

extern "C" int zipserv_fp8_decode(const void* encoded, void* output,
                                  size_t raw_bytes, int bits, int tile_bytes,
                                  cudaStream_t stream) {
    if (!encoded || !output || uintptr_t(encoded) % 8 || uintptr_t(output) % 4
            || !raw_bytes || raw_bytes > (size_t(1) << 36)
            || (bits != 2 && bits != 3 && bits != 16 && bits != 32) || (tile_bytes != 1024 && tile_bytes != 2048)
            || raw_bytes % tile_bytes) return int(cudaErrorInvalidValue);
    const auto* source = static_cast<const unsigned char*>(encoded);
    auto* target = static_cast<uint32_t*>(output);
    if (bits == 2) fp8_decode<2><<<raw_bytes / tile_bytes, 256, 0, stream>>>(source, target, tile_bytes);
    else if (bits == 3) fp8_decode<3><<<raw_bytes / tile_bytes, 256, 0, stream>>>(source, target, tile_bytes);
    else if (bits == 16) fp8_decode<16><<<raw_bytes / tile_bytes, 256, 0, stream>>>(source, target, tile_bytes);
    else fp8_decode<32><<<raw_bytes / tile_bytes, 256, 0, stream>>>(source, target, tile_bytes);
    return int(cudaGetLastError());
}

extern "C" int zipserv_fp8_gemv(const void* weights, const void* scales,
                                const void* x, void* y, int n, int k, int m,
                                int bits, int tile_bytes, cudaStream_t stream) {
    // bits=0 is the identical raw-layout arithmetic control, never a serving fallback.
    if (!weights || !scales || !x || !y || uintptr_t(weights) % 8
            || uintptr_t(x) % 2 || uintptr_t(y) % 2 || n < 1 || n > 131072
            || k < 128 || k > 131072 || k % 128 || m < 1 || m > 8
            || (bits != 0 && bits != 2 && bits != 3 && bits != 16 && bits != 32)
            || (tile_bytes != 1024 && tile_bytes != 2048)
            || (size_t(n) * k) % tile_bytes) return int(cudaErrorInvalidValue);
    const auto* w = static_cast<const unsigned char*>(weights);
    const auto* s = static_cast<const unsigned char*>(scales);
    const auto* a = static_cast<const __nv_bfloat16*>(x);
    auto* b = static_cast<__nv_bfloat16*>(y);
    const dim3 grid(n, m);
    if (bits == 0) fp8_gemv<2, false><<<grid, 256, 0, stream>>>(w, s, a, b, n, k, tile_bytes);
    else if (bits == 2) fp8_gemv<2, true><<<grid, 256, 0, stream>>>(w, s, a, b, n, k, tile_bytes);
    else if (bits == 3) fp8_gemv<3, true><<<grid, 256, 0, stream>>>(w, s, a, b, n, k, tile_bytes);
    else if (bits == 16) fp8_gemv<16, true><<<grid, 256, 0, stream>>>(w, s, a, b, n, k, tile_bytes);
    else fp8_gemv<32, true><<<grid, 256, 0, stream>>>(w, s, a, b, n, k, tile_bytes);
    return int(cudaGetLastError());
}

// Decode a weight tile once into shared BF16 storage, then reuse it across
// 32 activation rows with Tensor Cores. This is a synchronous WMMA probe;
// it deliberately does not claim Marlin's pipeline or scheduling quality.
template<int Bits, bool Compressed, bool Tiled = false>
__global__ void fp8_mma(const unsigned char* data, const unsigned char* scales,
                       const __nv_bfloat16* x, __nv_bfloat16* y,
                       int n, int k, int m, int tile_bytes) {
    constexpr int BM = 32, BN = 64, BK = 128, LD = BK + 8;
    __shared__ __align__(32) __nv_bfloat16 a[BM * LD];
    __shared__ __align__(32) __nv_bfloat16 b[BN * LD];
    __shared__ __align__(32) float c[BM * BN];
    const int first_n = blockIdx.x * BN, first_m = blockIdx.y * BM;
    const int warp = threadIdx.x / 32;
    const int warp_m = warp / 4, warp_n = warp % 4;
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);
    for (int first_k = 0; first_k < k; first_k += BK) {
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            const int row = i / BK, col = i % BK;
            a[row * LD + col] = first_m + row < m
                ? x[size_t(first_m + row) * k + first_k + col] : __float2bfloat16_rn(0);
        }
        for (int i = threadIdx.x; i < BN * BK / 4; i += blockDim.x) {
            const int row = i / (BK / 4), col = (i % (BK / 4)) * 4;
            const size_t index = Tiled
                ? (size_t(blockIdx.x) * (k / BK) + first_k / BK) * (BN * BK / 4) + i
                : (size_t(first_n + row) * k + first_k + col) / 4;
            const unsigned codes = Compressed ? fp8_word<Bits>(data, index, tile_bytes)
                : reinterpret_cast<const uint32_t*>(data)[index];
            const unsigned scale = scales[size_t(first_n + row) * (k / 32) + (first_k + col) / 32];
            const float multiplier = __uint_as_float(scale ? scale << 23 : 0x00400000u);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const __half value = __nv_cvt_fp8_to_halfraw((codes >> (8 * j)) & 255, __NV_E4M3);
                b[row * LD + col + j] = __float2bfloat16_rn(__half2float(value) * multiplier);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int inner = 0; inner < BK; inner += 16) {
            wmma::load_matrix_sync(fa, a + warp_m * 16 * LD + inner, LD);
            wmma::load_matrix_sync(fb, b + warp_n * 16 * LD + inner, LD);
            wmma::mma_sync(fc, fa, fb, fc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(c + warp_m * 16 * BN + warp_n * 16, fc, BN, wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        const int row = i / BN, col = i % BN;
        if (first_m + row < m)
            y[size_t(first_m + row) * n + first_n + col] = __float2bfloat16_rn(c[i]);
    }
}

extern "C" int zipserv_fp8_mma(const void* weights, const void* scales,
                               const void* x, void* y, int n, int k, int m,
                               int bits, int tile_bytes, cudaStream_t stream) {
    if (!weights || !scales || !x || !y || uintptr_t(weights) % 8
            || uintptr_t(x) % 2 || uintptr_t(y) % 2 || n < 64 || n > 131072 || n % 64
            || k < 128 || k > 131072 || k % 128 || m < 1 || m > 1024
            || (bits != 0 && bits != 2 && bits != 3)
            || (tile_bytes != 1024 && tile_bytes != 2048)
            || (size_t(n) * k) % tile_bytes) return int(cudaErrorInvalidValue);
    const auto* w = static_cast<const unsigned char*>(weights);
    const auto* s = static_cast<const unsigned char*>(scales);
    const auto* a = static_cast<const __nv_bfloat16*>(x);
    auto* b = static_cast<__nv_bfloat16*>(y);
    const dim3 grid(n / 64, (m + 31) / 32);
    if (bits == 0) fp8_mma<2, false><<<grid, 256, 0, stream>>>(w, s, a, b, n, k, m, tile_bytes);
    else if (bits == 2) fp8_mma<2, true><<<grid, 256, 0, stream>>>(w, s, a, b, n, k, m, tile_bytes);
    else fp8_mma<3, true><<<grid, 256, 0, stream>>>(w, s, a, b, n, k, m, tile_bytes);
    return int(cudaGetLastError());
}

extern "C" int zipserv_fp8_mma_tiled(const void* weights, const void* scales,
                                     const void* x, void* y, int n, int k, int m,
                                     int bits, int tile_bytes, cudaStream_t stream) {
    if (!weights || !scales || !x || !y || uintptr_t(weights) % 8
            || uintptr_t(x) % 2 || uintptr_t(y) % 2 || n < 64 || n > 131072 || n % 64
            || k < 128 || k > 131072 || k % 128 || m < 1 || m > 1024
            || (bits != 0 && bits != 3 && bits != 16 && bits != 32) || (tile_bytes != 1024 && tile_bytes != 2048)
            || (size_t(n) * k) % tile_bytes) return int(cudaErrorInvalidValue);
    const auto* w = static_cast<const unsigned char*>(weights);
    const auto* s = static_cast<const unsigned char*>(scales);
    const auto* a = static_cast<const __nv_bfloat16*>(x);
    auto* b = static_cast<__nv_bfloat16*>(y);
    const dim3 grid(n / 64, (m + 31) / 32);
    if (bits == 0) fp8_mma<3, false, true><<<grid, 256, 0, stream>>>(w, s, a, b, n, k, m, tile_bytes);
    else if (bits == 3) fp8_mma<3, true, true><<<grid, 256, 0, stream>>>(w, s, a, b, n, k, m, tile_bytes);
    else if (bits == 16) fp8_mma<16, true, true><<<grid, 256, 0, stream>>>(w, s, a, b, n, k, m, tile_bytes);
    else fp8_mma<32, true, true><<<grid, 256, 0, stream>>>(w, s, a, b, n, k, m, tile_bytes);
    return int(cudaGetLastError());
}
