// Copyright 2026 Kyrylo Bilchenko
// SPDX-License-Identifier: Apache-2.0
#include "../build/L_API.cuh"
#include "../csrc/CodecUtils.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define CUDA(call) do { auto e = (call); if (e != cudaSuccess) \
    throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e)); } while (0)
static std::string case_name;
#define REQUIRE(condition) do { if (!(condition)) throw std::runtime_error(case_name + " line " + std::to_string(__LINE__) + ": " + #condition); } while (0)

template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { CUDA(cudaMalloc(&p, std::max(size_t(1), n) * sizeof(T))); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    void upload(const T* source, size_t n) {
        if (n) CUDA(cudaMemcpy(p, source, n * sizeof(T), cudaMemcpyHostToDevice));
    }
};

struct Packed {
    uint8_t* common = nullptr;
    __nv_bfloat16* rare = nullptr;
    uint64_t *b1 = nullptr, *b2 = nullptr, *b3 = nullptr;
    int *small = nullptr, *medium = nullptr, *global = nullptr;
    int max_common = 0, max_rare = 0, tiles = 0, m, k;
    uint8_t start;
    Packed(std::vector<__nv_bfloat16>& w, int rows, int cols, int forced_base = -1) : m(rows), k(cols) {
        uint64_t counts[256] = {};
        for (auto value : w) ++counts[(__bfloat16_as_ushort(value) >> 7) & 255];
        int base = forced_base >= 0 ? forced_base : zipserv::best_exponent_window(counts);
        int exponents[7];
        for (int i = 0; i < 7; ++i) exponents[i] = base + i;
        start = uint8_t(base - 1);
        tiles = InitBF16MatrixTripleBitmap(w.data(), m, k, 8, 16, 64, 8, 64, 64, exponents,
            &common, &rare, &b1, &b2, &b3, &small, &medium, &global, max_common, max_rare);
        REQUIRE(tiles == (m / 64) * (k / 64));
    }
    ~Packed() {
        free(common); free(rare); free(b1); free(b2); free(b3);
        free(small); free(medium); free(global);
    }
};

struct GpuPacked {
    Packed& h;
    Device<uint8_t> common;
    Device<__nv_bfloat16> rare;
    Device<uint64_t> b1, b2, b3;
    Device<int> medium, global;
    explicit GpuPacked(Packed& p) : h(p),
        common(p.global[2 * p.tiles]), rare(p.global[2 * p.tiles + 1]),
        b1(p.m * p.k / 64), b2(p.m * p.k / 64), b3(p.m * p.k / 64),
        medium(p.tiles * 8), global((p.tiles + 1) * 2) {
        common.upload(p.common, p.global[2 * p.tiles]);
        rare.upload(p.rare, p.global[2 * p.tiles + 1]);
        b1.upload(p.b1, p.m * p.k / 64); b2.upload(p.b2, p.m * p.k / 64);
        b3.upload(p.b3, p.m * p.k / 64);
        medium.upload(p.medium, p.tiles * 8); global.upload(p.global, (p.tiles + 1) * 2);
    }
    cudaError_t mm(cudaStream_t stream, const __nv_bfloat16* x, __nv_bfloat16* y,
                   int n, __nv_bfloat16* scratch, int split) {
        return BF16TripleBitmap_MM_API(stream, common.p, rare.p, b1.p, b2.p, b3.p,
            medium.p, global.p, h.max_common, h.max_rare, h.start, x, y, h.m, n, h.k, scratch, split);
    }
    void check_decode(const std::vector<__nv_bfloat16>& expected, cudaStream_t stream) {
        Device<__nv_bfloat16> output(expected.size());
        CUDA(BF16TripleBitmap_Decompress_API(stream, common.p, rare.p, b1.p, b2.p, b3.p,
            medium.p, global.p, h.max_common, h.max_rare, h.start, output.p, h.m, h.k));
        CUDA(cudaStreamSynchronize(stream));
        std::vector<__nv_bfloat16> actual(expected.size());
        CUDA(cudaMemcpy(actual.data(), output.p, actual.size() * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < actual.size(); ++i)
            REQUIRE(__bfloat16_as_ushort(actual[i]) == __bfloat16_as_ushort(expected[i]));
    }
};

int main() try {
    cudaStream_t stream;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int decoded = 0, gemms = 0;
    std::mt19937 rng(42);
    // Exact decoding includes every BF16 bit pattern and both exponent endpoints.
    for (int mode = 0; mode < 4; ++mode) {
        std::vector<__nv_bfloat16> words(65536);
        for (int i = 0; i < 65536; ++i) {
            uint16_t bits = mode == 0 ? i : mode == 1 ? ((i & 255) | ((i & 256) << 7)) :
                mode == 2 ? (0x7f80 | (i & 127) | ((i & 128) << 8)) : 0;
            words[i] = __ushort_as_bfloat16(bits);
        }
        case_name = "decode mode=" + std::to_string(mode);
        Packed packed(words, 256, 256);
        GpuPacked gpu(packed);
        gpu.check_decode(words, stream); ++decoded;
    }
    // Exact, integer-valued GEMMs exercise changing shared-memory requirements,
    // minimal K, final offset sentinels, every N tile, tails, and Split-K reduction.
    for (int distribution : {0, 1, 0}) {
        for (int k : {64, 128, 512}) {
            const int m = 64;
            std::vector<__nv_bfloat16> w(m * k);
            for (auto& value : w) value = __float2bfloat16(float(int(rng() % 3) - 1));
            case_name = "distribution=" + std::to_string(distribution) + " k=" + std::to_string(k);
            Packed packed(w, m, k, distribution ? 100 : 121); // all rare vs some common
            GpuPacked gpu(packed);
            gpu.check_decode(w, stream); ++decoded;
            for (int n : {1, 3, 7, 8, 9, 16, 17, 31, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113, 127, 128, 129, 193}) {
                std::vector<__nv_bfloat16> x(n * k);
                for (auto& value : x) value = __float2bfloat16(float(int(rng() % 3) - 1));
                Device<__nv_bfloat16> dx(x.size()), dy(n * m + 256), scratch(n * m * 8);
                dx.upload(x.data(), x.size());
                std::vector<__nv_bfloat16> expected(n * m);
                for (int col = 0; col < n; ++col)
                    for (int row = 0; row < m; ++row) {
                        float sum = 0;
                        for (int j = 0; j < k; ++j)
                            sum += __bfloat162float(w[row * k + j]) * __bfloat162float(x[col * k + j]);
                        expected[col * m + row] = __float2bfloat16(sum);
                    }
                for (int split : {1, 2, 4, 8}) {
                    if ((k / 64) % split) continue;
                    case_name = "distribution=" + std::to_string(distribution) + " k=" + std::to_string(k) + " n=" + std::to_string(n) + " split=" + std::to_string(split);
                    CUDA(cudaMemsetAsync(dy.p, 0x55, (n * m + 256) * 2, stream));
                    CUDA(gpu.mm(stream, dx.p, dy.p, n, scratch.p, split));
                    CUDA(cudaStreamSynchronize(stream));
                    std::vector<__nv_bfloat16> actual(n * m + 256);
                    CUDA(cudaMemcpy(actual.data(), dy.p, actual.size() * 2, cudaMemcpyDeviceToHost));
                    for (int i = 0; i < n * m; ++i)
                        REQUIRE(__bfloat16_as_ushort(actual[i]) == __bfloat16_as_ushort(expected[i]));
                    for (int i = n * m; i < n * m + 256; ++i)
                        REQUIRE(__bfloat16_as_ushort(actual[i]) == 0x5555);
                    ++gemms;
                }
                if (n == 9 && k == 128 && distribution == 0) {
                    cudaGraph_t graph; cudaGraphExec_t executable;
                    CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                    CUDA(gpu.mm(stream, dx.p, dy.p, n, scratch.p, 2));
                    CUDA(cudaStreamEndCapture(stream, &graph));
                    CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                    CUDA(cudaGraphLaunch(executable, stream));
                    CUDA(cudaStreamSynchronize(stream));
                    std::vector<__nv_bfloat16> actual(expected.size());
                    CUDA(cudaMemcpy(actual.data(), dy.p, actual.size() * 2, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < actual.size(); ++i)
                        REQUIRE(__bfloat16_as_ushort(actual[i]) == __bfloat16_as_ushort(expected[i]));
                    CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph));
                }
            }
            REQUIRE(gpu.mm(stream, nullptr, nullptr, 1, nullptr, 1) == cudaErrorInvalidValue);
            REQUIRE(gpu.mm(stream, nullptr, nullptr, -1, nullptr, 1) == cudaErrorInvalidValue);
            REQUIRE(gpu.mm(stream, nullptr, nullptr, 1, nullptr, 0) == cudaErrorInvalidValue);
            REQUIRE(gpu.mm(stream, nullptr, nullptr, 1, nullptr, 3) == cudaErrorInvalidValue);
            REQUIRE(gpu.mm(stream, nullptr, nullptr, 0, nullptr, 1) == cudaSuccess);
        }
    }
    CUDA(cudaStreamDestroy(stream));
    std::cout << "PASS: " << decoded << " bit-exact decodes; " << gemms
              << " GEMMs; guards; nondefault stream; CUDA graphs\n";
    return 0;
} catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
