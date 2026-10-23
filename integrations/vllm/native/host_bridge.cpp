// Copyright 2026 Kyrylo Bilchenko
// SPDX-License-Identifier: Apache-2.0
// Adapted from MxWave's C ABI bridge; CPU packing has no CUDA or Torch ABI dependency.
#include "host_bridge.h"
#include "../../../csrc/HostPacker.h"
#include <cstdlib>
#include <cstring>
#include <limits>

thread_local std::string zipserv_last_error;

extern "C" const char* zipserv_error() { return zipserv_last_error.c_str(); }
extern "C" int zipserv_abi_version() { return 2; }
extern "C" void zipserv_free(Packed* out) {
    for (auto& pointer : out->buffers) { std::free(pointer); pointer = nullptr; }
}

extern "C" int zipserv_compress(const void* weights, int m, int k, Packed* out) {
    if (!out) { zipserv_last_error = "Missing output descriptor"; return -1; }
    std::memset(out, 0, sizeof(*out));
    // The upstream compressor uses signed int offsets and products.
    if (!weights || m <= 0 || k <= 0 || m % 64 || k % 64 ||
        uint64_t(m) * k > uint64_t(std::numeric_limits<int>::max()) - 4096) {
        zipserv_last_error = "BF16 matrix must have positive 64-aligned dimensions and int32 offsets";
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
        zipserv_last_error = error.what();
        return -1;
    }
}
