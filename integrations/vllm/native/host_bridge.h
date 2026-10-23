// Copyright 2026 Kyrylo Bilchenko
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <string>

// ABI v2, layout v1. Lengths are bytes; each buffer is independently allocated.
// The descriptor owns all seven allocations until zipserv_free() is called.
struct Packed {
    void* buffers[7];
    uint64_t lengths[7];
    int32_t m, k, max_common, max_rare, start_exp;
};
extern thread_local std::string zipserv_last_error;
extern "C" {
const char* zipserv_error();
int zipserv_abi_version();
void zipserv_free(Packed* out);
int zipserv_compress(const void* weights, int m, int k, Packed* out);
}
