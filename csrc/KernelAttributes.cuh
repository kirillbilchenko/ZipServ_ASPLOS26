// Copyright 2026 Kyrylo Bilchenko
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <map>
#include <mutex>

// The experiment targets persistent CUDA primary contexts, as used by vLLM.
// A cudaDeviceReset/context teardown requires restarting the worker; cached
// attributes cannot be reused across a reset. Actual launch bytes still control
// occupancy. Failed attribute calls are never cached.
template<typename Kernel>
cudaError_t zipserv_set_shmem(Kernel kernel, int required, int maximum) {
#ifdef ZIPSERV_EXPERIMENT_SHMEM_ONCE
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    static std::mutex mutex;
    static std::map<std::pair<int, const void*>, bool> configured;
    const auto key = std::make_pair(device, reinterpret_cast<const void*>(kernel));
    std::lock_guard<std::mutex> lock(mutex);
    if (configured.find(key) != configured.end()) return cudaSuccess;
    if (required > maximum) return cudaErrorInvalidValue;
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maximum);
    if (status == cudaSuccess) configured.emplace(key, true);
    return status;
#else
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, required);
#endif
}
