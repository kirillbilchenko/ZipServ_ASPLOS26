// Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
// Appended only to the hash-checked, isolated vLLM 0.32 Marlin host code.
#define ZFR_EXPORT extern "C" __attribute__((visibility("default")))
static thread_local std::string zfr_error;
// ABI 2 changes the accepted input domain: old bindings must reject it rather
// than accidentally pass a raw tile to a kernel that omits raw-tile handling.
ZFR_EXPORT int zipserv_marlin_fp8_abi() {
  return ZIPSERV_MARLIN_ASYNC && ZIPSERV_MARLIN_ASYNC_COMPACT_ONLY ? 2 : 1;
}
ZFR_EXPORT int zipserv_marlin_fp8_compact_only() {
  return ZIPSERV_MARLIN_ASYNC && ZIPSERV_MARLIN_ASYNC_COMPACT_ONLY;
}
ZFR_EXPORT int zipserv_marlin_fp8_compressed() { return ZIPSERV_MARLIN_RANGE16 || ZIPSERV_MARLIN_ASYNC; }
ZFR_EXPORT int zipserv_marlin_fp8_codec() { return ZIPSERV_MARLIN_ASYNC ? (ZIPSERV_MARLIN_ASYNC_BYTEPLANE ? 3 : 2) : ZIPSERV_MARLIN_RANGE16; }
ZFR_EXPORT const char* zipserv_marlin_fp8_last_error() { return zfr_error.c_str(); }

ZFR_EXPORT int zipserv_marlin_fp8_run(
    const void* x, const void* weight, void* scales, void* y, void* temporary,
    void* locks, int m, int n, int k, int device, int atomic_add,
    uintptr_t stream_pointer) {
  try {
    zfr_error.clear();
    if (!x || !weight || !scales || !y || !temporary || !locks ||
        m < 1 || m > 128 || n < 64 || k < 64 || n % 64 || k % 64 ||
        (n % 128 && k % 128) ||
        int64_t(n) * k > INT32_MAX || (atomic_add != 0 && atomic_add != 1))
      throw std::runtime_error("Invalid bounded BF16/MXFP8 experiment arguments");
    if ((uintptr_t(x) | uintptr_t(weight) | uintptr_t(scales) | uintptr_t(y) |
         uintptr_t(temporary) | uintptr_t(locks)) % 16)
      throw std::runtime_error("Buffers must be 16-byte aligned");
    int current = -1, sms = 0, major = 0;
    if (cudaGetDevice(&current) != cudaSuccess || current != device ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        major < 8)
      throw std::runtime_error("Require the current CUDA device with SM80 or newer");
    // Unused optional pointers are valid buffers, including a_s which the
    // upstream host increments even for BF16 inputs. They are never read.
    MARLIN_NAMESPACE_NAME::marlin_mm(
        x, weight, y, temporary, temporary, temporary, scales, temporary,
        temporary, m, n, k, k, locks, vllm::kBFloat16, vllm::kFE4M3fn,
        vllm::kBFloat16, vllm::kFE8M0fnu, false, false, k / 32, 32,
        device, reinterpret_cast<cudaStream_t>(stream_pointer), -1, -1, sms,
        atomic_add != 0, true, false);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return 0;
  } catch (const std::exception& error) {
    zfr_error = error.what();
    return 1;
  }
}

#if ZIPSERV_MARLIN_RANGE16
// Exercise the same shared-memory copy as the GEMM, including the async path.
__global__ void zipserv_marlin_fp8_decode_kernel(
    const uint8_t* encoded, int4* output, int vectors) {
  __shared__ int4 tile[128];
  const int i = blockIdx.x * 128 + threadIdx.x;
  if (i < vectors)
    MARLIN_NAMESPACE_NAME::zipserv_range16_copy(tile + threadIdx.x, encoded, i, encoded[4]);
  MARLIN_NAMESPACE_NAME::cp_async_fence();
  MARLIN_NAMESPACE_NAME::cp_async_wait<0>();
  __syncthreads();
  if (i < vectors) output[i] = tile[threadIdx.x];
}
ZFR_EXPORT int zipserv_marlin_fp8_decode(
    const void* encoded, void* output, int raw_bytes, uintptr_t stream_pointer) {
  if (!encoded || !output || raw_bytes <= 0 || raw_bytes % 2048 ||
      (uintptr_t(encoded) | uintptr_t(output)) % 16) return 1;
  const int vectors = raw_bytes / 16;
  zipserv_marlin_fp8_decode_kernel<<<(vectors + 127) / 128, 128, 0,
      reinterpret_cast<cudaStream_t>(stream_pointer)>>>(
      static_cast<const uint8_t*>(encoded), static_cast<int4*>(output), vectors);
  return int(cudaGetLastError());
}
#endif

#if ZIPSERV_MARLIN_ASYNC
__global__ void zipserv_marlin_fp8_async_decode_kernel(
    const uint8_t* encoded, int4* output) {
  __shared__ int4 tile[64];
  uint32_t* flag_ptr = nullptr;
#if !ZIPSERV_MARLIN_ASYNC_COMPACT_ONLY
  __shared__ uint32_t flag;
  flag_ptr = &flag;
#endif
#if ZIPSERV_MARLIN_ASYNC_PRODUCER
  MARLIN_NAMESPACE_NAME::zipserv_async_copy_tile_lane(tile, flag_ptr, threadIdx.x, encoded,
                                                     blockIdx.x * 64 + threadIdx.x);
#else
  MARLIN_NAMESPACE_NAME::zipserv_async_copy(tile, flag_ptr, threadIdx.x, encoded,
                                           blockIdx.x * 64 + threadIdx.x);
#endif
  MARLIN_NAMESPACE_NAME::cp_async_fence();
  MARLIN_NAMESPACE_NAME::cp_async_wait<0>();
#if ZIPSERV_MARLIN_ASYNC_PRODUCER
  MARLIN_NAMESPACE_NAME::zipserv_async_expand_tile(tile, flag_ptr, 0, encoded[4]);
  __syncthreads();
  output[blockIdx.x * 64 + threadIdx.x * 2] = tile[threadIdx.x * 2];
  output[blockIdx.x * 64 + threadIdx.x * 2 + 1] = tile[threadIdx.x * 2 + 1];
#else
  __syncthreads();
  if (threadIdx.x < 32) {
    MARLIN_NAMESPACE_NAME::I4 first, second;
    MARLIN_NAMESPACE_NAME::zipserv_async_read(first, second, tile, flag_ptr, threadIdx.x * 2, encoded[4]);
    output[blockIdx.x * 64 + threadIdx.x * 2] = *reinterpret_cast<int4*>(&first);
    output[blockIdx.x * 64 + threadIdx.x * 2 + 1] = *reinterpret_cast<int4*>(&second);
  }
#endif
}
ZFR_EXPORT int zipserv_marlin_fp8_decode(
    const void* encoded, void* output, int raw_bytes, uintptr_t stream_pointer) {
  if (!encoded || !output || raw_bytes <= 0 || raw_bytes % 1024 ||
      (uintptr_t(encoded) | uintptr_t(output)) % 16) return 1;
  constexpr int threads = ZIPSERV_MARLIN_ASYNC_PRODUCER ? 32 : 64;
  zipserv_marlin_fp8_async_decode_kernel<<<raw_bytes / 1024, threads, 0,
      reinterpret_cast<cudaStream_t>(stream_pointer)>>>(
      static_cast<const uint8_t*>(encoded), static_cast<int4*>(output));
  return int(cudaGetLastError());
}
#endif
