// SPDX-License-Identifier: Apache-2.0
// Copyright MxWave contributors.
// Appended to the hash-checked, externally staged vLLM grouped Marlin host code.
static thread_local std::string mxwave_error;
#define MXWAVE_EXPORT extern "C" __attribute__((visibility("default")))

MXWAVE_EXPORT const char* mxwave_moe_last_error() { return mxwave_error.c_str(); }
MXWAVE_EXPORT int mxwave_moe_compressed() { return MXWAVE_MOE_ASYNC; }
MXWAVE_EXPORT int mxwave_moe_affine_only() { return MXWAVE_MOE_AFFINE_ONLY; }
MXWAVE_EXPORT int mxwave_moe_shared_stage() { return MXWAVE_MOE_SPARSE_STAGE; }
MXWAVE_EXPORT int mxwave_moe_encoding() {
  return MXWAVE_MOE_SPARSE ? 3 : (MXWAVE_MOE_ALIGNED_NIBBLES ? 2 : (MXWAVE_MOE_NIBBLES ? 1 : 0));
}

MXWAVE_EXPORT int mxwave_moe_plan(
    int m, int n, int k, int block, int topk, int experts, int device, int* plan) {
  int sms = 0, shared = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  auto result = MARLIN_NAMESPACE_NAME::determine_exec_config(
      vllm::kBFloat16, vllm::kFE2M1f, vllm::kBFloat16, vllm::kFE8M0fnu,
      m, n, k, experts, topk, (block + 15) / 16, block == 8, 4, 32,
      false, false, false, 4, shared, sms);
  plan[0] = result.tb_cfg.thread_k;
  plan[1] = result.tb_cfg.thread_n;
  plan[2] = result.blocks_per_sm;
  return plan[0] > 0 && plan[1] > 0 ? 0 : 1;
}

MXWAVE_EXPORT int mxwave_moe_run(
    const void* x, const void* weight, void* scales, void* output, void* temporary,
    void* locks, void* sorted_ids, void* expert_ids, void* padded_count, void* routing_weights,
    int m, int n, int k, int experts, int topk, int block, int multiply_weights,
    int device, int tile_k, int tile_n, int blocks_per_sm, uintptr_t stream_pointer) {
  try {
    if (m < 1 || n < 64 || k < 64 || n % 64 || k % 64 || experts < 1 || topk < 1)
      throw std::runtime_error("Invalid grouped Marlin shape");
    if (!x || !weight || !scales || !output || !temporary || !locks || !sorted_ids ||
        !expert_ids || !padded_count || !routing_weights)
      throw std::runtime_error("Null grouped Marlin buffer");
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    MARLIN_NAMESPACE_NAME::marlin_mm(
        x, weight, output, temporary, nullptr, nullptr, scales, nullptr, nullptr,
        sorted_ids, expert_ids, padded_count, routing_weights, block, experts, topk,
        multiply_weights != 0, m, n, k, locks, vllm::kBFloat16, vllm::kFE2M1f,
        vllm::kBFloat16, vllm::kFE8M0fnu, false, false, 32, device,
        reinterpret_cast<cudaStream_t>(stream_pointer), tile_k, tile_n, sms,
        blocks_per_sm, false, true, false);
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return 0;
  } catch (const std::exception& error) {
    mxwave_error = error.what();
    return 1;
  }
}

MXWAVE_EXPORT int mxwave_moe_decode(
    const void* descriptor, void* output, int bytes, uintptr_t stream_pointer) {
  if (bytes <= 0 || bytes % 16 || !descriptor || !output) return 1;
  const int vectors = bytes / 16;
  MARLIN_NAMESPACE_NAME::mxwave_decode_moe_scales<<<
      (vectors + 127) / 128, 128, 0, reinterpret_cast<cudaStream_t>(stream_pointer)>>>(
      reinterpret_cast<const MARLIN_NAMESPACE_NAME::MxWaveMoeScales*>(descriptor),
      reinterpret_cast<int4*>(output), vectors);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

MXWAVE_EXPORT int mxwave_moe_decode_bf16(
    const void* descriptor, void* output, int bytes, uintptr_t stream_pointer) {
  if (bytes <= 0 || bytes % 16 || !descriptor || !output) return 1;
  const int vectors = bytes / 16;
  MARLIN_NAMESPACE_NAME::mxwave_decode_moe_bf16_scales<<<
      (vectors + 127) / 128, 128, 0, reinterpret_cast<cudaStream_t>(stream_pointer)>>>(
      reinterpret_cast<const MARLIN_NAMESPACE_NAME::MxWaveMoeScales*>(descriptor),
      reinterpret_cast<int4*>(output), vectors);
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}
