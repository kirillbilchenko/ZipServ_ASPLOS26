// Copyright 2026 Kyrylo Bilchenko. Licensed under Apache-2.0.
// Experimental exact byte reconstruction. No model arithmetic is performed.
// The caller validates the CPU container and owns immutable input/output buffers
// through stream completion. This is a component benchmark, not a vLLM plugin.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

template<int Width, bool WarpOffsets>
__global__ void decode_vectors(const unsigned char* encoded, unsigned char* output,
                              size_t meta_offset, size_t data_offset, size_t tiles,
                              int tile_bytes, const int* selected = nullptr,
                              size_t tiles_per_expert = 0, int experts = 0) {
    size_t tile = blockIdx.x;
    if (selected) {
        const int expert = selected[blockIdx.y];
        if (expert < 0 || expert >= experts) return;
        tile += size_t(expert) * tiles_per_expert;
    }
    const auto* offsets = reinterpret_cast<const uint64_t*>(encoded + meta_offset);
    const auto* tags = encoded + meta_offset + (tiles + 1) * 8 + tiles * 4;
    const unsigned char* payload = encoded + data_offset + offsets[tile];
    unsigned char* target = output + tile * size_t(tile_bytes);
    if (tags[tile] == 0) {
        for (int word = threadIdx.x; word < tile_bytes / 4; word += blockDim.x)
            reinterpret_cast<uint32_t*>(target)[word] = reinterpret_cast<const uint32_t*>(payload)[word];
        return;
    }
    const int count = tile_bytes / Width;
    const int prefix_bytes = WarpOffsets ? 2 * (count / 32) : 0;
    const unsigned char* indexes = payload + prefix_bytes;
    const unsigned char* exceptions = indexes + count * (WarpOffsets ? 1 : 2);
    const unsigned char* table = encoded + 64;
    for (int first = 0; first < count; first += blockDim.x) {
        const int vector = first + threadIdx.x;
        const bool active = vector < count;
        unsigned index = 0;
        bool escaped = false;
        if (active) {
            if constexpr (WarpOffsets) {
                index = indexes[vector];
                escaped = index == 255;
            } else {
                index = reinterpret_cast<const uint16_t*>(indexes)[vector];
                escaped = (index & 0x8000u) != 0;
            }
        }
        unsigned mask = 0;
        if constexpr (WarpOffsets) mask = __ballot_sync(0xffffffffu, active && escaped);
        if (active) {
            const unsigned char* source;
            if (escaped) {
                unsigned exception;
                if constexpr (WarpOffsets) {
                    const unsigned lane = threadIdx.x & 31;
                    exception = reinterpret_cast<const uint16_t*>(payload)[vector / 32]
                        + __popc(mask & ((1u << lane) - 1));
                } else {
                    exception = index & 0x7fffu;
                }
                source = exceptions + exception * Width;
            } else {
                source = table + index * Width;
            }
            #pragma unroll
            for (int word = 0; word < Width / 4; ++word)
                reinterpret_cast<uint32_t*>(target + vector * Width)[word]
                    = reinterpret_cast<const uint32_t*>(source)[word];
        }
    }
}

// Routing is consumed on device. One bounded block deduplicates expert IDs;
// no device-to-host count transfer or graph-dependent allocation is needed.
__global__ void select_experts(const int* routing, size_t routes, int experts,
                               int* selected, int slots) {
    __shared__ int active[1024];
    __shared__ unsigned totals[32];
    const int tid = threadIdx.x;
    active[tid] = 0;
    if (tid < slots) selected[tid] = -1;
    __syncthreads();
    for (size_t i = tid; i < routes; i += blockDim.x) {
        const int expert = routing[i];
        // Negative IDs are padding. Out-of-range IDs cannot cause unsafe writes.
        if (expert >= 0 && expert < experts) atomicExch(active + expert, 1);
    }
    __syncthreads();
    const bool present = tid < experts && active[tid];
    const unsigned mask = __ballot_sync(0xffffffffu, present);
    const unsigned warp = tid / 32, lane = tid & 31;
    if (!lane) totals[warp] = __popc(mask);
    __syncthreads();
    unsigned first = 0;
    for (unsigned i = 0; i < warp; ++i) first += totals[i];
    const unsigned position = first + __popc(mask & ((1u << lane) - 1));
    if (present && position < unsigned(slots)) selected[position] = tid;
}

extern "C" int zipserv_dictionary_select_experts(const int* routing, size_t routes,
                                                 int experts, int* selected,
                                                 int slots, cudaStream_t stream) {
    if (!routing || !selected || uintptr_t(routing) % 4 || uintptr_t(selected) % 4
            || !routes || routes > (1u << 24) || experts < 1 || experts > 1024
            || slots != int(routes < size_t(experts) ? routes : size_t(experts)))
        return int(cudaErrorInvalidValue);
    select_experts<<<1, 1024, 0, stream>>>(routing, routes, experts, selected, slots);
    return int(cudaGetLastError());
}

extern "C" int zipserv_dictionary_decode_selected(
        const void* encoded, void* output, size_t packed_bytes, size_t raw_bytes,
        size_t meta_offset, size_t data_offset, int tile_bytes,
        const int* selected, int slots, int experts, cudaStream_t stream) {
    // Deliberately restricted to the measured vector4-warp format. The output
    // retains original expert addressing, so the existing consumer is unchanged.
    if (!encoded || !output || !selected || !raw_bytes
            || (tile_bytes != 1024 && tile_bytes != 2048)
            || experts < 1 || experts > 1024 || slots < 1 || slots > experts
            || raw_bytes % (size_t(experts) * tile_bytes)
            || uintptr_t(encoded) % 4 || uintptr_t(output) % 4 || uintptr_t(selected) % 4
            || meta_offset < 64 || meta_offset % 8 || data_offset % 16
            || data_offset <= meta_offset || data_offset > packed_bytes)
        return int(cudaErrorInvalidValue);
    const size_t tiles = raw_bytes / tile_bytes;
    const size_t per_expert = tiles / experts;
    if (per_expert > 2147483647u || (tiles + 1) * 8 + tiles * 5 > data_offset - meta_offset)
        return int(cudaErrorInvalidValue);
    decode_vectors<4, true><<<dim3(unsigned(per_expert), slots), 256, 0, stream>>>(
        static_cast<const unsigned char*>(encoded), static_cast<unsigned char*>(output),
        meta_offset, data_offset, tiles, tile_bytes, selected, per_expert, experts);
    return int(cudaGetLastError());
}

extern "C" int zipserv_dictionary_decode(const void* encoded, void* output,
                                        size_t packed_bytes, size_t raw_bytes,
                                        size_t meta_offset, size_t data_offset,
                                        int tile_bytes, int vector_bytes, int warp_offsets,
                                        cudaStream_t stream) {
    // This first kernel deliberately accepts complete aligned tiles only.
    if (!encoded || !output || raw_bytes == 0 || (tile_bytes != 1024 && tile_bytes != 2048)
            || raw_bytes % tile_bytes || (vector_bytes != 4 && vector_bytes != 8)
            || (warp_offsets != 0 && warp_offsets != 1) || meta_offset < 64
            || reinterpret_cast<uintptr_t>(encoded) % 4 || reinterpret_cast<uintptr_t>(output) % 4
            || meta_offset % 8 || data_offset % 16
            || data_offset <= meta_offset || data_offset > packed_bytes)
        return int(cudaErrorInvalidValue);
    const size_t tiles = raw_bytes / tile_bytes;
    if (tiles > 2147483647u || (tiles + 1) * 8 + tiles * 5 > data_offset - meta_offset)
        return int(cudaErrorInvalidValue);
    const auto* input = static_cast<const unsigned char*>(encoded);
    auto* target = static_cast<unsigned char*>(output);
    if (vector_bytes == 4 && warp_offsets)
        decode_vectors<4, true><<<unsigned(tiles), 256, 0, stream>>>(input, target, meta_offset, data_offset, tiles, tile_bytes);
    else if (vector_bytes == 4)
        decode_vectors<4, false><<<unsigned(tiles), 256, 0, stream>>>(input, target, meta_offset, data_offset, tiles, tile_bytes);
    else if (warp_offsets)
        decode_vectors<8, true><<<unsigned(tiles), 256, 0, stream>>>(input, target, meta_offset, data_offset, tiles, tile_bytes);
    else
        decode_vectors<8, false><<<unsigned(tiles), 256, 0, stream>>>(input, target, meta_offset, data_offset, tiles, tile_bytes);
    return int(cudaGetLastError());
}
