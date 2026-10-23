# ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

**📄 ASPLOS'26 Paper** | [Paper](https://arxiv.org/abs/2603.17435) | [Code](https://github.com/HPMLL/ZipServ_ASPLOS26)

**Authors**: Ruibo Fan¹, Xiangrui Yu¹, Xinglin Pan¹, Zeyu Li¹, Weile Luo¹, Qiang Wang², Wei Wang³, Xiaowen Chu¹,³  
¹ HKUST(GZ), ² HIT(SZ), ³ HKUST

## Overview

ZipServ combines lossless BF16 weight compression with fused CUDA decompression and matrix multiplication. Weight reconstruction is bit-exact; inference outputs can differ from another backend because GEMM accumulation and Split-K reduction round differently. The performance figures below are the original authors' reported results, not measurements of this fork.

> **Note**: ZipServ is optimized for **GDDR memory-based GPUs** (e.g., RTX 4090, L40S, RTX 5090, RTX 6000 Pro) where memory bandwidth is the primary bottleneck. Performance gains may vary on HBM-based datacenter GPUs.

### Key Advantages

- **Lossless Compression**: Up to 30% weight size reduction with bit-exact weight reconstruction
- **Significant Speedup**: Up to 2.21× kernel-level speedup over cuBLAS, 1.22× average end-to-end speedup over vLLM
- **Hardware Co-design**: Optimized for NVIDIA Tensor Core architectures
- **First Practical System**: First demonstration that lossless compression can simultaneously provide storage savings and inference acceleration

## Core Techniques

### 1. Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE)

Addressing the fundamental mismatch between traditional entropy coding (e.g., Huffman, ANS) and GPU SIMT execution model, ZipServ proposes TCA-TBE encoding:

- **Fixed-width exponent codes**: Three bitmaps select common exponents or exact BF16 exceptions; tile payload lengths vary and are located through offsets
- **Constant-Time Parallel Decoding**: Achieves branch-free parallel decoding through lightweight bitwise operations, eliminating control-flow divergence
- **Tensor Core Alignment**: Hierarchical tiling organization (8×8, 16×64, 64×64) seamlessly aligns with Tensor Core computation patterns

### 2. ZipGEMM Fused Kernel

Addressing the redundant memory access caused by traditional decoupled pipelines (decompress to global memory, then compute), ZipServ designs a fused decompression-GEMM kernel:

- **"Load-Compressed, Compute-Decompressed"**: Compressed weights are loaded directly into registers and decoded on-the-fly
- **Eliminate Intermediate Buffers**: Decompressed results feed directly into Tensor Cores without going through global memory
- **Maximize Compute Intensity**: Overlaps data movement with computation to fully utilize memory bandwidth

## Quick Start

### Prerequisites

- **CUDA Toolkit**: 11.8 or later
- **GPU**: NVIDIA Ampere architecture or newer (Compute Capability ≥ 8.0)
- **Compiler**: g++ with C++17 support

### Build

```bash
# Initialize environment
source Init.sh

# Build core library
cd build && make

# Build benchmarks
cd ../kernel_benchmark && source test_env && make
```

### Run Benchmarks

```bash
# Matrix multiplication benchmark
cd kernel_benchmark
./test_mm M K N SplitK

# Example: ./test_mm 4096 4096 128 1
```

## Local backports and vLLM 0.32.0

This fork includes native correctness fixes and an experimental optional vLLM
plugin adapted from MxWave. See the
[vLLM build and usage instructions](integrations/vllm/README.md).

The adapter compresses eligible raw BF16 linears; an explicit MXFP4 mode can
compress an untied BF16 output head. NVFP4 scale kernels remain in the separate
MxWave project. Full-model serving on vLLM 0.32 is not yet qualified.

```bash
make -C tests cpu
make -C tests memcheck SM=121  # select your actual GPU architecture
```

To inspect a raw little-endian BF16 matrix without a GPU:

```bash
c++ -std=c++17 -O2 tools/profile_bf16.cpp -o tools/profile_bf16
./tools/profile_bf16 /path/to/weights.bf16 M K
```

The profiler reports actual TCA-TBE byte accounting and estimates alternative
formats. It does not implement or benchmark those alternative CUDA decoders.
For Blackwell targets, use a toolkit supporting the selected SM; this fork was
tested with CUDA 13.0 on SM121. Native Makefiles accept `SMS=121` as well.

## CPU packing

The CPU encoder in [csrc/HostPacker.h](csrc/HostPacker.h) builds the same
triple-bitmap format without requiring CUDA headers. It counts tile payloads
first, allocates their exact sizes, and writes independent tiles in parallel.
Exponent classification uses the consecutive-window arithmetic directly. Only
alignment padding is zeroed. The native API remains compatible; the vLLM bridge
also skips the small-tile count buffer that GPU decoding does not use.

Automatic packing uses one worker for matrices below 1,048,576 values and up to
eight workers for larger matrices. C++ callers can select a worker count through
`pack_bf16`; input words must stay immutable until it returns. The Python adapter
shares the native packed allocations with CPU tensors. The allocations remain
alive until the last tensor storage view is released; no extra payload copy is
needed during the handoff.

CPU compatibility checks and benchmarks run with a C++17 compiler on Linux or
macOS; CUDA is needed only for the GPU tests:

```bash
make -C tests cpu sanitize
make -C tests thread-sanitize  # requires compiler ThreadSanitizer support
make -C tests benchmark-host

# Implementation, rows, columns, repetitions, secondary exponent cluster %,
# and optional worker count (0 = automatic). Each command is a fresh process.
./tools/bench_host_packer verify   16384 8192 1 5
./tools/bench_host_packer legacy   16384 8192 1 5
./tools/bench_host_packer serial   16384 8192 1 5
./tools/bench_host_packer parallel 16384 8192 1 5 8
```

The benchmark includes exponent selection and allocation, uses deterministic
synthetic BF16 words, and compares against the frozen encoder from `612dc6c`.
The separate `verify` mode checks every output byte for the chosen matrix size.
`peak_rss_mib` includes the input matrix and all process memory; repeated packing
can increase the allocator's retained memory, so use one repetition for
single-call peak measurements. Timings exclude Python tensor copies, disk I/O,
GPU transfers and inference. CPU tests compare every packed byte and offset to
the old encoder and independently decode the result, including all BF16 bit
patterns, all exponent windows and concurrent calls.

The Python handoff can also be built and measured without CUDA or vLLM. Use a
Python environment containing Torch; the compiler can be selected with `CXX`:

```bash
python integrations/vllm/build_native.py --cpu-only --output build/libzipserv_cpu.so
python tools/bench_python_packer.py --mode copy --library build/libzipserv_cpu.so
python tools/bench_python_packer.py --mode share --library build/libzipserv_cpu.so
python -m pytest -q tests/test_runtime.py tests/test_host_runtime.py
```

Both benchmark modes use the current native encoder and the same synthetic
BF16 input. `copy` reproduces the former Python handoff; `share` uses the new
shared buffers. Each invocation packs once in a fresh process and reports total
process peak RSS, including Torch and the input matrix. This measures CPU
preparation, without disk loading, GPU transfer or inference.

## Citation

If you use ZipServ in your research, please cite our ASPLOS'26 paper:

```bibtex
@inproceedings{zipserv2026,
  title={ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression},
  author={Fan, Ruibo and Yu, Xiangrui and Pan, Xinglin and Li, Zeyu and Luo, Weile and Wang, Qiang and Wang, Wei and Chu, Xiaowen},
  booktitle={Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS'26)},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

**Note**: This is research prototype code. Please conduct thorough testing and validation before production use.

