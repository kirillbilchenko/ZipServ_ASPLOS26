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

