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

## Local backports and optional vLLM integration

The `docs/` paths mentioned below identify separately retained experiment notes
and evidence, not files shipped in this source checkout. The relevant findings
and limitations are included here and in the tracked integration README.

This fork includes native correctness fixes and an experimental optional vLLM
plugin adapted from MxWave. See the
[vLLM build and usage instructions](integrations/vllm/README.md).

Reviewed source profiles cover vLLM 0.31.0 and 0.32.0; GPU serving results are
specific to 0.32.0. On two GB10s with stock vLLM and DFlash2, additional packing
saved **10.260 GiB for CT MXFP4** and **10.332 GiB for NVIDIA NVFP4**, each against
the same checkpoint without ZipServ packing. NVIDIA single-request changes
ranged from −3.0% to +4.1%; concurrent throughput ranged from −16.4% to +3.3%.
These are opt-in memory savings with measured trade-offs, not a general speedup
or a quality-equivalence result. See the
[matched results and controls](integrations/vllm/README.md#stock-vllm--dflash2-results-2026-10-26).

For the GLM CT experiments, start with
[what the current CT results mean](integrations/vllm/README.md#what-the-current-ct-results-mean).
Existing CT saves 9.725 GiB against its same-runtime optional-weight-off control;
the new narrow BF16 add-on saves another 0.404 GiB. The modern plugin's 10.283 GiB
result uses a different baseline and is not an additional saving over CT.

The adapter compresses eligible raw BF16 linears; an explicit MXFP4 mode can
compress an untied BF16 output head. The separate opt-in
[`zipserv_mxfp4_scales` entry point](integrations/vllm/README.md#standalone-mxfp4-expert-scale-plugin)
bundles lossless MXFP4 expert-scale storage and its fused Marlin consumer.
It has on/off flags, conservative stock fallback and per-worker coverage and
memory audits. A separate opt-in
[`zipserv_nvfp4_scales` consumer](integrations/vllm/README.md#experimental-nvfp4-scale-controls)
supports qualified NVFP4 scales on the BF16 Marlin path. See each integration's
validation scope before applying it to a full serving stack.

The experimental
[`zipserv_mixed_weights` entry point](integrations/vllm/README.md#experimental-mixed-bf16--fp8--fp4-storage)
adds explicit BF16 and MXFP8 weight targets alongside those FP4 scales.
`tools/serve_compressed.py` provides independent on/off switches, and
`tools/compare_mixed_model.py` separates per-type payload savings from measured
whole-model memory, speed and quality. The mixed path requires its reviewed
vLLM API and model-consumer contracts.
The [measured CT profile](integrations/vllm/README.md#measured-ct-profile-2026-10-24)
reports completed two-worker memory, single/concurrent request speed and full
GSM8K results, including the observed accuracy difference. Exact weight storage
does not guarantee identical floating-point GEMM results.

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

## FP4/FP8 compression experiments

The CPU-only screen compares lossless encodings of existing FP4 code bytes,
FP8 exponents, and byte-sized scales. It includes palette/exception coding,
canonical Huffman, one- and 32-stream rANS, and dictionaries of exact 4/8/16/32-byte
vectors. Dictionary variants trade compact byte indexes for two-byte records
with direct exception addresses, avoiding a runtime exception-prefix scan.
The `-warp` variants retain byte indexes and add one two-byte exception offset
per 32 vectors, enabling warp-local exception addressing without block barriers.
FP8 sign/mantissa bits are preserved. Every candidate is serialized
and independently decoded; reported sizes include codebooks, headers, 64-bit
tile offsets, tags, stream states/lengths, raw fallback and alignment.

```bash
python3 tools/screen_low_precision.py \
  --source /path/to/checkpoint --tensor layers.0.mlp.down_proj.weight \
  --kind fp8-e4m3 --goal balanced --report /tmp/fp8-screen.json

python3 tools/screen_low_precision.py \
  --source /path/to/runtime-scales.bin --kind scale-u8 --layout runtime \
  --goal memory --max-slowdown-pct 5 --min-saving-pct 10 \
  --bandwidth-gbps 100,200 --report /tmp/scale-screen.json

python3 -m unittest discover -s tests -p test_low_precision.py -v
```

`--kind` explicitly declares the physical byte representation; it is not
inferred from tensor names. Safetensors dtypes and offsets are checked. Packed
FP4 inputs must have two codes per byte. The tool does not quantize weights or
reorder checkpoint bytes into a native kernel layout. `--layout runtime`
requires an already captured runtime byte stream. The default sample is four
disjoint 16-KiB windows. These sample results must not be extrapolated into a
whole-model saving without a full audit, especially when combining codes and
scales of very different sizes.

The goals select **measured candidates**, with configurable defaults:

| Goal | Maximum slowdown | Minimum resident saving | Selection |
|---|---:|---:|---|
| `speed` | 0% | 0% | Lowest latency within limits |
| `balanced` | 3% | 5% | Smallest resident size within limits |
| `memory` | 10% | 1% | Smallest resident size within limits |

`--max-slowdown-pct` and `--min-saving-pct` override those policy limits. They
are **not performance predictions or guarantees**. Without `--timings`, the
report identifies the smallest CPU encoding but retains `raw` as the runtime
recommendation. These flags are not vLLM serving switches. The GPU dictionary
experiment reconstructs existing packed scale buffers. An opt-in eager vLLM
scale-cache adapter is described below.

Optional `--bandwidth-gbps` values produce an analytical decoding-time budget:
bytes avoided divided by assumed effective bandwidth. This assumes each stored
byte is read once and excludes additional lookup traffic, scratch, launch cost
and occupancy changes. It does not predict GPU latency. CPU byte accounting
also excludes GPU scratch, decoded caches and allocator rounding.
A vector dictionary's coverage measures repeated bytes, not temporal cache hits.

`--timings` accepts JSON with `schema_version: 1`, the report's `sample_sha256`
and `layout`, nonempty `hardware` and `workload`, `scope` (`kernel` or
`end_to_end`), and a `baseline` with positive `resident_bytes` and at least five
positive `samples_us`. Its `candidates` maps candidate IDs to records with
their exact `container_sha256`, paired `samples_us`, measured `resident_bytes`
including all decoder buffers/caches, `exact_gpu_roundtrip: true`, and
`output_check_passed: true` after GPU reconstruction and matched output checks.
Only matching evidence is eligible. The gate uses the
95th percentile of paired latency changes; that sample percentile is not a
statistical confidence bound. A kernel result never qualifies whole-model
latency or quality. Report same-workload performance separately from gains
made possible by larger batches.

This screen adapts MxWave's Apache-2.0 CPU rANS reference; source revision and
changes are recorded in [the provenance file](tools/compression-provenance.json).

The experimental CUDA decoder supports direct and warp-offset 4/8-byte vector
dictionaries in complete 1024/2048-byte tiles. It preserves the existing byte
representation, including already packed nibble scales. The independent GPU
checks require CUDA-enabled Torch and NumPy; choose your actual architecture:

```bash
nvcc -std=c++17 -O3 -shared -Xcompiler=-fPIC -arch=sm_121 \
  tools/scale_dictionary.cu -o build/libzipserv_scale_dictionary.so
python tools/bench_scale_dictionary.py \
  --library build/libzipserv_scale_dictionary.so --checks-only \
  --report /tmp/dictionary-gpu-check.json
```

The optional real-weight GEMM benchmark additionally requires vLLM, an MXFP4
source checkpoint, and the existing MxWave aligned-nibble native library and
`moe_runtime.py` bridge. Supply `--source`, `--moe-library`, and `--moe-support`
(the directory containing that bridge), without `--checks-only`. Their hashes
are recorded; the benchmark does not install or rebuild the external consumer.
It compares the unchanged matrix kernel with a predecoded hit and a forced
decode miss, under warm and pressured caches, using eight source experts and
TP2 rank-zero shapes on one GPU. Its default GPU tensor-allocation cap is
512 MiB. This is not a whole-model or tensor-parallel serving benchmark.
All GEMM arms share the same input, scales, output and scratch addresses.
Timing events are captured inside the CUDA graphs to exclude CPU submission
gaps. The predecoded arm repeats the baseline as a timing-noise control;
paired medians and sample p95 changes are reported for both. Cache pressure
uses a 64-MiB write before timing and does not guarantee complete cache eviction.

A decoded cache uses memory. The reports distinguish compressed storage,
decoded buffers and the comparison process's peak allocation. The helper
`cache_reuse_hint` estimates the hit rate needed to meet a latency limit from
measured hit/miss costs. It does not predict a routing distribution, implement
a replacement policy, or account for that policy's additional overhead.

### FP8 GPU prototypes and routed FP4-scale decoding

`tools/fp8_palette.py` and `tools/fp8_range.py` implement experimental exact
E4M3 containers. The palette format retains exceptional exponents. The range
format uses seven bits only for complete 16/32-value blocks whose exponents
fit an eight-value window; other blocks remain raw FP8. Both account for
headers, indexes, masks, padding and raw fallback. They do not change model
quantization or provide a serving/checkpoint format.

```bash
nvcc -std=c++17 -O3 -shared -Xcompiler=-fPIC -arch=sm_121 \
  tools/fp8_palette.cu -o build/libzipserv_fp8_experiment.so
python tools/bench_fp8_palette.py \
  --library build/libzipserv_fp8_experiment.so --checks-only \
  --report /tmp/fp8-gpu-check.json
python tools/bench_fp8_palette.py \
  --library build/libzipserv_fp8_experiment.so \
  --source /path/to/ct-checkpoint --report /tmp/fp8-real.json
python3 -m unittest discover -s tests -p test_fp8_palette.py -v
```

The real-weight harness requires CUDA-enabled Torch, NumPy, safetensors and
vLLM's MXFP8 Marlin helper. It validates declared group-32 E4M3/E8M0 tensors,
reads complete TP2 shards and records their hashes. `--cases` selects
`layer:rank:gate_up|down` fixtures; `--tokens` controls activation rows.
The default covers 60 MiB of actual CT weights, using a 1.5 GiB CUDA allocation
cap. It compares unchanged Marlin, decode-then-Marlin, a small scalar fused
probe, synchronous Tensor Core probes and identical uncompressed controls.
The Tensor Core variants use NVIDIA's
[WMMA API](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#wmma)
and decode directly into shared memory. A 64x128 weight-tile order tests
contiguous block loads. These are probes, not a replacement for Marlin's
optimized pipeline. Only SM121 was measured.

The measured palette format saves 7.5–9.0% of these FP8 payloads. It fails the
inference-speed gate. In the 48 MiB dense-shard, eight-token case, tile-ordered
fusion reduces measured L2 traffic from 156 MB for separate decoding to 70 MB,
but stock Marlin uses 59 MB. Latencies are approximately 0.72 ms for that fused
probe and 0.22 ms for stock. The fused probe executes about 14 times as many
instructions. Traffic counters come from a separate Nsight application-replay
profile; timing comes from paired unprofiled CUDA graphs. This is not a
whole-model saving or speedup. `tools/profile_fp8_traffic.py --arm` selects
stock/separate/fused/raw controls for an external profiler.

The cheaper range-block follow-up also remains experimental. Actual serialized
16-value containers save 6.7–8.9%; 32-value containers save 4.5–7.6% on the same
three shards. For the dense shard at one token, the 16-value scalar fused path
takes 0.299 ms versus 0.321 ms for the palette and 0.227 ms for stock Marlin.
The format removes per-value exception streams while preserving every FP8 bit,
but its gains do not close the serving-performance gap. The reports retain all
other shapes, identical-arithmetic controls and the unchanged stock baseline.

### FP8 decoding inside Marlin (experimental; fails the speed gate)

`tools/build_fp8_marlin.py` stages hash-pinned Apache-2.0 sources from vLLM
0.32.0 and by default builds two isolated libraries: unchanged BF16/MXFP8 Marlin and an
optional ZFR1 16-value decoder in its weight-loading step. It preserves
Marlin's matrix arithmetic, scale handling, scheduling and reduction. One
block lookup reconstructs the 16 bytes of an original Marlin weight load;
the candidate does not allocate a full decoded weight buffer. Raw tiles keep
the original asynchronous copy. Compressed blocks use synchronous loads and
shared-memory stores. The SM121 experiment preserves exact outputs and saves
6.74–8.89% of native weight bytes, but increases latency in all 12 measured
layer/batch combinations. Keep this experimental path disabled for serving.

The format and integer decoder are reusable across backends. This first fused
integration consumes Marlin's prepared byte layout. A CUTLASS, Triton or other
kernel would need its own loading integration and validation; installing this
experiment does not enable compression in all vLLM execution paths.

The explicit Python binding `tools/fp8_marlin.py` accepts
`compression="off"|"on"|"auto"`, defaulting to `off`. `auto` retains raw bytes
unless the complete compressed container clears `min_saving_pct`; this is a
storage filter, not a latency estimate. The current scope is BF16 activations,
E4M3 weights with group-32 E8M0 scales, M=1..128, and already prepared native
Marlin tile dimensions: N divisible by 64/K by 128, or N by 128/K by 64.
Weights, scales and bias are not requantized; this experimental ABI has no
bias argument. It does not install a serving hook or patch installed vLLM.

Local staging and tests require Python, NumPy and a C++ compiler. CUDA build
and GPU checks require Linux, CUDA/NVCC, Torch and vLLM 0.32.0.
Direct NVCC compilation is the default; `--build-tool cmake` uses CMake/Ninja.
The default target is Spark's SM121; `--arch` selects another supported
numeric CUDA target, whose behavior remains separately unvalidated.

```bash
python3 -m unittest discover -s tests -p test_fp8_marlin.py -v
python3 tools/build_fp8_marlin.py --fetch \
  --source /tmp/vllm-032-fp8-source --output /tmp/zipserv-fp8-marlin

# On the CUDA host, using its installed Torch headers:
python3 tools/build_fp8_marlin.py \
  --source /tmp/vllm-032-fp8-source --output /tmp/zipserv-fp8-marlin \
  --arch 121 --build --torch-include /path/to/site-packages/torch/include
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python3 tools/bench_fp8_marlin.py --checks-only \
  --raw-library /tmp/zipserv-fp8-marlin/build/libzipserv_marlin_fp8_raw.so \
  --range-library /tmp/zipserv-fp8-marlin/build/libzipserv_marlin_fp8_range16.so \
  --report /tmp/marlin-fp8-checks.json
python3 tools/bench_fp8_marlin.py --source /path/to/ct-checkpoint \
  --raw-library /tmp/zipserv-fp8-marlin/build/libzipserv_marlin_fp8_raw.so \
  --range-library /tmp/zipserv-fp8-marlin/build/libzipserv_marlin_fp8_range16.so \
  --report /tmp/marlin-fp8-real.json
```

Before timing, also run the checks command under Compute Sanitizer's
`--tool racecheck` and `--tool synccheck` to check the mixed synchronous and
asynchronous shared-memory path. The prepared suite contains six byte-decoder
cases and 60 GEMM shapes, with guard regions, non-default streams and CUDA
graphs. All 66 cases pass on Spark under memory, race and synchronization
checking, with zero reported errors or hazards. Native CPU tests pass with
AddressSanitizer and UndefinedBehaviorSanitizer, including exhaustive
four-value exponent combinations at all nine legal window bases.

The real-CT benchmark defaults to the same three FP8 shards and M=1,8,32,128
as the earlier experiment. It compares stock vLLM, an isolated raw Marlin
control and the integrated decoder using matched inputs and paired CUDA-graph
timings. The controlled arms disable atomic reduction for byte-exact checks;
an additional stock arm uses vLLM's default reduction policy. Reports separate
serialized weight savings, unchanged scales and ordinary Marlin reduction
workspace. Raw and compressed controls coexist in the benchmark process, so
its peak allocation is not deployment memory use. No weight-sized decode
scratch is retained during timing. The tensor-allocation cap is 1.5 GiB.

For the 48 MiB dense shard at M=8, matched warm medians are 205.35 microseconds
for stock, 205.04 for the isolated raw build, and 460.81 for integrated ZFR1.
The candidate stores 45,859,248 bytes instead of 50,331,648. A separate matched
Nsight Compute profile shows L2 traffic falling from 58.62 MB to 55.46 MB,
while SM instructions rise from 6.66 million to 20.15 million. Registers fall
from 116 to 96 per thread and measured active-warp occupancy stays about
16.67%. Instruction and load-dependency cost are the next optimization targets. The profiles
do not isolate the contribution of each individual decoder instruction.

For external Nsight application replay, pass
`--profile-arm stock|raw|range16 --cases 0:0:gate_up --tokens 8`. Each arm marks
one operation with `zipserv_fp8_marlin` and records source/input hashes.
Use matched profiler settings and compare L2 traffic, instructions, register
use and occupancy; use the unprofiled paired runs for latency. Passing these
component checks would still require a full CT serving and quality evaluation
before enabling the path in production. Source pins and licensing are in
`tools/marlin_fp8_sources.json` and `tools/MARLIN_NOTICE`.

For an Nsight Systems timeline, the harness keeps its named range open until
GPU work completes. Torch's unregistered NVTX strings need the setting below,
as described in NVIDIA's [capture guide](https://docs.nvidia.com/nsight-systems/UserGuide/):

```bash
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 nsys profile \
  --sample=none --cpuctxsw=none --trace=cuda,nvtx,osrt \
  --capture-range=nvtx --nvtx-capture=zipserv_fp8_marlin \
  --capture-range-end=stop --output /tmp/marlin-range-trace \
  python3 tools/bench_fp8_marlin.py --profile-arm range16 \
  --source /path/to/ct-checkpoint --cases 0:0:gate_up --tokens 8 \
  --raw-library /tmp/zipserv-fp8-marlin/build/libzipserv_marlin_fp8_raw.so \
  --range-library /tmp/zipserv-fp8-marlin/build/libzipserv_marlin_fp8_range16.so \
  --report /tmp/marlin-range-trace.json
```

### Asynchronous FP8 follow-up: faster decoding, still below the speed gate

`--variants raw range16 async` adds the lossless ZFA1 experiment. It stages
contiguous compressed bytes with `cp.async`, then reconstructs fragments from
shared memory. Each 1024-byte tile stores seven bits per value plus sparse
corrections for the omitted exponent bit, or falls back to raw bytes. Headers,
indices, exceptions and padding are included in the reported memory savings.
The original weights and scales are preserved exactly.

On the same 48 MiB CT dense shard at M=8, one paired Spark run measured:

| Representation | Weight memory saved | Warm median |
|---|---:|---:|
| Stock vLLM 0.32 Marlin | 0% | 204.94 us |
| Previous synchronous ZFR1 decoder | 8.89% | 463.46 us |
| Asynchronous ZFA1, 16-byte alignment | 7.34% | 228.59 us |

ZFA1 roughly halves the previous decoder's latency, but its paired median is
still **11.48% slower than stock**. Including unchanged scales, this shard saves
7.12%; these percentages are not whole-model savings. At M=8 the two smaller
shards save 7.33% and 5.23% of weight bytes, but run about 91% and 76% slower.
Larger batches also regress. No serving hook is enabled.

Additional experiments compare 32-byte tile alignment and direct conversion
from packed codes to BF16 fragments. Alignment reduces dense-layer L2 traffic
from 79.05 MB to 56.19 MB, but lowers weight savings to 6.20% and leaves the
paired warm penalty near 11.5%. Direct conversion reduces SM instructions from
21.12 million to 19.35 million, but also fails the speed gate. The best trade-off
depends on shape; none of the 60 tested configuration/shape combinations meets
a 5% warm-median slowdown limit. Keep compression disabled for serving.

The options remain explicit experiments:

```bash
# Select the async decoder; default is --async-decode bytes.
python3 tools/build_fp8_marlin.py \
  --source /tmp/vllm-032-fp8-source --output /tmp/zipserv-fp8-async \
  --variants raw range16 async --async-decode bytes \
  --build --torch-include /path/to/site-packages/torch/include

python3 tools/bench_fp8_marlin.py --source /path/to/ct-checkpoint \
  --raw-library /tmp/zipserv-fp8-async/build/libzipserv_marlin_fp8_raw.so \
  --range-library /tmp/zipserv-fp8-async/build/libzipserv_marlin_fp8_range16.so \
  --async-library /tmp/zipserv-fp8-async/build/libzipserv_marlin_fp8_async.so \
  --async-alignment 16 --report /tmp/async-real.json
```

Use a separate build output with `--async-decode direct` for the direct BF16
conversion, or `--async-alignment 32` for the padding trade-off. The Python
owner accepts `codec="async"` and `async_alignment=16|32` alongside its existing
`compression="off"|"on"|"auto"` flag. `auto` filters storage savings only; it does
not promise stock speed. `force_raw=True` is an ablation control that retains
format overhead, not a compression option for deployment. Neither decoder
retains a full decoded weight buffer during GEMM.

The 78-case GPU suite passes memory, race and synchronization checks for the
byte decoder at both alignments and the final direct decoder at alignment 16.
Alignment 32 with the direct decoder also passes memory checks. All 60 real
configuration/shape results are byte-exact to the stock control; 28 local tests
pass, including native ASan/UBSan tests. Matched Nsight timelines contain one
GEMM kernel and no memcpy/memset in the measured range. Sources, reports and
profiles are preserved in the local research evidence under
`docs/evidence/fp8-async-gpu-20261024/`; that folder is excluded from commits.

CPU packing is outside the measured inference path. A Rust host packer could
improve preparation, but it would not remove these GPU instruction and memory
costs. Pipeline/occupancy tuning needs a new comparison against an equally
tuned raw control before pursuing denser encoding.

The producer-stage follow-up is selected with `--async-decode producer` in a
separate build directory. One warp expands each compressed tile in place before
the existing CTA barrier; Marlin then uses its original fragment loads and MMA.
Use `--reference-async-library /path/to/previous/async.so` in the benchmark to
compare decoders using the same immutable compressed GPU buffer.

On the 48 MiB shard at M=8, the final paired run measured stock **206.99 us**, the
previous decoder **229.88 us**, and producer expansion **228.29 us**, retaining
the same **7.34% weight saving**. The producer's paired median slowdown against
stock was **10.43%**. At M=128 it improved from 842.68 to 659.18 us, but stock took
437.46 us. Smaller layers regressed substantially; none of the final 24 producer
shape/alignment combinations met the follow-up's stricter 2% latency gate.
This remains an isolated experiment, with no CT serving integration enabled.

The final copy ordering reduces L2 traffic from the first producer revision's
101.71 MB to 78.94 MB, but still executes 21.15 million SM instructions versus
stock's 6.66 million. Both revisions pass 78 GPU correctness cases under memory,
race and synchronization checking. The final trace contains one GEMM and no
memcpy/memset in its measured range. Less traffic alone did not meet the gate.

`tools/audit_fp8_inventory.py --source /path/to/ct-checkpoint --report /tmp/fp8-inventory.json`
counts compatible checkpoint bytes without loading payloads. The CT source has
135 target-model MLP tensors totaling **1.40625 GiB across TP2**, excluding its
stored next-token prediction layer. An assumed 6–7.3% saving would save only
**86.4–105.1 MiB total** for this family; this is an estimate, not a measured
full-model allocation delta. Runtime BF16-to-FP8 conversions need separate
coverage. Detailed results are in the local research record
`docs/HUFFMAN_RANS.md` and `docs/evidence/fp8-producer-gpu-20261024/`.

Two further build-time experiments are available; both remain opt-in:

- `--async-exceptions uniform` converges exception handling before unpacking.
  With `--async-decode producer`, it reduces the measured dense M=8 kernel's SM
  instruction count from 21.15 million to 16.85 million with identical encoded
  bytes. Several smaller-layer cases improve roughly 7–9% over the previous
  producer, but the stock latency gate still fails.
- `--exact-scale-cache` sizes this restricted E8M0 ABI's scale pipeline at one
  byte per scale, enabling a wider tile. Apply it to raw controls too. On the
  dense M=128 case, compressed latency falls to 503.33 us versus 679.12 us for
  the previous producer; the equally tuned raw kernel takes 425.98 us. The
  compressed path remains 18.53% slower than that raw control in the paired
  comparison. Changed tiling also produces small BF16 rounding differences
  against stock on two real cases, so the strict stock-exact test stops.

`tools/check_fp8_dispatch.py` diagnoses that distinction without relaxing the
original gate: all 12 real cases reconstruct exact weight bytes and produce
outputs identical to the equally tuned raw kernel. This is not a model-quality
qualification. The combined variant passes all 78 synthetic cases under
memory, race and synchronization checking. Defaults and serving remain
unchanged; details and the retained failed run are under
`docs/evidence/fp8-uniform-gpu-20261024/`.

A separate byte-aligned format, **ZFB1**, is available with build flags
`--async-decode producer --async-exceptions uniform --async-layout byteplane`
and benchmark flag `--async-codec byteplane`. Its library advertises codec ID 3;
ZFA1 stays codec ID 2. Choose a fresh build directory. Use
`--reference-async-library /path/to/previous/async.so --reference-async-codec async`
for a paired comparison; different formats encode separate buffers from the same
native weights. The default remains ZFA1 packed-seven-bit codes.

ZFB1 keeps seven of every eight words byte-aligned and stores the eighth word's
bits in their unused bit-6 positions. It preserves every source bit and the
previous format's container sizes. With conservative scale sizing, all 24 real
CT shape/alignment cases reconstruct exact GPU weight bytes and match stock
GEMM outputs. At 16-byte alignment, dense M=128 warm medians improve from
651.53 to 519.79 us; stock is 440.43 us. Dense M=8 remains 227.36 us versus
206.18 us stock, retaining 7.343% weight savings. None of the 24 cases meets the
2% latency limit in either warm or cache-pressure measurements. GPU instruction
counts drop 13.5–15.8% in the dense profiles, while L2 traffic stays essentially
unchanged. All 78 synthetic checks pass memory, race and synchronization checks.
This remains an opt-in component experiment; it does not enable a CT serving integration.
See `docs/evidence/fp8-byteplane-gpu-20261024/` and `docs/HUFFMAN_RANS.md` for
paired samples, format details and profiler evidence.

The producer also has an opt-in `--async-tile-policy compact-only` build.
It omits raw-tile flags and fallback handling, and advertises **ABI 2** so older
bindings reject it. The container format and compression ratio are unchanged.
The Python binding rejects a forced compressed-only binding if any tile is raw;
`compression="auto"` keeps the original raw weights when the tensor is ineligible.
The default build remains `mixed`.

To compare it, pass the previous general library as `--async-library`, the new
library as `--compact-only-library`, and `--async-codec byteplane` to
`tools/bench_fp8_marlin.py`. Eligible kernels share the exact same immutable
encoded GPU buffer. Ineligible tensors use the raw control and are explicitly
reported as fallback with zero retained savings.

On the real CT shards, the smaller gate/up layer improves about 5–10% over the
general compressed kernel, while the dense layer changes by less than 1% in
paired warm comparisons. All 16 eligible shape/alignment cases still fail the
2% warm slowdown limit. Two smaller-layer cache-pressure cases are about 3%
faster than stock, but their warm penalties remain about 50–61%; this does not
qualify serving. All 143 GPU checks pass memory, race and synchronization
checking. Dense profiles execute 7.7–10.1% fewer SM instructions with unchanged
register counts and essentially unchanged L2 traffic. Details and all samples are in
`docs/evidence/fp8-compact-gpu-20261024/` and `docs/HUFFMAN_RANS.md`.

### Routed FP4-scale decoding

`tools/scale_dictionary.cu` also exposes device-side expert deduplication and
selective `vector4-warp` decoding. It writes only routed experts at their
original addresses, preserving the existing consumer. The full-address scratch
buffer is still required. These additions do not enable a serving hook or
change the existing decoded-scale cache.

```bash
python tools/bench_selected_scales.py \
  --library build/libzipserv_scale_dictionary.so --checks-only \
  --report /tmp/selected-scale-check.json
python tools/bench_selected_scale_gemm.py \
  --library build/libzipserv_scale_dictionary.so \
  --aligned-library /path/to/libmxwave_moe_aligned.so \
  --ct-library /path/to/libmxwave_moe_sparse.so \
  --source /path/to/ct-checkpoint --report /tmp/selected-scale-gemm.json
```

The GEMM comparison additionally requires the separately installed MxWave
`moe_runtime` bridge. It loads all 288 experts of one real layer under a 3 GiB
CUDA allocation cap. All eight tested outputs match CT's existing fused
palette kernel exactly. Selective decoding removes much of the earlier
prototype's wasted work, but the complete GEMM remains 6.6–31.6% slower by
paired median than that CT kernel. Dictionary scale storage is 41.1% smaller;
codes plus scales save only 0.63%, before a required 36 MiB shared scratch
buffer. Synthetic routing and activations do not qualify full-model serving.
Keep the existing CT recipe; these experiments are opt-in tools only.

### FP4 scale expansion and shared-memory reuse

`tools/build_fp4_scale_expansion.py` builds an isolated comparison with CT's
existing sparse-palette MXFP4 consumer. `--expansions palette affine` selects
the original palette lookup and an arithmetic expansion for consecutive
three-value palettes. The byte decoder retains the original lookup for other
palettes and the exact exception path. The existing CT GEMM interface still
requires verified consecutive palettes. The algorithm contains no model-name
or layer-specific rules; this integration is currently qualified only as a
CT/Marlin component with E8M0 group-32 scales on SM121. It is not an NVFP4 codec.

The experiment also found a shared-memory hazard in the frozen CT baseline:
a final speculative weight read can overlap reuse of that storage for partial
sum reduction. `--tail-prefetch barrier` (default) orders that handoff in both
rebuilt arms. `skip` removes the unused read but slows the tested small warm
down-projection cases by about 11%; `keep` reproduces the original diagnostic
behavior and its race-check failures. No serving library is replaced.

The arithmetic expansion preserves the existing compressed representation;
it saves **zero additional weight/scale bytes** and requires no extra decoded
cache. The selected complete layer retains 27.02 MiB of scales instead of
108 MiB, alongside 1,728 MiB of unchanged FP4 codes: CT's existing savings are
74.99% of scales, or 4.41% of codes plus scales. These are tensor payloads for
one TP2 rank-zero layer, not a whole-model allocation measurement.

Both corrected consumers pass 256 GPU decoder fixtures and eight guarded GEMM
fixtures, with bit-exact CT outputs, graph replay and non-default streams.
Memory and race checks pass. Native synchronization checks pass with independent
CPU routing fixtures; the installed vLLM small-batch routing kernel itself
fails synccheck on a tiny synthetic case. That failure is preserved separately;
real-layer timings use the unchanged vLLM device router and all 288 experts.

```bash
python tools/build_fp4_scale_expansion.py \
  --ct-source /path/to/pinned/ct-source \
  --ct-manifest /path/to/qualified/ct/manifest.json \
  --source /path/to/pinned/vllm-source --output /tmp/fp4-expansion \
  --expansions palette affine --tail-prefetch barrier \
  --torch-include /path/to/torch/include
python tools/bench_fp4_scale_expansion.py \
  --ct-library /path/to/qualified/ct/libmxwave_moe_sparse.so \
  --palette-library /tmp/fp4-expansion/palette/libzipserv_fp4_scales.so \
  --affine-library /tmp/fp4-expansion/affine/libzipserv_fp4_scales.so \
  --source /path/to/ct-checkpoint --report /tmp/fp4-real.json
```

The builder verifies all external CT source hashes and recreates every staged
source hash from the frozen baseline before applying either change. The
separately supplied MxWave runtime and pinned vLLM sources remain required;
see `tools/FP4_SCALE_NOTICE`. This experiment is not registered as a vLLM plugin.

Two paired real-layer runs place the affine candidate between 1.74% faster and
1.49% slower than frozen CT in warm cases, with no consistent improvement over
the corrected palette control. Its profiled down-projection executes 0.81% more
SM instructions, with essentially unchanged L2 traffic and active-warps
percentage. This does not justify selecting the arithmetic decoder by default.
The shared-memory reuse correction is independently useful. Full evidence,
including the original failures, is in
`docs/evidence/fp4-scale-expansion-gpu-20261024/`.

Across all 42 GLM MoE layers and both ranks, the existing scale audit records
8.859 GiB raw versus 2.221 GiB packed: **6.639 GiB already saved**. Current scale
and embedding packing, after reported expert workspace, accounts for 6.941 GiB.
These historical byte receipts are distinct from a new whole-model allocator
measurement. This decoder experiment adds zero bytes of savings to that baseline.

### Optional decoded-scale cache

The experimental vLLM 0.32 adapter now has `--scale-cache on|off`,
`--scale-cache-mib`, and `--scale-cache-policy retain|lru` switches. It keeps
compressed scale dictionaries on the GPU, reconstructs misses into shared
scratch, and retains selected decoded scale tensors within the requested
per-worker memory budget. Disabling the cache releases its arena; compressed
backing and the shared decode scratch remain necessary. FP4 weight codes and
the attention KV cache are unchanged.

This path requires the separately supplied MxWave aligned-nibble MoE consumer
and eager execution. A miss currently reconstructs the whole scale tensor,
including inactive experts. A small cache can therefore have a low hit rate;
LRU can also thrash on a repeated full-model scan. `retain` avoids that scan
pattern by keeping its initial admissions. These are experimental policies,
with no guaranteed speedup. See the [adapter instructions](integrations/vllm/README.md#experimental-mxfp4-scale-cache)
for dependencies, serving flags, GPU checks, and the real-model off/on/on/off
benchmark. The benchmark reports actual hits, retained bytes and matched-output
latency, rather than treating dictionary coverage as a cache hit rate.

### Adding BF16 ZipServ to the existing CT candidate

A separate experiment now adds the current ZipServ runtime to eleven remaining
BF16 attention output projections while preserving the CT candidate's fused
scale compression, other adapters, speculative decoding and CUDA graphs.
Its startup flag is `--mode off|on` in `tools/launch_ct_bf16_extra.py`.
This trial is restricted to the existing two-GB10 GLM CT environment and
its exact legacy vLLM image; the supported vLLM 0.32 integration is unchanged.

The full-model run saved an additional 413 MiB across both workers (29.35% of
those weights, 0.23% of live CUDA allocations). All 22 TP shards decoded
exactly. Repeated generated outputs varied in both CT and the candidate, so
timing differences do not establish a general speedup or output equivalence.
See the [CT experiment instructions](integrations/vllm/README.md#experimental-addition-to-the-existing-ct-serving-recipe)
for the narrow target policy, preserved settings and benchmark tools.

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

