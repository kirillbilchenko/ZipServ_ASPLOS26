# Experimental vLLM 0.32.0 integration

This package adapts the local MxWave BF16 integration to this ZipServ checkout.
It builds the native code here and registers a vLLM plugin without modifying
vLLM files. See [Verification](#verification) for the validation scope.

## Build and install

Use a working **vLLM 0.32.0** environment and CUDA compiler supporting the target
GPU. Verification used Torch 2.13.0+cu130, compressed-tensors 0.17.0,
CUDA 13.0 and GB10 SM121. A compatible GPU build of vLLM must already work.

From the repository root:

```bash
# Choose the actual GPU SM: 80, 86, 89, 90, 100, 120 or 121.
python integrations/vllm/build_native.py --arch 121

# Preserve the installed platform-specific vLLM/Torch stack.
python -m pip install --no-deps ./integrations/vllm

export ZIPSERV_LIBRARY="$PWD/build/libzipserv_vllm.so"
export VLLM_PLUGINS=zipserv
```

CUDA_PATH, when set, selects its bin/nvcc; otherwise the builder uses nvcc on
PATH or /usr/local/cuda/bin/nvcc. The builder emits a neighboring JSON manifest
with source/compiler/library hashes and ABI version. Rebuild after native
changes. Library paths must be accessible on every worker. If other general
plugins are required, include them in VLLM_PLUGINS as well.

For an existing offline environment with setuptools >=77, installation can use
`--no-build-isolation`. The wheel contains the Python package and notices;
the native library is built separately.

## CPU-only packing

The CPU encoder and Python runtime can be used from a source checkout on Linux
or macOS without CUDA or vLLM. A C++17 compiler and a working Torch installation
are sufficient:

```bash
python integrations/vllm/build_native.py --cpu-only --output build/libzipserv_cpu.so
export ZIPSERV_LIBRARY="$PWD/build/libzipserv_cpu.so"
export PYTHONPATH="$PWD/integrations/vllm"
```

`CXX` selects the CPU compiler. Without an explicit output, CPU builds create
`build/libzipserv_cpu.so` on Linux or `build/libzipserv_cpu.dylib` on macOS.
The manifest records `cuda: false`. CPU-only builds expose packing and cleanup;
GPU execution requires rebuilding with `--arch` and selecting that library.

`compress()` returns CPU tensors sharing the native allocations. A shared owner
keeps all seven buffers and the native library alive until the last tensor
storage view is released, even if the original `PackedWeight` is deleted.
Empty streams use ordinary empty tensors. No floating-point conversion or
quantization is performed, and the native ABI and packed format are unchanged.

## Raw BF16 model

```bash
vllm serve /path/to/raw-bf16-model --dtype bfloat16 --quantization zipserv
```

No quantization sidecar or checkpoint rewrite is required. This method uses
vLLM's quantization interface to select **lossless storage**, not to round BF16
weights. Source linear weights must actually be BF16: FP16 input is rejected.

Eligible LinearBase weight shards are packed on CPU, transferred to CUDA and
the dense parameter is released. Dimensions must be positive multiples of 64
and satisfy the native int32/grid limits. Unaligned or nonshrinking weights
remain dense. Embeddings, output heads, MoE methods and attention/cache kernels
are not replaced by the raw-BF16 adapter; attention's ordinary linear projections
are eligible. Packing can temporarily increase host memory.

Default Split-K is one. Programmatic ZipServConfig accepts a positive split_k
and exact module-prefix ignore list. Other checkpoint formats are rejected.

## Existing compressed-tensors MXFP4 model

```bash
vllm serve /path/to/mxfp4-model --dtype bfloat16 --quantization zipserv-mxfp4
```

This narrower mode requires quant_method=compressed-tensors and
format=mxfp4-pack-quantized in the existing model config. It preserves the stock
quantized methods and replaces only an **untied, unquantized BF16 output head**.
Tied, quantized or transformed heads are rejected. It does not compress
MXFP4/NVFP4 scales or introduce a new FP4 compute kernel.

## Verification

```bash
make -C tests cpu
make -C tests memcheck SM=121

python integrations/vllm/check_runtime.py --report /tmp/zipserv-runtime.json
python integrations/vllm/check_adapter.py
```

After package installation, the scripts can import zipserv_vllm directly.
For source-only testing, set PYTHONPATH to the absolute integrations/vllm
directory. check_adapter.py initializes and destroys actual single-rank NCCL
groups, loads real vLLM linear classes, and checks fused shard loading,
decompression, fallback and Dynamo capture.

CPU-only Python checks build the host bridge in a temporary directory and test
bitwise decoding, shared pointers, retained views and failure cleanup. They need
a C++17 compiler and a Torch/pytest environment, without CUDA or vLLM:

```bash
python -m pytest -q tests/test_runtime.py tests/test_host_runtime.py
```

Verified: native SM121 correctness and memcheck, shared-buffer CPU ownership
on Torch 2.13/2.14, Torch custom-op schema/fake behavior, graphs/streams, vLLM
0.32 TP1 linear loaders, package installation and entrypoint discovery. The
shared-buffer CUDA runtime checks also pass under Compute Sanitizer. Native reconstruction preserves every BF16 word;
floating-point GEMM outputs can differ from another backend's rounding.

Not yet qualified: full-model serving on 0.32, TP2, hybrid-head model execution,
LoRA, weight refit, other GPUs and production throughput. Dynamo tests use
backend=eager and are not an Inductor performance benchmark. Use the included
scripts as component checks, then compare the intended model and serving load.

The plugin rejects other vLLM release versions. PackedWeight is an in-process
trusted object produced by this encoder; it is not a validated interchange
format for arbitrary packed buffers. See package LICENSE, NOTICE and
[source-provenance.json](source-provenance.json) for the MxWave attribution.
