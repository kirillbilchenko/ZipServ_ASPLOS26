# Experimental vLLM 0.32.0 integration

This package provides BF16, MXFP8 weight and MXFP4 expert-scale storage adapters
for this ZipServ checkout.
It builds the native code here and registers a vLLM plugin without modifying
vLLM files. See [Verification](#verification) for the validation scope.

## What the current CT results mean

All memory figures here are totals across **both GPUs**. The same-runtime CT
comparison is the appropriate reference for deciding whether this fork improves
the existing deployment.

Detailed run artifacts identified by `docs/evidence/` paths are retained in
separate experiment backups. They are excluded from this source checkout's
commits; the results and their limitations are summarized below.

| Comparison | Live CUDA memory | What has been established |
|---|---:|---|
| CT runtime, optional weight representations disabled | 188.935 GiB | Reference with the same runtime, speculative decoding and KV budget |
| Existing CT compression recipe | 179.210 GiB | Saves **9.725 GiB**; sampled concurrent throughput +9–15%; GSM8K 1248/1319 in both arms |
| Existing CT plus the new narrow BF16 add-on | 178.807 GiB | Saves **another 0.404 GiB**; speed evidence is mixed; full GSM8K for this add-on is pending |
| Existing CT plus new KDA FP8 packing | **Not measured in serving** | Component payload saving **0.320 GiB**, before runtime buffers; current kernel is slower |

Existing CT's largest payload reductions come from FP4 **scales**, BF16
embedding packing and its pre-existing BF16-to-MXFP8 KDA conversion. FP4 weight
codes are unchanged. The first two preserve stored values; KDA conversion is
quantization. Equal aggregate GSM8K scores do not establish identical answers
or general quality equivalence. Existing CT also had 6–8% higher long-prompt
first-token latency in that comparison.

The separate **vLLM 0.32 plugin experiment** saved 10.283 GiB against its own
packing-disabled baseline (176.576 → 166.293 GiB). That establishes working
BF16/FP4-scale/FP8 integration under that profile. It does **not** establish an
extra 10.283 GiB over existing CT: the runtime, context, KV budget, speculation,
and KDA representation differ. Its GSM8K results were 1251/1319 without packing,
1245/1319 with all three types enabled, and 1250/1319 with BF16/FP4 enabled and
FP8 packing disabled. Each arm has one full run; they do not isolate a causal
quality effect. The historical 14.877 GiB number uses yet another ModelOpt
comparison and cannot be added to these savings.

The delivered value is a maintained, optional vLLM integration with CPU packing
improvements, fused consumers, separate controls and source/memory audits.
The demonstrated **incremental memory benefit on existing CT is currently
0.404 GiB**. Broader coverage and faster FP8 decoding remain experiments.
Use the detailed [direct CT comparison](#incremental-benefit-over-existing-ct-2026-10-23),
quality readout (`docs/evidence/quality-remaining-20261025/final-readout.json`)
and [KDA results](#broader-ct-coverage-and-kda-component-results-2026-10-25)
when comparing the next candidate.

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

## Experimental MXFP4 scale cache

This separate, opt-in adapter compresses the **prepared MXFP4 MoE scales** and
feeds exactly reconstructed aligned-nibble buffers to an existing MxWave
Marlin consumer. It leaves the FP4 weight codes unchanged. It is not the BF16
`zipserv-mxfp4` output-head adapter above, nor a generic FP4/FP8 serving codec.

The current integration targets the CT GLM checkpoint and qualified
`mxwave_glm53._runtime.model_plugin` package on vLLM 0.32.0. That package,
its `moe_runtime`, and its raw/aligned native libraries must already be
available. They are **not bundled** with ZipServ. The adapter verifies the
package's upstream source contract. An independently qualified loose-module
backend can instead be selected with `--moe-support /path/to/support`, which
must contain `model_plugin.py` and `moe_runtime.py` compatible with 0.32.0.

Build the dictionary decoder from the repository root, selecting the actual
GPU architecture, then install this package as above:

```bash
mkdir -p build
nvcc -std=c++17 -O3 -shared -Xcompiler=-fPIC -arch=sm_121 \
  tools/scale_dictionary.cu -o build/libzipserv_scale_dictionary.so

export MXWAVE_MOE_MODE=aligned
export MXWAVE_MOE_DISPATCH=dispatcher
export MXWAVE_MOE_RAW_LIBRARY=/path/to/libmxwave_moe_raw.so
export MXWAVE_MOE_ALIGNED_LIBRARY=/path/to/libmxwave_moe_aligned.so
export MXWAVE_MOE_RECORD_CODEC=/path/to/qualified-model/config.json
export VLLM_PLUGINS=zipserv_scale_cache

python tools/serve_scale_cache.py \
  --scale-cache on --scale-cache-mib 256 --scale-cache-policy retain \
  --dictionary-library "$PWD/build/libzipserv_scale_dictionary.so" -- \
  serve /path/to/qualified-model --enforce-eager \
  --compilation-config '{"mode":0,"cudagraph_mode":"NONE"}' \
  --host 127.0.0.1 --port 18794 --served-model-name zipserv-scale-cache-test
```

Add the model's normal tensor-parallel and memory settings. Every worker needs
the same configuration and accessible native library paths. The cache budget
is **per worker**, so 256 MiB on two workers allocates 512 MiB in total.
`--scale-cache off` retains no decoded cache arena. Both modes still need the
compressed dictionaries, tiny descriptors, and one shared scratch buffer as
large as the largest registered decoded scale tensor. Memory reports count
all four owners and distinguish live CUDA allocations from allocator reserves.
Raw fallback scale tensors and other model state are outside that scale-only
accounting.

`retain` keeps the first tensors that fit and bypasses subsequent misses.
`lru` replaces the least recently used tensors, which can produce zero hits
when every token scans a working set larger than the cache. The current unit
is a **whole scale tensor**, not an expert. Misses also decode inactive experts;
the existing small, eight-expert GEMM timings do not predict full-model miss
cost. CUDA graphs/compilation are rejected, and GPU uses on different streams
are serialized for safe shared-buffer reuse. These restrictions can affect
performance relative to a production serving configuration.

Preparation verifies original scale bits against native reconstruction and
GPU dictionary reconstruction. Sources outside the aligned-nibble range keep
their original raw scale descriptors. The checkpoint and FP4 values are not
rewritten. Dictionaries are currently prepared in memory on every startup;
the decoded GPU cache does not accelerate checkpoint loading.

For an isolated real-model A/B run, start the server with
`VLLM_SERVER_DEV_MODE=1` and keep its API bound to loopback. The benchmark uses
worker RPCs to switch modes in the same loaded model, and should run without
other requests:

```bash
python tools/bench_scale_cache_model.py \
  --base http://127.0.0.1:18794 --model zipserv-scale-cache-test \
  --workers 2 --cache compare --cache-mib 256 --cache-policy retain \
  --tokens 24 --repeats 3 --report /tmp/scale-cache-abba.json
```

Use `--workers 1` for a single-worker model that fits that device; the cache
itself does not require tensor parallelism. `--cache on` or `--cache off`
measures only one configuration. Comparison mode performs off/on/on/off
trials, renders prompts through the model's chat template, records cold
requests separately, warms the shared compute paths, checks exact generated
token IDs, and reports TTFT, decode rate, request
latency, cache counters and memory per worker. Positive percentage changes
in its summary indicate improvement with caching. The baseline is the same
dictionary-compressed model with caching disabled; this does not establish
performance against a native-scale or graph-enabled production baseline.
Short sequential prompts do not qualify longer contexts, concurrency or
general model quality.

The default `--output-check strict` stops on any token mismatch, including
variation already present with caching off. For diagnosis,
`--output-check record` completes the same workload while preserving every
output and marking `output_check_passed: false` and
`status: measured_with_output_variation` if a mismatch occurs. It reports
all fixed-input/fixed-length pairs and the subset with identical output
tokens separately. This diagnostic mode does not qualify model equivalence
or a performance recommendation. All registered GPU scale groups must still
reconstruct exactly before the run and after every measured phase; a source
check failure always aborts. `--prompt-format text` explicitly selects raw
text completion instead of the default chat template.

To measure compression completely off, start a separate control instance
with `--scale-representation raw --scale-cache off`. Use the same checkpoint,
vLLM image, eager configuration, parallelism, KV budget and request workload.
This mode observes the original E8M0 scale tensors after stock loading and
leaves both those tensors and vLLM's GEMM unchanged. It does not allocate
dictionary backing, decode scratch or a decoded cache, and does not require
`--dictionary-library`. The model's original CT FP4 quantization remains.
The qualified model package is still needed for the same checkpoint projection
aliases and vLLM source contract.

```bash
python tools/serve_scale_cache.py \
  --scale-representation raw --scale-cache off -- \
  serve /path/to/qualified-model --enforce-eager \
  --compilation-config '{"mode":0,"cudagraph_mode":"NONE"}' \
  --host 127.0.0.1 --port 18794 --served-model-name zipserv-scale-cache-test

# Against that separate, idle control instance:
python tools/bench_scale_cache_model.py \
  --representation raw --cache off --workers 2 --tokens 24 --repeats 6 \
  --report /tmp/scale-raw-control.json
```

`--scale-cache off` alone continues to use dictionaries and decode every
access. It is not the compression-off baseline. The raw control's scale
hashes can be compared to the dictionary run's original-source hashes.
Cross-process timings are less controlled than the same-process cache A/B;
report the two comparisons separately. Neither eager profile represents
the graph-enabled production service.

Independent cache checks use the real GPU native decoder as a consumer,
exercise on/off transitions, budget bounds, eviction, multiple streams and
failure handling:

```bash
python3 -m unittest discover -s tests -p test_scale_cache_policy.py -v
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python tools/check_scale_cache.py \
    --library build/libzipserv_scale_dictionary.so \
    --moe-library "$MXWAVE_MOE_ALIGNED_LIBRARY" \
    --report /tmp/scale-cache-gpu-check.json
```

The adapter is inactive unless `ZIPSERV_SCALE_DICTIONARY=1`. The launcher sets
that variable and translates its flags to `ZIPSERV_SCALE_CACHE` (0/1),
`ZIPSERV_SCALE_CACHE_MIB`, `ZIPSERV_SCALE_CACHE_POLICY`, and
`ZIPSERV_SCALE_DICTIONARY_LIBRARY`. `--scale-representation` selects
`ZIPSERV_SCALE_REPRESENTATION` (`dictionary` by default, or `raw`). It installs the
`zipserv_vllm.scale_cache_plugin.ScaleCacheWorker` extension for reports and
idle-boundary configuration. Reconfiguration invalidates existing residents;
resetting counters alone preserves them.

## Experimental addition to the existing CT serving recipe

`tools/ct_bf16_extra_plugin.py` tests the current ZipServ runtime on eleven
remaining BF16 MLA output projections while retaining CT's existing fused
MXFP4 scale decoder, compressed embedding, MXFP8 KDA adapters, shared dense
output head, speculative decoder and CUDA graphs. It is a separate experiment
adapter for vLLM `0.1.dev20051+g487ecf187`, the pinned GLM CT checkpoint and
two GB10 workers. It does not relax the supported plugin's vLLM 0.32 guard.

`tools/launch_ct_bf16_extra.py --mode off|on` creates private serving clones
from the stopped, qualified CT containers. It requires idle GPUs and an
already prepared `/tmp/zipserv-ct-extra.*` directory containing the current
native library, Python runtime and trial tools. The original CT containers,
model mounts, device mappings and configuration must exist; this launcher is
not a general-purpose CT deployment command. It retains the original adapter
environment, RDMA transport, 500k context capacity, 6 GiB KV per worker and
graph settings, with separate cache/report directories and a loopback API on
port 18795. It omits the production secret mount.

The `off` arm observes the original BF16 weights. The `on` arm verifies each
TP shard against the source checkpoint, packs it, checks exact GPU recovery
and removes the dense parameter. Packed bytes are registered as non-trainable
parameters so CT's existing prefetcher can discover their live addresses.
The experimental dispatch uses Split-K 8 through 32 tokens, Split-K 4 through
128, Split-K 1 through 768, and temporary decoding followed by the stock GEMM
above that. That last path has a 64 MiB decoded weight temporary per worker;
its allocation is reused by the allocator, not a persistent decoded cache.
The setting is fixed at startup because captured graphs retain their operands.

`tools/bench_ct_bf16_extra.py` screens real weight shards with synthetic
activations and CUDA graphs. `tools/bench_ct_bf16_model.py` measures the actual
CT stack, checks both workers before and after timing, records live allocation
and speculative-decoding counters, and compares fixed-length output token IDs.
The candidate requires the baseline report via `--reference`:

```bash
# Against separate, otherwise identical private CT server starts:
python tools/bench_ct_bf16_model.py --mode off --report /tmp/ct-off.json
python tools/bench_ct_bf16_model.py --mode on \
  --reference /tmp/ct-off.json --report /tmp/ct-on.json
```

The measured additional saving was 413 MiB across both GPUs: 29.35% of the
selected 1.375 GiB weight pool, but only 0.23% of CT's approximately 179.21 GiB
live Torch CUDA allocation. All 22 shards recovered exactly before and after
serving. Both the original CT and candidate varied their generated tokens on
repeated requests. Those diagnostic timings do not establish model-output
equivalence, a general speedup or production quality. The experiment is
inactive unless its separate plugin is registered and
`ZIPSERV_CT_EXTRA_BF16=off|on` is explicitly set.

## Standalone MXFP4 expert-scale plugin

`zipserv_mxfp4_scales` compresses prepared E8M0 group-32 scales in eligible
compressed-tensors MXFP4 MoE layers. The Python codec, native consumer and
vLLM integration are bundled here; an external MxWave runtime is unnecessary.
Every 16 scale bytes use a four-byte record with a three-value palette or an
exact exception. Packing occurs on the CPU during model loading. GPU GEMMs
expand records into shared memory as they consume them, preserving all FP4
codes and scale bytes. This path does not retain a dense decoded-scale cache.
Reduction scratch is shared across layers, with one allocation per used CUDA
stream; the worker audit includes its cost. Allocator and CUDA-graph pools can
make whole-model memory changes differ from packed-payload savings.

Build the consumer from the vLLM source revision recorded in
`native/mxfp4/moe_sources.json`:

```bash
# /path/to/vllm-native must contain revision
# db9527a46873454610df6dbedf79a36d6bf1a7f6; the builder verifies individual hashes.
python integrations/vllm/build_mxfp4_native.py \
  --source /path/to/vllm-native --arch 121 --output build/mxfp4-native
python -m pip install --no-deps ./integrations/vllm

export VLLM_PLUGINS=zipserv_mxfp4_scales
python tools/serve_mxfp4.py --mxfp4-scales on \
  --mxfp4-library "$PWD/build/mxfp4-native/libzipserv_mxfp4.so" \
  --min-scale-saving-percent 1 --scale-audit -- \
  serve /path/to/model --dtype bfloat16 --moe-backend marlin
```

Use `--mxfp4-scales off` for the unchanged stock-scale/GEMM control; no native
library is required in that mode. Settings are fixed at startup because CUDA
graphs retain operand addresses. `--min-scale-saving-percent` screens estimated
allocation savings for each scale tensor, including 512-byte allocation rounding
and metadata. It is not a latency prediction or a whole-model saving threshold.
The native build adds a barrier before shared weight storage is reused by the
reduction. Its manifest records source, compiler, architecture and binary hashes.

The equivalent environment settings are `ZIPSERV_MXFP4_SCALES=on|off`,
`ZIPSERV_MXFP4_LIBRARY`, and `ZIPSERV_MXFP4_MIN_SAVING_PERCENT` (default `1`).
With no mode selected, the entry point is inert. Explicit conflicting CT or
dictionary-scale owners are rejected to prevent double compression.

Eligibility is determined from tensor layout and runtime configuration, without
model-name checks: Marlin experts, BF16 activations, contiguous finite E8M0
group-32 scales and supported tile shapes. Incompressible scales and unsupported
backends, expert parallelism, LoRA, expert bias or weight offloading retain stock storage. Native
and Python API incompatibilities fail before raw scale tensors are released.
This implementation is for MXFP4 expert scales; NVFP4 and FP4 linear layers need
their own consumers. GPU qualification currently covers GB10 SM121.

`--scale-audit` installs the optional
`zipserv_vllm.mxfp4_backend.ScaleWorker` extension. Its
`zipserv_mxfp4_report(verify=True)` worker RPC reports every visited layer,
compressed and fallback groups, payload and scratch bytes, CUDA allocation,
source-owner release, exact GPU reconstruction and observed eager/capture calls.
Graph replays do not increment Python call counts. An existing worker extension
can inherit this mixin; the launcher refuses to overwrite it. RPC access follows
the server's normal security settings; the launcher does not expose a debug API.

```bash
python3 -m unittest discover -s tests -p 'test_mxfp4*.py'
PYTHONPATH="$PWD/integrations/vllm" python integrations/vllm/check_mxfp4.py \
  --library build/mxfp4-native/libzipserv_mxfp4.so --report /tmp/mxfp4-check.json
```

Component checks cover consecutive and nonconsecutive palettes, exact exceptions,
stock output comparison, CUDA graph replay, nondefault streams, ownership and
nonshrinking fallback. Full-model coverage and serving performance are checked
separately with `tools/bench_mxfp4_model.py` against private, otherwise identical
on/off servers. It requires an expected layer count and verifies every worker
before and after serving. Fixed prompts are a smoke test, not a broad quality
evaluation.

## Experimental mixed BF16 / FP8 / FP4 storage

The `zipserv_mixed_weights` entry point adds explicit post-load BF16 and MXFP8
targets alongside the MXFP4 scale plugin. It is inert unless
`ZIPSERV_MIXED_BF16` or `ZIPSERV_MIXED_FP8` is set. A matched full-model CT
experiment is reported below; its results do not qualify other serving profiles.

`tools/serve_compressed.py` exposes independent controls:

| Flag | Stored representation | Inference consumer |
| --- | --- | --- |
| `--mxfp4-scales on` | Exact E8M0 scale palettes and exceptions | Fused MXFP4 MoE Marlin kernel; FP4 codes unchanged |
| `--bf16-weights on` | Existing ZipServ BF16 payload | Fused BF16 GEMM through 768 tokens; exact decode and stock GEMM above |
| `--fp8-weights on` | Exact ZFB1 prepared Marlin weight bytes | Fused MXFP8 GEMM through 128 bias-free tokens; exact decode and stock GEMM otherwise |

Every mode defaults to `off`. Supply the corresponding `--bf16-library`,
`--fp8-library` and `--mxfp4-library` for enabled modes. The FP8 library is the
mixed-tile producer/uniform/byteplane variant built by `tools/build_fp8_marlin.py`
without `--exact-scale-cache` or the compact-only policy. FP8 scales remain raw.
No persistent decoded-weight cache is retained; temporary prefill reconstruction
and per-stream FP8 reduction scratch must be included in memory measurements.
Fused FP8 warmup also reserves the stock Marlin locks for each active CUDA stream,
microbatch and workspace lane. Larger mixed prefill/decode batches can then use
the stock fallback after vLLM locks workspace allocation.

BF16 and FP8 activation requires explicit, repeatable `--bf16-target` and
`--fp8-target` module-name patterns. Inspect `zipserv_weight_inventory` and the
model's direct weight readers before choosing targets. Ordinary linear forwards
can use the adapters; an indexer, absorbed attention matrix or custom router may
read `.weight` directly. Shared source storage and unsupported consumers fall
back, and every fallback is reported. The generic plugin contains no model-name
selection. BF16 output rows are padded to 64 and sliced back to their logical
size; input columns must already be 64-aligned. Padding does not quantize weights.

`--min-weight-saving-percent` and `--min-scale-saving-percent` default to 1 and
compare estimated CUDA allocation sizes including metadata. They filter storage
savings, not latency. FP8 component measurements have shown a slowdown despite
smaller weights, so measure the intended model with FP8 both disabled and enabled.

`--audit` installs `MixedWorker`, which adds `zipserv_mixed_report(verify=True)`
and retains the FP4 report and whole-model inventory. Reports include byte-exact
reconstruction, source-owner release, per-type storage, raw fallbacks, scratch and
dispatch counts. Graph replays do not increment Python dispatch counts. Use the
whole-model CUDA allocation difference for net savings; per-type payload totals
exclude KV cache and graph/output workspaces.

The current mixed experiment requires BF16 activations, compilation mode 0 and
either eager execution or `FULL_DECODE_ONLY` CUDA graphs. LoRA, offloading, sleep
mode, weight transfer and batch-invariant GEMM are rejected. The adapter checks
the reviewed vLLM 0.32 API source hashes. Lossless stored weights can still produce
different floating-point GEMM rounding, so evaluate answer quality separately.

`integrations/vllm/check_mixed.py` checks aligned and padded consumers, graph
replay and ownership. `tools/eval_ct_gsm8k.py` provides a frozen, full 1,319-question
GSM8K protocol with all errors/truncations retained and paired answer outcomes.
The baseline and candidate must use identical scoring code, prompts and serving
settings. Keep partial runs separate from complete quality scores.

`tools/compare_mixed_model.py --control RUN_OFF --candidate RUN_ON --report FILE`
checks matching launch settings, image, model mounts, prompt tokens and scale
hashes before comparing completed `benchmark.json` reports. Each run directory
must also contain `start-receipt.json` and `weight-inventory.json`; the candidate
needs `mixed-after-benchmark.json`. The result separates compressed-target
payload savings from the net CUDA allocation difference and records raw
fallbacks using their actual tensor dtype. Complete `gsm8k/` results add paired
answer outcomes; missing or partial quality runs remain explicitly pending.

### Measured CT profile, 2026-10-24

The matched GLM CT comparison used two GB10 workers, vLLM 0.32.0, BF16
activations, an 8,192-token context, 4 GiB of FP8 KV cache per worker, Marlin
linear/MoE backends, and `FULL_DECODE_ONLY` graphs with capture sizes `[1,4]`.
Speculative decoding and prefix caching were disabled in both arms.
Both arms retained the checkpoint's original FP4/FP8 quantization. The control
disabled additional ZipServ storage; the candidate enabled BF16, FP8 and FP4
scale packing together. This control is distinct from the older CT production
deployment with MXFP8 KDA, a much longer context and speculative decoding.

| Measurement | Additional storage off | All three on |
| --- | ---: | ---: |
| Live Torch CUDA allocation, both workers | 176.58 GiB | 166.29 GiB |
| Median single-request decode, 12 fixed 64-token requests | 16.29 tokens/s | 20.56 tokens/s |
| Median first-token latency for those requests | 184.67 ms | 185.93 ms |
| Full GSM8K output throughput, concurrency 4 | 43.46 tokens/s | 44.69 tokens/s |
| GSM8K correct answers, same zero-shot protocol | 1,251/1,319 (94.84%) | 1,245/1,319 (94.39%) |
| API errors / truncated GSM8K answers | 0 / 0 | 0 / 0 |

Net allocation fell **10.28 GiB (5.82%)** across both workers. Compressed-target
payload savings were 6.64 GiB for FP4 scales, 3.62 GiB for BF16 weights including
eligible KDA projections, and 105.39 MiB for FP8 weights. Payload totals exclude
additional scratch and graph/allocator effects; use the net allocation result
for the whole-model saving. FP4 code bytes remain unchanged. Every worker
compressed 84 scale groups across all 42 MoE layers, 181 BF16 modules and 90
FP8 modules. Post-evaluation checks recovered all original bytes, confirmed
source-owner release and found no live-allocation growth from the benchmark.

The historical **14.877 GiB CT-versus-ModelOpt** saving used different
checkpoint formats, quantization and serving settings. It overlaps this
compression benefit and must not be added to 10.28 GiB. The total advantage of
the new full ZipServ setup over an equally configured ModelOpt deployment has
not been measured. Here, "all three on" means eligible BF16 weights, FP8
weights and FP4 scales; it does not mean every model tensor or FP4 code byte
has been compressed.

Single-request decode improved 26.2%; the concurrent GSM8K workload improved
2.8% in output tokens/s. Its natural answer lengths differed between runs, so
that second number is not a fixed-output throughput comparison. First-token
latency was effectively unchanged. Quality was 0.45 percentage points lower in
the compressed run: 18 previously wrong answers became correct and 24 changed
the other way. This is one run per arm, and both arms also varied outputs in
repeated short requests. Weight storage is exact, but fused BF16 arithmetic can
round differently; these results do not establish quality neutrality or a
reproducible accuracy regression.

`tools/bench_mixed_concurrency.py` supplies a separate fixed-output FP8 on/off
ablation while BF16 and FP4 scale compression remain enabled. It synchronizes
four requests, preserves exact prompt/output token counts, separates warmup,
and audits all source bytes before and after. Its reports retain errors and
generated tokens; it does not provide a quality score.

That ablation completed with the same checkpoint, runtime, settings and sources:

| Measurement, BF16 and FP4 scales on in both arms | FP8 off | FP8 on |
| --- | ---: | ---: |
| Net allocation saving versus the additional-storage-off control | 10.25 GiB | 10.28 GiB |
| Median single-request decode | 20.20 tokens/s | 20.56 tokens/s |
| Fixed four-request batch throughput | 42.05 tokens/s | 41.76 tokens/s |
| Median first-token latency in those concurrent batches | 0.87 s | 1.07 s |
| Full GSM8K correct answers, same modern protocol | 1,250/1,319 (94.77%) | 1,245/1,319 (94.39%) |

FP8 packing added only **31.06 MiB of net allocation saving** in this profile,
despite saving 105.39 MiB of weight payload. Additional scratch and graph/pool
allocation differences consume the rest. Single-request decode was 1.8% faster
with FP8 on, while concurrent throughput was 0.7% lower. The three measured
batch ranges overlap (40.91–42.06 off, 41.57–42.40 on); this does not establish
a consistent throughput advantage for either setting. The concurrent workload
includes a 1,062-token prompt and its latency includes scheduling. Its TTFT
must not be compared with the sequential-request TTFT in the earlier table.

BF16 plus FP4 scales retains almost all the memory benefit; keep FP8 optional
and measure the intended workload. There was one server start per arm and
three concurrent batches. All post-run source/ownership checks passed, with no
live-allocation growth during the concurrent test. A separate full FP8-off
GSM8K run completed on 25 October: 1,250/1,319 correct, zero API errors and
truncations. Against the modern packing-off baseline's 1,251 correct answers,
23 became correct and 24 became incorrect. FP8-off scored five more correct
answers than all-three-on, but one run per arm does not isolate a reproducible
FP8 quality effect. Its before/after byte and ownership audits passed, and
live allocation remained exactly 178,588,636,672 bytes. See the
complete quality readout (`docs/evidence/quality-remaining-20261025/final-readout.json`).
`tools/compare_fp8_ablation.py --fp8-off RUN_OFF --fp8-on RUN_ON --report FILE`
verifies that only the FP8 setting changes, and compares `benchmark.json` and
`concurrent-fixed.json` in those directories.

The FP8 saving above covers 1.40625 GiB of target MLP weights. It does not cover
CT's separate post-load KDA MXFP8 adapter: the legacy CT recipe has another
4.3247 GiB of logical KDA FP8 values, or 4.4599 GiB including logical scales.
Native Marlin padding increases KDA code storage to 4.3330 GiB. Component
checks of that path are recorded below; serving integration is not qualified.
Modern KDA remains BF16 and uses BF16
packing instead. The current 31 MiB result must not be presented as coverage
of every FP8 tensor in CT. BF16 and IEEE FP16 are also distinct formats; native
FP16 codec integration is not qualified by these BF16 tests.

### Broader CT coverage and KDA component results, 2026-10-25

A read-only inventory of the original CT recipe found these native FP8 code
pools across both TP ranks, excluding scales and workspaces:

| Weight path | Native FP8 code storage | New byteplane codec coverage |
|---|---:|---|
| Target MLP | 1.40625 GiB | Modern runtime qualified above; CT has separate fast paths |
| KDA | 4.33301 GiB | Isolated component checks passed; no serving adapter yet |
| Draft model | 0.89844 GiB | Different BF16 scale format; current fused codec is incompatible |

All 136 captured KDA shards were actually encoded and independently decoded.
Weight payload fell from **4.33301 to 4.01268 GiB**, saving **328.02 MiB
(7.39%)**, with the existing 138.66 MiB of native scales unchanged. On 12
representative shards and 48 GEMM cases (M=1/8/32/128), GPU reconstruction and
outputs were bit-exact to the legacy CT FP32-reduction control. CT's default
policy also matched that control for these inputs. Graph replay, output guards
and steady replay allocation checks passed.

The codec was slower on every sampled operation: **5.0–24.6% under cache
pressure** (median 11.5%) and **6.9–88.3% with warm caches** (median 29.8%).
These are isolated operation latencies, not model throughput penalties. The
payload saving is also not a measured reduction in live CUDA allocation.
This result does not justify default-on KDA packing. No full-model or GSM8K
test includes this new KDA codec. The initial harness crashed because a loop
released an external timing event still referenced by its graph; the corrected
version retained both events and completed on both Sparks. See the
component readout (`docs/evidence/ct-coverage-20261025/kda-component-readout.json`).

A follow-up tested a cheaper exception-read path against the previous decoder,
a true uncompressed native kernel, and a forced-raw container control. All 48
GEMM cases across 12 captured KDA shards matched the legacy FP32 control exactly;
14 CPU checks, including native address/undefined-behavior sanitizers, passed.
The new decoder improved paired median latency by only **0.21% under cache
pressure** and **2.02% with warm caches**, with regressions in some cases. It
still cost **12.52%** and **23.33%** more than the legacy control on median,
respectively. The true raw control was within0.4% of the legacy control on
median, confirming that the compression path carries the extra cost.

The experiment did not justify enabling KDA FP8 packing. The reproducible
builder option `--async-exception-loads indexed` remains experimental, requires
uniform exception handling, and does not change the default `predicated`
implementation or the packed format. No serving adapter or full-model claim
was added. See the
optimization readout (`docs/evidence/ct-kda-opt-20261025/readout.json`).

Broader BF16 coverage also requires consumer-specific changes: CT's split
projections, router kernels, certified head and prefetch tables may read dense
weights directly. The head is shared by the target and draft and must be
counted once per rank. The existing narrow BF16 add-on below remains the
demonstrated incremental serving result. The
inventory (`docs/evidence/ct-coverage-20261025/coverage-readout.json`)
reports candidate storage pools, not promised savings or full codec eligibility.

### Incremental benefit over existing CT, 2026-10-23

The direct CT add-on test retained CT's scale/embedding adapters, MXFP8 KDA,
speculative decoder, 500K configured context, KV budget and graph settings.
It toggled only extra ZipServ BF16 packing for 11 eligible layers (22 TP shards).
Live allocation fell **179.2104 -> 178.8068 GiB across both GPUs**, an additional
**413.24 MiB (0.404 GiB, 0.225%)**. Original source shards were verified before
and after. This is the measured incremental saving over the optimized CT recipe;
the modern plugin's 10.28 GiB figure uses a different, packing-off baseline.

The concurrent sample improved from 75.78–77.42 to 98.62 tokens/s, but
single-request results varied across the two bracketing baseline starts, and
outputs varied within and across arms. This does not establish a general
speedup or model-quality equivalence. No full GSM8K result is attached to this
BF16 add-on. It remains opt-in. The test used a narrow adapter for the pinned CT
runtime, not the full vLLM 0.32 plugin. See the
direct comparison (`docs/evidence/ct-bf16-extra-20261023/comparison.json`)
and configuration checks (`docs/evidence/ct-bf16-extra-20261023/configuration-comparison.json`).

The new plugin supplies reusable integration and separate coverage controls;
it has not demonstrated a large memory or performance upgrade over existing CT.
The following CT off/on test measures the value of CT's own optional recipe,
not the incremental benefit of replacing it with the new plugin.

### Separate exact-CT runtime control, 2026-10-24

The private `tools/launch_ct_weight_control.py` experiment uses CT's pinned
vLLM `0.1.dev20051+g487ecf187`, its original overlays, DFlash speculation,
500,000-token configured context, 6 GiB of FP8 KV cache per worker, prefix
caching and production graph/scheduler settings. It imports the existing CT
adapters; it does not load this vLLM 0.32 ZipServ plugin or relax its version guard.

`--mode off` disables optional expert-scale/embedding packing, post-load KDA
MXFP8 conversion, draft-head FP8, certified-head compressed copies and mHC
BF16 twins. Checkpoint FP4/FP8 quantization and FP8 KV cache remain unchanged.
The block-FP8 draft checkpoint's context-KV correctness fix remains enabled.
The observer verifies all 68 original BF16 KDA matrices and 84 raw expert-scale
groups per worker, dense embedding/head storage and absence of optional copies.

The sequence was compression off, normal CT, then a fresh compression-off
repeat. Each launch measured six repetitions of each single-request prompt
and three four-request cohorts, with 64 output tokens per request. Off ranges
below are the two runs' medians. Actual prompts contained 24, 28 and 2,456
tokens; these are not measurements at 500K context length.
The GSM8K row comes from separate full evaluations on 25 October, one fresh
server start per arm, using the same pinned CT runtime and speculative settings.

| Measurement | Optional weight representations off | Normal CT recipe |
| --- | ---: | ---: |
| Live Torch CUDA allocation, both GPUs | 188.94 GiB in both runs | 179.21 GiB |
| Short chat decode | 35.44–36.87 tokens/s | 42.51 tokens/s |
| Code prompt decode | 59.80–62.43 tokens/s | 73.30 tokens/s |
| Longer-prompt decode | 37.25–47.27 tokens/s | 50.46 tokens/s |
| Four-request aggregate throughput | 81.01–85.54 tokens/s | 93.33 tokens/s |
| Longer-prompt first-token latency | 1.79–1.83 s | 1.94 s |
| Full GSM8K correct answers, concurrency 4 | 1,248/1,319 (94.62%) | 1,248/1,319 (94.62%) |

Normal CT saved **9.725 GiB (5.15%)** of live allocation and delivered 9–15%
higher concurrent throughput in these runs. Payload savings were 6.639 GiB
for FP4 scales, 0.349 GiB for the embedding and 4.181 GiB from KDA conversion.
Optional copies and other allocation differences consumed 1.444 GiB of that
payload reduction. Original FP4 code bytes remain unchanged. The longer-prompt
first-token latency was 6–8% higher with CT. One CT launch and two control
launches do not establish a universal speedup; generated outputs varied within
and across arms. CT includes lossy KDA conversion. Prefix caching was enabled but recorded no hits
in this workload, including the repeated short code prompt.

The full GSM8K pair used all 1,319 questions, seed 42, temperature zero,
concurrency four, low reasoning, at most 1,024 output tokens and no retries.
Both runs had zero API errors and truncations. There were 21 newly correct and
21 newly incorrect answers, so identical aggregate scores do not mean identical
outputs or establish general quality equivalence. Original raw source hashes
passed before/after checks; CT's expected adapter coverage and worker settings
were verified. This comparison tests CT's existing optional recipe, including
lossy KDA conversion, and excludes the extra ZipServ BF16 add-on. See the
paired quality results (`docs/evidence/quality-remaining-20261025/comparison-exact-ct.json`).

Both raw controls had identical final live allocations and source hashes.
All 90 measured requests and 33 warmup requests completed. The comparison
checks image, mounted sources, serving arguments, non-weight environment and
actual worker state. See
complete results (`docs/evidence/ct-weight-control-20261024/final-readout.json`)
and `tools/compare_ct_weight_control.py` for the recorded comparison.

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

Full-model TP2 evidence is limited to the mixed CT profile above; the raw-BF16
quantization adapter, other models and hybrid-head configurations still need
their own full-model checks. LoRA, weight refit, other GPUs and production
throughput remain unqualified. Dynamo tests use
backend=eager and are not an Inductor performance benchmark. Use the included
scripts as component checks, then compare the intended model and serving load.

The plugin rejects other vLLM release versions. PackedWeight is an in-process
trusted object produced by this encoder; it is not a validated interchange
format for arbitrary packed buffers. See package LICENSE, NOTICE and
[source-provenance.json](source-provenance.json) for the MxWave attribution.
