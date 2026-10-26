# Experimental vLLM integration

This package provides BF16, MXFP8 weight, MXFP4 and NVFP4 expert-scale storage adapters
for this ZipServ checkout.
It builds the native code here and registers a vLLM plugin without modifying
vLLM files. See [Verification](#verification) for the validation scope.

Source compatibility profiles cover
vLLM 0.31.0 and 0.32.0, with adapter-specific source checks. Installing the
Python package no longer replaces an existing vLLM; the optional `vllm` extra
selects that reviewed release range. Version recognition is not GPU or model
qualification: historical serving results below remain specific to 0.32.0.
The legacy CT scale-cache adapter keeps its separate runtime restriction.

The experimental ModelOpt NVFP4 scale adapter passed component GPU checks and
a 93-request TP2 serving screen on GB10 with stock vLLM 0.32 and DFlash2.
It requires the **BF16 Marlin activation path**, including plain NVFP4 recipes
when that consumer is selected. It does not implement native W4A4 arithmetic.
The revised BF16/head/scale arm saved **10.682 GiB across two GPUs**. Against
a refreshed stock control, single-request decode ranged from −1.1% to +8.6%
and concurrent throughput from +0.6% to +6.3%. The original comparison included
a 16.4% slowdown, and the repeated stock control itself varied by up to 11.2%.
These screens do not establish a reliable general speedup. The adapters stay
opt-in; no quality-equivalence or production-throughput claim follows.
See the [matched stock-runtime results](#stock-vllm--dflash2-results-2026-10-26)
and [NVFP4 controls](#experimental-nvfp4-scale-controls).

## Stock vLLM + DFlash2 results, 2026-10-26

In the initial screen, each checkpoint was compared against itself with extra
ZipServ packing off/on. The revised NVIDIA run is reported separately below.
Both used stock vLLM 0.32.0, BF16 Marlin kernels, TP2 on two GB10s, the same
five-layer DFlash2 drafter, decode CUDA graphs, 32K context and a fixed 4 GiB
KV pool per GPU. FP4 weight codes were unchanged and FP8 packing was off.
These are separate from the production CT runtime comparisons below.

| Checkpoint | Off, total live CUDA allocation | On, total | Saved, total | Serving result |
|---|---:|---:|---:|---|
| CT MXFP4 | 178.996 GiB | 168.736 GiB | 10.260 GiB | 8/10 single cases faster; concurrency mixed |
| Plain NVIDIA NVFP4 | 188.667 GiB | 178.334 GiB | 10.332 GiB | Single cases −3.0% to +4.1%; concurrency −16.4% to +3.3% |

The NVIDIA saving is about **5.17 GiB per GPU (5.48%)**. All 93 requests
completed without errors or preemptions, before/after byte and ownership
audits passed, and live allocation was unchanged after timing. Each arm used
one model load and three repetitions; small differences are not established
speedups. No GSM8K quality result is attached to this NVIDIA comparison.

NVIDIA coverage was 264 BF16 linears and 83/84 NVFP4 scale groups per rank.
One scale group stayed raw because savings were below the configured 1% floor.
The NVIDIA head stayed dense in that original run: its `ParallelLMHead` uses an
`UnquantizedLinearMethod`, which the previous mixed adapter did not qualify
for heads. The current adapter adds that method to the head path described
below. Embeddings, routers, other raw-weight readers and FP4 codes stayed
stock. BF16 payload reduction was approximately 1.923 GiB per rank. NVFP4
scales saved approximately 3.210/3.213 GiB after 24 MiB of scratch per rank.
These components describe the allocation reduction; they are not additional
savings to add to the totals above.

NVIDIA concurrent aggregate throughput, three rounds per level:

| Requests | Packing off, tokens/s | Packing on, tokens/s | Change | Off/on TTFT p90, seconds |
|---|---:|---:|---:|---:|
| 1 | 50.22 | 48.82 | −2.8% | 0.54 / 0.56 |
| 2 | 39.42 | 38.85 | −1.4% | 1.79 / 1.82 |
| 3 | 48.33 | 40.39 | −16.4% | 1.84 / 1.01 |
| 4 | 66.27 | 63.18 | −4.7% | 23.12 / 23.76 |
| 5 | 50.08 | 46.83 | −6.5% | 29.12 / 30.77 |
| 6 | 53.96 | 55.72 | +3.3% | 48.71 / 51.09 |

The fixed KV pool causes queueing above about three active short requests;
the saved weight memory was not reassigned to KV. Prompt mixes differ across
levels, so compare within a row. At C3, draft acceptance also changed from
50.38% to 46.75%. BF16 Split-K rounding and generated outputs can differ despite
exact weight reconstruction; this test does not isolate decoder cost from
changed speculation. BF16-only and scale-only comparisons remain follow-up work.

Bounded real-request Torch/CUDA traces retained graphs and DFlash2 on both
ranks. At steady C3, 12 iterations each verified 12 scheduled tokens for three
sequences. NVFP4 fused GEMMs accounted for 53–54% of summed kernel time, BF16
GEMMs/reductions 22–23%, and NCCL 6–8%. All corresponding compressed GEMMs ran
inside CUDA graphs. This identifies profiling targets, not decompression
overhead: fused-kernel time also includes matrix multiplication. The initial
C3 trace included one prefill step and was followed by a separately retained
decode-only trace. Instrumented timings are excluded from serving results.

Matched compression-off traces are now complete. For the same twelve steady-C3
steps, summed NVFP4 MoE kernel time changed from 1,045.98 to 1,048.35 ms on rank 0
and from 1,040.69 to 1,028.88 ms on rank 1 (+0.23%/−1.13%). Identified BF16
GEMM/reduction families fell from 569.63/568.92 ms to 515.81/501.01 ms, while NCCL
time increased from 103.48/106.78 ms to 120.62/158.50 ms. These are overlapping
kernel totals from short traces, not additive critical-path costs; opaque kernel
names are excluded from the BF16 classification. They do not
attribute the earlier whole-run slowdown entirely to scale decoding, nor prove
that the MoE kernels have saturated memory bandwidth. Separate BF16-only and
scale-only serving arms remain unmeasured.

The observed 77,440 × 4,096 BF16 head is 0.591 GiB per rank. Its measured
component payload saving is about 0.174 GiB per rank. This is smaller than an
estimate based on a different head shape; component timing alone does not
establish a serving speedup.

Native qualification included exact scale round trips, stock/raw/delta5 GEMM
comparisons, graphs, nondefault streams and guarded outputs. The expanded
six-shape component gate passed Compute Sanitizer memcheck with zero errors.
A separate component timing screen included a 37% slowdown for one M=1 shape;
the experimental codec is not a general speed optimization.

Completed benchmark and trace archives are retained separately on Spark2.
NVIDIA benchmark archive SHA-256 values are
`876acb70e9f61f20ca8629c97f5eab451a08e2d07a5bbf32edcb8b62c7844ac1`
(rank 0) and
`0c6e84fd1aa52e73ac9b653c59aa3082fa3d4fd9271cb029502504611cf2f11f`
(rank 1). Large evidence files and private experiment launch scripts are not
part of the source distribution.

## Experimental NVFP4 scale controls

Build against the pinned vLLM source revision in
`native/mxfp4/moe_sources.json`, using a fresh output directory:

```bash
python integrations/vllm/build_nvfp4_native.py \
  --source /path/to/pinned-vllm-source --output /path/to/new-nvfp4-build --arch 121

export VLLM_PLUGINS=zipserv_nvfp4_scales
export ZIPSERV_NVFP4_SCALES=on
export ZIPSERV_NVFP4_LIBRARY=/path/to/new-nvfp4-build/libzipserv_nvfp4_delta5.so
export ZIPSERV_NVFP4_MIN_SAVING_PERCENT=1
```

Use these with an already working ModelOpt NVFP4 model served by Marlin with
BF16 activations, compilation mode 0, and eager or decode-only CUDA graphs.
`ZIPSERV_NVFP4_SCALES=off` retains stock execution and needs no native library;
leaving the flag unset makes registration inert. Scale-only coverage does not
compress BF16 weights: the separately configured mixed adapter controls those.
The optional `zipserv_vllm.nvfp4_backend.ScaleWorker` worker extension exposes
`zipserv_nvfp4_report(verify=True)` for byte, ownership and allocation audits.

The codec preserves post-Marlin scale bytes using five-bit offsets per expert
part, retaining wide ranges raw. Its minimum-saving policy includes metadata
and allocator padding; aggregate reports also deduct shared scratch. Another
expert-scale plugin cannot own the same weights. CSF/B12X, native W4A4,
offload, expert parallelism, LoRA, weight replacement, sleep mode and
torch.compile are outside this adapter's qualified scope. XSFP4 and FP4-Local
compressed-scale checkpoints are not automatically supported by this adapter.

### Revised head and prefill serving result

The mixed adapter now recognizes untied `ParallelLMHead` instances using either
stock unquantized method. Explicitly targeted heads use Split-K 1; shared source
storage and unsupported raw-weight readers remain ineligible. On real NVIDIA
head weights with synthetic activations, the component screen saved
0.1741/0.1738 GiB per rank and beat the dense head in its 8–128-token cases.
This is component evidence, not an additional measured serving speedup.

The earlier experimental `--experiment-wide-prefill` build used a 160-token tile
for 129–160 tokens with at least 4,096 weight rows and Split-K 1.
It decoded each weight tile once for that token range. On a real 12,576 × 4,096
projection (padded to 12,608 rows), paired component timing at 147 tokens was
29–43% faster than the current fused path on the two GB10s, while still slower
than dense GEMM. Other token ranges retained the existing kernel. Native checks
covered both tile boundaries, output guards, nondefault streams and graph replay;
Compute Sanitizer reported zero errors. The tile was removed after the padding
fix eliminated its aggregate benefit; see the follow-up below.

A proposed dense fallback above 128 tokens failed its component speed gate:
full-weight reconstruction made the 147–256-token cases 19–58% slower than the
current fused path. The serving candidate therefore retained the 768-token
fallback threshold, which is unchanged by the later padding fix.

The combined serving rerun used commit `452430d`, the wider-tile native build,
and the same checkpoint, container images, server command, five-layer drafter,
decode graphs and fixed KV budget as a refreshed stock control. Both arms
completed 93 requests with no failures or preemptions. Before/after reconstruction,
ownership and coverage audits passed, and live allocation stayed unchanged:

| Allocation | Stock | Revised compression | Saved |
|---|---:|---:|---:|
| Rank 0 | 94.333 GiB | 88.992 GiB | 5.341 GiB |
| Rank 1 | 94.333 GiB | 88.993 GiB | 5.341 GiB |
| Both GPUs | 188.667 GiB | 177.985 GiB | **10.682 GiB (5.66%)** |

This is **0.350 GiB more net saving** than the original compressed run. Coverage
is now 265 BF16 modules, including the shared target/drafter head with Split-K 1,
and 83 NVFP4 scale groups per rank. The eleven separately audited BF16 drafter
weights are unchanged. Embeddings, routers, FP4 codes and FP8 weights remain
on their previous representations.

Concurrent aggregate throughput, three rounds per level:

| Requests | Fresh stock, tokens/s | Revised, tokens/s | Change vs fresh stock | Change vs original compressed | Fresh stock/revised TTFT p90, seconds |
|---|---:|---:|---:|---:|---:|
| 1 | 46.74 | 49.67 | +6.3% | +1.7% | 0.56 / 0.56 |
| 2 | 40.07 | 40.79 | +1.8% | +5.0% | 1.80 / 1.91 |
| 3 | 42.91 | 44.55 | +3.8% | +10.3% | 1.82 / 0.78 |
| 4 | 64.61 | 65.80 | +1.8% | +4.1% | 26.25 / 23.82 |
| 5 | 50.54 | 51.15 | +1.2% | +9.2% | 38.51 / 31.32 |
| 6 | 52.79 | 53.11 | +0.6% | −4.7% | 49.85 / 48.83 |

Seven of ten single-request cases improved against the fresh control; changes
ranged from −1.1% to +8.6%. These are noninterleaved screens with one load per
arm, not a statistical speed qualification. The stock C3 result alone changed
from 48.33 to 42.91 tokens/s between runs (−11.2%); the revised 44.55 lies between
them. Draft acceptance and output lengths can change with GEMM rounding.
The fixed KV budget still causes queueing at higher concurrency. No freed
memory was reassigned to KV, and this was not a GSM8K or quality-equivalence test.

Final C1 and steady-C3 traces matched the earlier scheduled token counts and
confirmed the compressed head executing inside CUDA graphs. The revised mixed
C3 trace instead scheduled 163 prefill plus eight decode tokens, versus
139 plus eight previously. Its 171-token step is outside the new 129–160-token
tile range and used the existing kernel. That prefill trace is not a matched
timing comparison: the wider tile's gain remains component evidence, and the
whole-serving changes cannot be attributed to it. Profiling left live allocation
unchanged and its observation hooks were restored afterward.

### BF16 row-padding follow-up

The mixed adapter now aligns partial output tiles by repeating existing source
rows, then discards those extra output columns as before. Zero padding could
introduce thousands of exponent exceptions into the last tile. Since the fused
kernel sizes shared memory using the maximum count across the whole matrix,
those unused rows could slow every block. This changes neither logical weight
bytes nor the CUDA kernel, and applies to any eligible partial BF16 output tile.
Aligned tensors, the head's Split-K policy and the 768-token fallback stay as
described above. Audits now report the padding mode and maximum tile counts.

On the exact 12,576 × 4,096 NVIDIA projection, repeating source rows reduced the
maximum rare count from 2,464 to 968/832 on the two GB10 ranks. At 171 tokens,
paired component timing fell from 1.153/1.105 ms to 0.799/0.763 ms, about 31%.
The screen used synthetic activations, rotating arm order, nine repetitions and
128 MiB of cache pressure before each timed call. All fused output bytes matched
the zero-padded control over 20 token counts, including the existing Split-K
decode policy. A wider 256-token kernel was also screened, but was slower on
several other shapes; it is not included in this change.

The full-model follow-up used one loaded NVIDIA NVFP4 model with stock vLLM
0.32.0, TP2 and the same DFlash2 recipe. It retained both packed representations
temporarily and switched only the affected eager-prefill weights between
requests. Decode graphs kept their captured repeated-row buffers throughout;
this comparison therefore measures prefill, not a decode-throughput change.
The same token-ID prompt, one generated token and a prefix-cache reset were
used in each pair, with alternating arm order and nine pairs per prompt size.

| Prompt tokens | Zero-pad TTFT median, ms | Repeated-row TTFT median, ms | Median paired latency change |
|---|---:|---:|---:|
| 147 | 336.0 | 332.7 | −0.3% |
| 171 | 364.3 | 375.4 | +0.6% |
| 256 | 403.2 | 385.3 | −2.8% |
| 512 | 520.8 | 509.0 | −3.7% |
| 768 | 715.2 | 673.2 | −9.4% |
| 2048 | 1369.7 | 1370.7 | +0.1% |

The percentage is the median of per-pair ratios, not the ratio of the two
median columns. **Bootstrap 95% intervals include no improvement in every
case.** These positive larger-prefill medians are insufficient to claim a
reliable end-to-end speedup. The 2048-token case uses decode-then-stock GEMM.

All 54 pairs generated the same token. Full-model log probabilities varied
even under unchanged settings, so an initial strict score-equality check was
retained as a stopped attempt. Separate observed requests compared both packed
variants on the *same live activations*: all 34 affected layers per rank were
byte-identical at 147 and 171 tokens, in both arms (272 comparisons). This is
not a model-quality benchmark. All control buffers were released afterward,
with live CUDA allocation returning exactly to its pre-comparison value.
Repacking the same weights saved another 4.152 MiB per rank in payload bytes;
the purpose of this change is lower decoder overhead, not substantial new
compression. The full 93-request serving regression passed with no failures or
preemptions. Before/after coverage, ownership and exact reconstruction audits
passed, and live CUDA allocation stayed at 177.954 GiB across both ranks.

With this padding fix, a separate paired component screen compared the old
160-token tile against the default 128-token tile on **all 265 live BF16 modules
per rank**, at 129, 147 and 160 tokens. All outputs matched. The sums of measured
component times were 0.6–4.9% slower with the wider tile across the six rank/count
combinations. These sums are not a serving critical-path measurement. The wide
tile and its build flag were removed; the head compression and padding fix stay.

Retained evidence: `nvfp4-prefill-20261026/padding-r{0,1}.json`,
`prefill-comparison.json`, `prefill-paired-initial.json`, and the exact-shard
and nearby-shape screens. They remain in separate experiment backups.

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
| Existing CT plus the new narrow BF16 add-on | 178.807 GiB | Saves **another 0.404 GiB**; new 300-question subset: 288/300 in both arms; warmed C4 throughput 2.6% lower in one pair; full GSM8K for this add-on is pending |
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
Router `GateLinear` modules retain the stock method. Startup checks cover the
scheduler token budget and graph padding; a layer that exceeds native indexing
limits stays dense with a logged reason. The forward check remains a guard.
Automatic registration warns and returns on unsupported or missing vLLM
versions, so installing the package does not break unrelated vLLM workers.

### BF16 dispatch experiments

The original adapter accepts these fields through `ZipServConfig.from_config`:

```python
options = {
    "quant_method": "zipserv",
    "dispatch": "tokens",          # fixed (default), tokens, or shape
    "split_k": 1,                  # used by fixed mode
    "large_batch_tokens": 768,     # provisional threshold, not calibrated
    "workspace_bytes": 64 << 20,   # budget for tokens/shape selection
    "decode_workspace_bytes": 256 << 20,  # cap for the decoded-weight temporary
    "partial_dtype": "bf16",       # bf16 (default) or fp32
}
```

`tokens` starts with Split-K 8 through 32 tokens, 4 through 128, and 1 above
128; it selects a smaller valid divisor when necessary. `shape` scores every
valid divisor using token tiles, SM count, K length, workspace and reduction
traffic. Both remain experiments: the initial score missed the best measured
split on several shapes, and the token policy regressed large heads.

These two modes require compilation mode 0 and either eager execution or
`FULL_DECODE_ONLY` graphs. Above the threshold they allocate a temporary dense
BF16 weight and call stock GEMM. This allocation is logged at load time; it is
not a persistent cache. Before packing, a layer whose `m*k*2` decoded bytes
exceed `decode_workspace_bytes` stays dense with a logged reason. The fallback
also checks this cap before allocating. This is a per-call decoded-weight cap,
not chunked decoding. For example, a 57,344×8,192 BF16 matrix needs 896 MiB
when decoded, so the default 256 MiB cap keeps that whole layer dense in
`tokens`/`shape` mode, including its small-batch calls. Large layers can
therefore lose compression coverage under these modes. Increasing the cap
permits that allocation; it does not make decoding incremental. The cap is
separate from Split-K scratch; it does not bound all concurrent allocations,
GEMM workspace or graph pools. It applies to the original BF16 adapter's new
dispatch modes and the CT BF16 experiment, not every experimental mixed path.
Capture sizes above the threshold keep the layer dense
at startup. Attempting the temporary fallback inside an external graph raises.

FP32 partials use the separate `zipserv_mm_fp32` bridge symbol with four-byte
workspace elements; the existing `zipserv_mm` ABI still uses two-byte BF16
workspace. Old libraries are rejected when FP32 is explicitly requested.
This changes accumulation accuracy and scratch size, not the weight codec.
The standalone interface is `linear(x, packed, split_k=4, partial_dtype="fp32")`.

Two independent build switches isolate the other kernel experiments:

```bash
python integrations/vllm/build_native.py --arch 121 --experiment-popcount --output build/popcount.so
python integrations/vllm/build_native.py --arch 121 --experiment-shmem-once --output build/shmem_once.so
```

The second initializes a worst-case limit once per kernel and device, under a
host mutex. Launches still use their actual shared-memory size. It assumes the
worker's CUDA primary context persists; restart the worker after a context reset.

On one GB10 with vLLM 0.32.0, a paired screen of four real GLM BF16 tensor shapes
found the following. Activations were synthetic; these are component results,
not whole-model serving or GSM8K results. Small batches used graph-internal
events; batches above 768 used eager execution for every arm.

| Case | Existing fixed Split-K 1 | Experimental policy | Stock dense | Finding |
| --- | ---: | ---: | ---: | --- |
| 512×4096, 1 token | 110.5 µs | 32.8 µs, token policy | 29.7 µs | Much better than the old path, still slower than dense |
| 4096×8192, 32 tokens | 329.7 µs | 296.0 µs, shape policy | 350.2 µs | About 10% lower latency than the old path |
| 4096×8192, 8192 tokens | 28.1 ms | 7.0–7.3 ms, temporary decode + stock | 6.36 ms | Prefill fallback helps substantially; it still has overhead |
| 77440×4096 head, 32 tokens | 2.34 ms | 2.62 ms, token policy | 2.91 ms | Global token thresholds can regress a working head |

Values are medians with 128 MiB cache pressure. Ratios in the evidence use
medians of paired samples and can differ from ratios of the table medians.
FP32 partials reduced relative L2 error by roughly 30–33% in the split cases,
with approximately 0–8% latency cost in this screen and twice the scratch bytes.
Popcount showed no consistent gain. One-time attribute setup reduced direct
host enqueue time about 3–4% (roughly 0.2–0.3 µs), without a consistent graph
replay speedup. None of the performance experiments changes defaults.

`tools/bench_bf16_dispatch.py` records all candidate splits, rotating paired
timings, warm/pressure results, FP32 reference errors, library hashes, exact
decode checks, guarded scratch and a two-layer graph replay. Native tests and
the new Python GPU checks passed CUDA memcheck with zero errors. The native
suite also checks the FP32 symbol directly, including float scratch contents,
guard regions, split-one behavior and graph replay (three additional cases).
Detailed
evidence is separately retained at `docs/evidence/bf16-audit-20261025/`.

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
The dictionary adapter retains descriptor ownership and rejects unknown
integer descriptors or storage aliases before either the native call or the
outer stock-GEMM fallback. Ordinary floating-point scale tensors keep the raw
fallback. CPU regression tests exercise both wrapper boundaries.

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
already prepared `/tmp/zipserv-ct-extra.*` directory or an
`experiments/zipserv-ct-extra-*` directory containing the current
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
The default token dispatch uses Split-K 8 through 32 tokens, Split-K 4 through
128, Split-K 1 through 768, and temporary decoding followed by the stock GEMM
above that. That last path has a 64 MiB decoded weight temporary per worker;
its allocation is reused by the allocator, not a persistent decoded cache.
The launcher also accepts `--dispatch shape --partial-dtype fp32` for the
new shape policy and FP32 scratch. `--workspace-bytes` bounds Split-K scratch;
`--decode-workspace-bytes` bounds the dense temporary (default 256 MiB).
This fixed-target experiment rejects a cap below its 64 MiB target size.
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

### Current BF16 dispatch on CT: 300-question check, 2026-10-25

The new run used `--dispatch shape --partial-dtype fp32` and the native
`shmem_once` build, with popcount disabled. CT's existing scale, embedding,
KDA, DFlash and graph paths remained enabled in both arms. The two GB10s ran
the same legacy CT image, 6 GiB KV per worker, 500k configured context,
32 maximum sequences and 6,919 batched tokens. This is an end-to-end test of
the experimental CT adapter, not the supported vLLM 0.32 raw-BF16 adapter.

| Metric | Existing CT | CT + new BF16 dispatch |
| --- | ---: | ---: |
| Live CUDA allocation, both GPUs | 179.210 GiB | 178.807 GiB |
| Peak Torch CUDA allocation through GSM8K, both GPUs | 180.467 GiB | 180.063 GiB |
| GSM8K answer exact match, frozen 300-question subset | 288/300 (96.0%) | 288/300 (96.0%) |
| API errors / truncated answers | 0 / 0 | 0 / 0 |
| GSM8K aggregate generated tokens/s | 83.89 | 83.11 |
| Warm fixed-output C4 aggregate tokens/s | 93.40 | 91.01 |
| Warm long-prompt first-token latency | 1.957 s | 1.987 s |

The added net saving is **413.24 MiB (0.404 GiB, 0.225%)**. The weight payload
shrinks by 413.26 MiB; allocator accounting differs slightly. Original BF16
source bits were verified on both ranks before and after execution. Four
questions became correct and four became incorrect, so equal totals do not
mean identical answers. The subset is selected once with seed 42 from the
existing hash-pinned 1,319-question protocol, using the same client, scorer,
prompts, temperature zero, 1,024-token limit and concurrency four in both arms.

The warmed speed test followed GSM8K, with six sequential repetitions per
workload and three four-request cohorts, each generating exactly 64 tokens.
No rank-zero runtime-compilation warnings occurred in either warmed pass.
The C4 point estimate is **2.6% lower** with ZipServ; cohort ranges overlap
(CT 88.46–95.55, ZipServ 90.25–92.44 tokens/s). Sequential decode-rate changes
were between -0.8% and +0.8%. One server load per arm, variable generated text
and speculative acceptance prevent claims of a general speedup, slowdown or
quality equivalence. The GSM8K throughput figure also has variable output
lengths and is not a replacement for fixed-output timing.

This establishes a small working memory saving on CT, with no demonstrated
serving-speed benefit. The fixed KV budget was deliberately retained; capacity
benefits from enlarging KV or the batch limit were not tested. Torch counters
exclude driver/NCCL allocations and host RAM. Both private test servers were
stopped after evaluation.

`tools/eval_ct_bf16_gsm8k.py` freezes the subset, audits both workers, preserves
responses and compares paired answers. Docker mount lists are compared by
unique destination, source and access mode because `docker inspect` can reorder
the list; the frozen serving clients and source files were not changed during
the run. Verified archives from both ranks and the readout are retained at
`docs/evidence/ct-bf16-gsm300-20261025/` and on Spark2.

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

The mixed BF16 path accepts `ZIPSERV_MIXED_BF16_PREFILL_TOKENS` (default 768)
and `ZIPSERV_MIXED_BF16_DECODE_WORKSPACE_BYTES` (default 1 GiB). Above the token
threshold, it allocates a temporary decoded weight and uses stock GEMM. A layer
whose padded dense weight exceeds the budget stays dense at startup; this is
not chunked decoding. Decode graph capture sizes must fit within the fused
threshold. The original raw-BF16 plugin has its separate 256 MiB default cap.
The mixed adapter reports the chosen threshold, maximum decoded bytes and
head Split-K policy per layer. Tune the threshold by measurement: decoding a
whole weight can cost more than the redundant work it avoids.

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
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Verified: native SM121 correctness and memcheck, shared-buffer CPU ownership
on Torch 2.13/2.14, Torch custom-op schema/fake behavior, graphs/streams, vLLM
0.32 TP1 linear loaders, package installation and entrypoint discovery. The
shared-buffer CUDA runtime checks also pass under Compute Sanitizer. Native reconstruction preserves every BF16 word;
floating-point GEMM outputs can differ from another backend's rounding.

Full-model TP2 evidence covers the mixed CT and NVIDIA profiles above; the raw-BF16
quantization adapter, other models and hybrid-head configurations still need
their own full-model checks. LoRA, weight refit, other GPUs and production
throughput remain unqualified. Dynamo tests use
backend=eager and are not an Inductor performance benchmark. Use the included
scripts as component checks, then compare the intended model and serving load.

Automatic registration warns and skips unsupported vLLM releases; explicitly
enabled adapters reject unreviewed versions or source profiles. PackedWeight is an in-process
trusted object produced by this encoder; it is not a validated interchange
format for arbitrary packed buffers. See package LICENSE, NOTICE and
[source-provenance.json](source-provenance.json) for the MxWave attribution.
