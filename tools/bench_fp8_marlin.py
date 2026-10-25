"""Bounded Marlin integration correctness and real-CT component measurements.

Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
No serving hooks, model generation, service restarts or remote connections.
Requires explicit raw/range libraries; an optional async library adds ZFA1.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from bench_fp8_palette import digest, measure, source_fixture
from bench_scale_dictionary import capture
from fp8_marlin import Weights, load_library


def equal_bytes(actual, expected, label):
    torch.cuda.synchronize()
    if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
        delta = (actual.float() - expected.float()).norm() if actual.dtype == torch.bfloat16 else None
        raise RuntimeError(f"{label}: byte comparison failed, L2={delta}")


def prepare(weight, scale):
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as marlin
    n, k = weight.shape
    layer = torch.nn.Module()
    layer.input_size_per_partition, layer.output_size_per_partition = k, n
    layer.register_parameter("weight", torch.nn.Parameter(weight.cuda(), requires_grad=False))
    layer.register_parameter("weight_scale", torch.nn.Parameter(scale.cuda(), requires_grad=False))
    marlin.prepare_mxfp8_layer_for_marlin(layer)
    return layer


def stock(x, layer, workspace, n, k, *, atomic=False):
    from vllm.scalar_type import scalar_types
    return torch.ops.vllm.marlin_gemm(
        a=x, b_q_weight=layer.weight, b_bias=None, b_scales=layer.weight_scale,
        a_scales=None, global_scale=None, b_zeros=None, workspace=workspace,
        b_q_type_id=scalar_types.float8_e4m3fn.id, size_m=x.shape[0], size_n=n, size_k=k,
        use_atomic_add=atomic, use_fp32_reduce=True)


def owners(layer, n, k, raw_lib, range_lib, async_lib=None, async_alignment=16, reference_async_lib=None,
           async_codec="async", reference_async_codec="async", compact_lib=None):
    result = {mode: Weights(layer.weight, layer.weight_scale, n, k,
                         raw_library=raw_lib, range_library=range_lib, compression=mode)
            for mode in ("off", "on")}
    if async_lib is not None:
        for mode, forced in (("async", False), ("async_raw", True)):
            result[mode] = Weights(layer.weight, layer.weight_scale, n, k, raw_library=raw_lib,
                range_library=async_lib, compression="on", codec=async_codec, force_raw=forced,
                async_alignment=async_alignment)
    if reference_async_lib is not None:
        if async_codec == reference_async_codec:
            # Same-format comparisons share immutable bytes, but never workspace.
            result["previous_async"] = result["async"].with_library(reference_async_lib)
            result["previous_async"].accounting["shares_payload_with"] = "async"
        else:
            # Distinct codecs MUST encode separate buffers from identical native weights.
            result["previous_async"] = Weights(layer.weight, layer.weight_scale, n, k,
                raw_library=raw_lib, range_library=reference_async_lib, compression="on",
                codec=reference_async_codec, async_alignment=async_alignment)
            for key in ("container_bytes", "source_sha256", "raw_fallback_tiles", "exception_groups"):
                if result["previous_async"].accounting[key] != result["async"].accounting[key]:
                    raise RuntimeError(f"Byte-layout comparison changed {key}")
    if compact_lib is not None:
        eligible = result["async"].accounting["raw_fallback_tiles"] == 0
        result["compact_only"] = (result["async"].with_library(compact_lib) if eligible
                                   else result["off"].with_library(raw_lib))
        result["compact_only"].accounting.update(compact_only_eligible=eligible,
            shares_payload_with="async" if eligible else "off")
        if not eligible:
            result["compact_only"].accounting["fallback_reason"] = "compact-only kernel cannot consume raw tiles"
    return result


def gemm_checks(layer, encoded, n, k):
    results = []
    for m in (1, 8, 9, 16, 17, 32, 48, 64, 65, 128):
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * .125
        guards = {mode: torch.full((m * n * 2 + 128,), 0xa5, dtype=torch.uint8, device="cuda")
                  for mode in encoded}
        bound = {mode: item.bind(x, output=guards[mode][64:-64].view(torch.bfloat16).reshape(m, n))
                 for mode, item in encoded.items()}
        reference = stock(x, layer, bound["off"].locks, n, k).clone()
        equal_bytes(stock(x, layer, bound["off"].locks, n, k), reference, "Stock repeatability")
        for mode, run in bound.items():
            equal_bytes(run(), reference, f"Native {mode} vs stock")
            graph = capture(run)
            before = torch.cuda.memory_allocated()
            for _ in range(3):
                graph.replay()
            equal_bytes(run.output, reference, f"Native {mode} graph vs stock")
            if torch.cuda.memory_allocated() != before:
                raise RuntimeError("GEMM graph grew CUDA allocations")
            guard = guards[mode]
            if not bool((guard[:64] == 0xa5).all() & (guard[-64:] == 0xa5).all()):
                raise RuntimeError("GEMM overwrote output guard bytes")
            del graph
        results.append({"shape_mnk": [m, n, k], "exact_stock_raw_range_graphs": True,
                        "checked_variants": list(encoded),
                        "atomic_add": False, "graph_growth_bytes": 0, "guards": True})
        del bound, reference, x, run, guards, guard
    return results


def checks(raw_lib, range_lib, async_lib=None, async_alignment=16, async_codec="async", compact_lib=None):
    """Every byte pattern, guards, streams, graph replay, reduction and shape tails."""
    rng = np.random.default_rng(181)
    mixed = bytearray(b"\x38\xb9\x42\xc3" * 2048)
    for block in (1, 62, 63, 64, 65, 127, 128, 511):
        mixed[block * 16:block * 16 + 16] = bytes(range(0, 256, 16))
    patterns = [bytes(range(256)) * 32, rng.bytes(8192), b"\0" * 8192,
                b"\x7f\xff" * 4096, bytes(mixed), b"\x38" * 4096 + rng.bytes(4096)]
    results = []
    for i, raw in enumerate(patterns):
        n, k = 64, len(raw) // 64
        weight = torch.from_numpy(np.frombuffer(raw, dtype="<i4").copy()).reshape(k // 16, n * 4).cuda()
        scales = torch.full((k // 32, n), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
        formats = [("range16", range_lib, False)]
        if async_lib is not None:
            formats += [(async_codec, async_lib, False), (async_codec, async_lib, True)]
        for codec, library, forced in formats:
            encoded = Weights(weight, scales, n, k, raw_library=raw_lib,
                range_library=library, compression="on", codec=codec, force_raw=forced,
                async_alignment=async_alignment if codec in ("async", "byteplane") else 16)
            guard = torch.full((len(raw) + 128,), 0xa5, dtype=torch.uint8, device="cuda")
            decode = encoded.bind_decode(guard[64:-64])
            graph = capture(decode)
            before = torch.cuda.memory_allocated()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            if torch.cuda.memory_allocated() != before:
                raise RuntimeError("Shared-copy graph grew CUDA allocations")
            equal_bytes(guard[64:-64], weight.view(torch.uint8).reshape(-1), "Shared-copy decoder")
            if not bool((guard[:64] == 0xa5).all() & (guard[-64:] == 0xa5).all()):
                raise RuntimeError("Shared-copy decoder overwrote guard bytes")
            results.append({"decode_pattern": i, "codec": codec, "force_raw": forced,
                "exact": True, "guards": True, "nondefault_stream": True, "graph_growth_bytes": 0})
            del graph, decode, encoded, guard
        del weight, scales
    if range_lib.zipserv_marlin_fp8_decode(None, None, 2048, None) == 0:
        raise RuntimeError("Native decoder accepted null buffers")
    if async_lib is not None and async_lib.zipserv_marlin_fp8_decode(None, None, 1024, None) == 0:
        raise RuntimeError("Async decoder accepted null buffers")

    for n, k in ((64, 128), (128, 64), (192, 256), (256, 192), (384, 256), (256, 1024)):
        # Finite FP8 values with both compressible and raw groups.
        values = ((rng.integers(8, 16, size=(n, k), dtype=np.uint8) << 3)
                  | rng.integers(0, 8, size=(n, k), dtype=np.uint8)
                  | (rng.integers(0, 2, size=(n, k), dtype=np.uint8) << 7))
        values[(values & 127) == 127] &= 254  # Keep GEMM fixtures finite.
        values.ravel()[::127] = 0
        values.ravel()[63::251] = 0x78
        weight = torch.from_numpy(values).view(torch.float8_e4m3fn)
        scales = torch.full((n, k // 32), 120, dtype=torch.uint8)
        layer = prepare(weight, scales)
        # Force raw and compact tiles to coexist in the actual Marlin layout.
        # Values stay finite, and stock consumes this identical modified weight.
        tail = rng.integers(0, 256, size=n * k // 4, dtype=np.uint8)
        tail[(tail & 127) == 127] = 0
        layer.weight.view(torch.uint8).reshape(-1)[-len(tail):].copy_(torch.from_numpy(tail))
        encoded = owners(layer, n, k, raw_lib, range_lib, async_lib, async_alignment, async_codec=async_codec)
        if async_lib is not None and not 0 < encoded["async"].accounting["raw_fallback_tiles"] < encoded["async"].accounting["tiles"]:
            raise RuntimeError("Synthetic GEMM fixture must exercise both raw and compressed async tiles")
        automatic = Weights(layer.weight, layer.weight_scale, n, k, raw_library=raw_lib,
                            range_library=range_lib, compression="auto", min_saving_pct=100)
        if automatic.compressed or automatic.data is not layer.weight:
            raise RuntimeError("Auto mode failed to retain original bytes below threshold")
        del automatic
        results.extend(gemm_checks(layer, encoded, n, k))
        del encoded, layer, weight, scales
    if compact_lib is not None:
        results.extend(compact_checks(raw_lib, range_lib, async_lib, compact_lib, async_alignment, async_codec))
    gc.collect()
    torch.cuda.empty_cache()
    return results


def compact_checks(raw_lib, range_lib, async_lib, compact_lib, alignment, codec):
    """Exercise the new accepted domain and rejection before any unsafe launch."""
    results = []
    rng = np.random.default_rng(188)
    n, k = 64, 128
    scales = torch.full((k // 32, n), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    for common in (0, 1):
        raw = (rng.integers(0, 256, 8192, dtype=np.uint8) & 0xbf) | (common << 6)
        max_groups = 40 if alignment == 32 else 48
        for tile, groups in enumerate((0, 1, 8, 9, 16, 24, 32, max_groups)):
            raw[tile * 1024:tile * 1024 + groups * 16:16] ^= 64
        weight = torch.from_numpy(raw.copy().view("<i4")).reshape(k // 16, n * 4).cuda()
        owner = Weights(weight, scales, n, k, raw_library=raw_lib, range_library=compact_lib,
                        compression="on", codec=codec, async_alignment=alignment)
        if owner.accounting["raw_fallback_tiles"] or owner.library is not compact_lib:
            raise RuntimeError("Compact decode fixture did not reach the specialized kernel")
        guard = torch.full((8192 + 128,), 0xa5, dtype=torch.uint8, device="cuda")
        decode = owner.bind_decode(guard[64:-64])
        graph = capture(decode)
        before = torch.cuda.memory_allocated()
        for _ in range(3): graph.replay()
        torch.cuda.synchronize()
        equal_bytes(guard[64:-64], weight.view(torch.uint8).reshape(-1), "Compact shared-copy bytes")
        if torch.cuda.memory_allocated() != before or not bool((guard[:64] == 0xa5).all() & (guard[-64:] == 0xa5).all()):
            raise RuntimeError("Compact shared-copy guard or allocation growth failure")
        results.append({"compact_decode_common_bit6": common, "exact": True,
                        "guards": True, "nondefault_stream": True, "graph_growth_bytes": 0})
        del owner, graph, decode, guard, weight
    for pattern, forced in (("single_raw_tile", False), ("all_raw", False), ("forced_raw", True)):
        raw = np.full(8192, 0x38, dtype=np.uint8)
        if pattern == "single_raw_tile": raw[-1024:] = rng.integers(0, 256, 1024, dtype=np.uint8)
        if pattern == "all_raw": raw[:] = rng.integers(0, 256, 8192, dtype=np.uint8)
        weight = torch.from_numpy(raw.copy().view("<i4")).reshape(k // 16, n * 4).cuda()
        general = Weights(weight, scales, n, k, raw_library=raw_lib, range_library=async_lib,
                          compression="on", codec=codec, force_raw=forced, async_alignment=alignment)
        if not general.accounting["raw_fallback_tiles"]: raise RuntimeError("Missing raw tile in guard fixture")
        before = torch.cuda.memory_allocated()
        for operation in (lambda: Weights(weight, scales, n, k, raw_library=raw_lib, range_library=compact_lib,
                              compression="on", codec=codec, force_raw=forced, async_alignment=alignment),
                          lambda: general.with_library(compact_lib)):
            try: operation()
            except ValueError as error:
                if "requires zero raw fallback tiles" not in str(error): raise
            else: raise RuntimeError("Compact kernel accepted a raw tile")
        if torch.cuda.memory_allocated() != before:
            raise RuntimeError("Rejected compact owner allocated a GPU payload")
        automatic = Weights(weight, scales, n, k, raw_library=raw_lib, range_library=compact_lib,
                            compression="auto", min_saving_pct=0, codec=codec, force_raw=forced,
                            async_alignment=alignment)
        if automatic.compressed or automatic.data is not weight or automatic.library is not raw_lib:
            raise RuntimeError("Ineligible compact auto mode failed to keep the raw input")
        results.append({"compact_rejection_pattern": pattern, "constructor_rejected": True,
                        "rebind_rejected": True, "no_gpu_payload_allocated": True, "auto_selected_raw": True})
        del weight, general, automatic
    del scales
    for index, (n, k) in enumerate(((64, 128), (128, 64), (192, 256), (256, 192), (384, 256), (256, 1024))):
        values = rng.integers(0, 256, (n, k), dtype=np.uint8) & 0xbf
        values |= (index % 2) << 6
        values[(values & 127) == 127] &= 254
        weight = torch.from_numpy(values).view(torch.float8_e4m3fn)
        scales = torch.full((n, k // 32), 120, dtype=torch.uint8)
        layer = prepare(weight, scales)
        # One finite correction per native tile, not enough to need raw fallback.
        layer.weight.view(torch.uint8).reshape(-1)[::1024] = 0 if index % 2 else 64
        encoded = owners(layer, n, k, raw_lib, range_lib, async_lib, alignment,
                         async_codec=codec, compact_lib=compact_lib)
        candidate = encoded["compact_only"]
        if candidate.library is not compact_lib or candidate.data is not encoded["async"].data:
            raise RuntimeError("Compact GEMM fixture did not share the specialized payload")
        for item in gemm_checks(layer, encoded, n, k):
            item["fixture"] = "compact-only, same immutable payload as mixed kernel"
            results.append(item)
        del encoded, candidate, layer, weight, scales
    return results


def experiment(raw_lib, range_lib, source, layer_index, rank, family, pressure, tokens, async_lib=None, async_alignment=16,
               reference_async_lib=None, async_codec="async", reference_async_codec="async", compact_lib=None):
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as marlin
    weight, scale, sources = source_fixture(source, layer_index, rank, family)
    n, k = weight.shape
    layer = prepare(weight, scale)
    encoded = owners(layer, n, k, raw_lib, range_lib, async_lib, async_alignment, reference_async_lib,
                     async_codec, reference_async_codec, compact_lib)
    # Debug reconstruction is verified and freed before GEMM timing; the candidate
    # has no weight-sized decode scratch and launches only the Marlin GEMM.
    decoded = torch.empty(n * k, dtype=torch.uint8, device="cuda")
    for mode, owner in encoded.items():
        if not owner.compressed: continue
        decode = owner.bind_decode(decoded)
        decode()
        equal_bytes(decoded, layer.weight.view(torch.uint8).reshape(-1), f"Real native weight bytes ({mode})")
    del decode, decoded
    result = {"layer": layer_index, "rank": rank, "family": family, "shape_nk": [n, k],
              "source": sources, "runtime_weight_sha256": digest(layer.weight),
              "runtime_scale_sha256": digest(layer.weight_scale),
              "compression": encoded["on"].accounting, "exact_gpu_runtime_bytes": True,
              "encoding_accounts": {key: owner.accounting for key, owner in encoded.items()},
              "unchanged_scale_bytes": layer.weight_scale.nbytes, "measurements": []}
    for m in tokens:
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * .125
        bound = {mode: item.bind(x) for mode, item in encoded.items()}
        reference = stock(x, layer, bound["off"].locks, n, k).clone()
        equal_bytes(stock(x, layer, bound["off"].locks, n, k), reference, "Stock self-repeat")
        for mode, run in bound.items():
            equal_bytes(run(), reference, f"Real native {mode} vs stock")
        def production():
            return marlin.apply_mxfp8_marlin_linear(x, layer.weight, layer.weight_scale, None, n, k)
        default_output = production()
        relative = float((default_output.float() - reference.float()).norm()
                         / reference.float().norm().clamp_min(1e-30))
        if not bool(torch.isfinite(default_output).all()) or relative > .01:
            raise RuntimeError("Controlled reduction disagrees with default vLLM excessively")
        operations = {"baseline": lambda: stock(x, layer, bound["off"].locks, n, k),
                      "stock_default_policy": production, "isolated_raw": bound["off"],
                      "integrated_range16": bound["on"]}
        if async_lib is not None:
            operations.update(integrated_async=bound["async"], async_raw_control=bound["async_raw"])
        if reference_async_lib is not None:
            operations["previous_async"] = bound["previous_async"]
        if compact_lib is not None:
            operations["compact_only"] = bound["compact_only"]
        timing = measure(operations, pressure)
        for mode in timing.values():
            samples = mode["samples_us"]
            mode["range_vs_isolated_raw_paired_median_pct"] = statistics.median(
                100 * (a / b - 1) for a, b in zip(samples["integrated_range16"], samples["isolated_raw"]))
            mode["range_vs_default_stock_paired_median_pct"] = statistics.median(
                100 * (a / b - 1) for a, b in zip(samples["integrated_range16"], samples["stock_default_policy"]))
            if async_lib is not None:
                mode["async_vs_range_paired_median_pct"] = statistics.median(
                    100 * (a / b - 1) for a, b in zip(samples["integrated_async"], samples["integrated_range16"]))
            if reference_async_lib is not None:
                mode["async_vs_previous_paired_median_pct"] = statistics.median(
                    100 * (a / b - 1) for a, b in zip(samples["integrated_async"], samples["previous_async"]))
            if compact_lib is not None:
                mode["compact_vs_mixed_paired_median_pct"] = statistics.median(
                    100 * (a / b - 1) for a, b in zip(samples["compact_only"], samples["integrated_async"]))
                mode["compact_vs_raw_paired_median_pct"] = statistics.median(
                    100 * (a / b - 1) for a, b in zip(samples["compact_only"], samples["isolated_raw"]))
        result["measurements"].append({"tokens": m, "exact_to_stock_control": True,
            "control_atomic_add": False, "default_policy_relative_l2": relative,
            "input_sha256": digest(x), "workspace_bytes_each": bound["on"].workspace_bytes,
            "timing": timing})
        del bound, operations, x, reference, default_output, run
    return result


def profile(raw_lib, range_lib, source, case, m, arm, async_lib=None, async_alignment=16, async_codec="async", compact_lib=None):
    """One NVTX-marked operation for external Nsight application replay."""
    layer_index, rank, family = case.split(":")
    weight, scale, sources = source_fixture(source, int(layer_index), int(rank), family)
    n, k = weight.shape
    layer = prepare(weight, scale)
    encoded = owners(layer, n, k, raw_lib, range_lib, async_lib, async_alignment, async_codec=async_codec, compact_lib=compact_lib)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * .125
    bound = {mode: owner.bind(x) for mode, owner in encoded.items()}
    reference = stock(x, layer, bound["off"].locks, n, k).clone()
    for run in bound.values():
        equal_bytes(run(), reference, "Profiler input correctness")
    operations = {"stock": lambda: stock(x, layer, bound["off"].locks, n, k),
                  "raw": bound["off"], "range16": bound["on"]}
    if async_lib is not None:
        operations.update({"async": bound["async"], "async_raw": bound["async_raw"]})
    if compact_lib is not None:
        operations["compact_only"] = bound["compact_only"]
    for _ in range(3):
        operations[arm]()
    pressure = torch.empty(128 * 2**20, dtype=torch.uint8, device="cuda")
    pressure.fill_(17)
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("zipserv_fp8_marlin"):
        output = operations[arm]()
        # Keep an NVTX-triggered Systems capture open until GPU work completes.
        torch.cuda.synchronize()
    equal_bytes(output, reference, "Profiled output")
    return {"arm": arm, "shape_mnk": [m, n, k], "source": sources,
            "runtime_weight_sha256": digest(layer.weight), "runtime_scale_sha256": digest(layer.weight_scale),
            "input_sha256": digest(x), "output_sha256": digest(output),
            "exact_to_stock_control": True, "control_atomic_add": False,
            "compression": encoded["on"].accounting,
            "encoding_accounts": {key: owner.accounting for key, owner in encoded.items()},
            "range": "zipserv_fp8_marlin", "timing": "use unprofiled paired benchmark only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-library", type=Path, required=True)
    parser.add_argument("--range-library", type=Path, required=True)
    parser.add_argument("--async-library", type=Path)
    parser.add_argument("--compact-only-library", type=Path,
                        help="ABI 2 producer for tensors with zero raw tiles; other tensors use raw")
    parser.add_argument("--reference-async-library", type=Path,
                        help="Optional previous decoder; different codecs get separate encodings of identical weights")
    parser.add_argument("--async-codec", choices=("async", "byteplane"), default="async")
    parser.add_argument("--reference-async-codec", choices=("async", "byteplane"), default="async")
    parser.add_argument("--async-alignment", type=int, choices=(16, 32), default=16)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--profile-arm", choices=("stock", "raw", "range16", "async", "async_raw", "compact_only"))
    parser.add_argument("--cases", nargs="+", default=["0:0:gate_up", "3:0:gate_up", "44:1:down"])
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 32, 128])
    args = parser.parse_args()
    # Nsight's application replay runs the same explicit profiling command more
    # than once. Ordinary correctness/timing reports must remain immutable.
    if args.report.exists() and not args.profile_arm:
        parser.error("Preserve previous evidence; choose a fresh report")
    if args.compact_only_library and args.async_library is None:
        parser.error("--compact-only-library requires the general --async-library control")
    if args.profile_arm == "compact_only" and args.compact_only_library is None:
        parser.error("Compact profiles require --compact-only-library")
    if args.reference_async_library and args.async_library is None:
        parser.error("A previous decoder comparison requires --async-library")
    if not args.checks_only and args.source is None:
        parser.error("--source is required for real-model component measurements")
    if any(m < 1 or m > 128 for m in args.tokens):
        parser.error("--tokens must be in 1..128")
    for case in args.cases:
        try:
            layer, rank, family = case.split(":")
            valid = int(layer) >= 0 and int(rank) in (0, 1) and family in ("gate_up", "down")
        except (ValueError, TypeError):
            valid = False
        if not valid:
            parser.error("Cases must be layer:rank:gate_up or layer:rank:down with rank 0/1")
    if args.profile_arm and (args.checks_only or len(args.cases) != 1 or len(args.tokens) != 1):
        parser.error("--profile-arm needs exactly one case/token count and no --checks-only")
    if args.profile_arm in ("async", "async_raw") and args.async_library is None:
        parser.error("Async profiles require --async-library")
    import vllm
    if vllm.__version__.split("+")[0] != "0.32.0":
        raise RuntimeError("This experiment is pinned to vLLM 0.32.0")
    pin = json.loads((Path(__file__).parent / "marlin_fp8_sources.json").read_text())
    package_root = Path(inspect.getfile(vllm)).parent.parent
    for relative, expected in pin["runtime_helpers_sha256"].items():
        if hashlib.sha256((package_root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Installed vLLM layout helper differs from the source pin: {relative}")
    torch.cuda.set_per_process_memory_fraction(min(1.0, (1.5 * 2**30)
        / torch.cuda.get_device_properties(0).total_memory))
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(181)
    raw_lib = load_library(args.raw_library, compressed=False)
    range_lib = load_library(args.range_library, compressed=True)
    async_lib = load_library(args.async_library, compressed=True, codec=args.async_codec) if args.async_library else None
    previous_lib = load_library(args.reference_async_library, compressed=True, codec=args.reference_async_codec) if args.reference_async_library else None
    compact_lib = load_library(args.compact_only_library, compressed=True, codec=args.async_codec) if args.compact_only_library else None
    if async_lib is not None and async_lib.zipserv_compact_only:
        parser.error("Use --compact-only-library for specialized kernels; --async-library must handle raw tiles")
    if compact_lib is not None and not compact_lib.zipserv_compact_only:
        parser.error("--compact-only-library requires an ABI 2 compact-only kernel")
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as marlin
    report = {"scope": "isolated component; not serving or full-model quality qualification",
        "vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "allocation_limit_bytes": int(1.5 * 2**30),
        "retained_controls": "raw stock and compressed copies coexist for paired timing; peak is not deployment RAM",
        "libraries_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.raw_library, args.range_library, args.async_library, args.reference_async_library, args.compact_only_library) if p is not None},
        "marlin_helper_sha256": hashlib.sha256(Path(inspect.getfile(marlin)).read_bytes()).hexdigest(),
        "experiment_sources_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("bench_fp8_marlin.py", "fp8_marlin.py", "fp8_range.py", "fp8_palette.py",
                         "fp8_async.py", "fp8_byteplane.py", "bench_fp8_palette.py", "bench_scale_dictionary.py", "marlin_fp8_sources.json")},
        "async_alignment": args.async_alignment, "async_codec": args.async_codec,
        "reference_async_codec": args.reference_async_codec,
        "compact_only_enabled": compact_lib is not None,
        "checks": [] if args.profile_arm else checks(raw_lib, range_lib, async_lib, args.async_alignment, args.async_codec, compact_lib), "experiments": []}
    if args.profile_arm:
        report["profile"] = profile(raw_lib, range_lib, args.source, args.cases[0],
                                    args.tokens[0], args.profile_arm, async_lib, args.async_alignment, args.async_codec, compact_lib)
    elif not args.checks_only:
        pressure = torch.empty(128 * 2**20, dtype=torch.uint8, device="cuda")
        for case in args.cases:
            layer, rank, family = case.split(":")
            report["experiments"].append(experiment(raw_lib, range_lib, args.source,
                int(layer), int(rank), family, pressure, args.tokens, async_lib, args.async_alignment, previous_lib,
                args.async_codec, args.reference_async_codec, compact_lib))
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            gc.collect()
            torch.cuda.empty_cache()
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.report), "checks": len(report["checks"]),
                      "experiments": len(report["experiments"])}))


if __name__ == "__main__":
    main()
