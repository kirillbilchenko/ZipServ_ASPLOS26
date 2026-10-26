"""Bounded CUDA component checks for dictionaries of existing packed scale bytes.

This reconstructs the original representation before calling an unchanged GEMM.
It does not install a serving plugin or implement an expert-cache replacement
policy. A cached-hit arm and a forced-miss arm bound that follow-up experiment.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

from low_precision_codec import CODECS, HEADER, align, decode_container, encode_container, vector_width
from compression_policy import cache_reuse_hint, percentile


class Decoder:
    """Own a validated immutable container and an allocation-free CUDA launch."""

    def __init__(self, library: ctypes.CDLL, packed: bytes, output: torch.Tensor | None = None):
        restored = decode_container(packed)
        _, _, codec_id, streams, _, tile, table_size, size = HEADER.unpack_from(packed)
        name = CODECS[codec_id]
        if (name not in ("vector4-direct", "vector4-warp", "vector8-direct", "vector8-warp")
                or tile not in (1024, 2048) or size % tile or streams != 1):
            raise ValueError("CUDA prototype requires supported vectors and complete 1024/2048-byte tiles")
        self.library, self.packed = library, packed
        self.input = torch.from_numpy(np.frombuffer(packed, dtype=np.uint8).copy()).cuda()
        self.output = torch.empty(size, dtype=torch.uint8, device="cuda") if output is None else output
        if (self.output.device.type != "cuda" or self.output.device.index != torch.cuda.current_device()
                or self.output.dtype != torch.uint8 or not self.output.is_contiguous()
                or self.output.numel() != size or self.output.data_ptr() % 4):
            raise ValueError("Output must be a four-byte-aligned contiguous CUDA byte buffer of exactly the decoded size")
        tiles = size // tile
        meta = 64 + align(table_size, 64)
        data = meta + align((tiles + 1) * 8 + tiles * 5, 64)
        self.arguments = (self.input.data_ptr(), self.output.data_ptr(), len(packed), size,
                          meta, data, tile, vector_width(name), int(name.endswith("-warp")))
        self.expected_sha256 = hashlib.sha256(restored).hexdigest()

    def launch(self) -> None:
        """Launch on the current stream; callers retain this owner through completion."""
        status = self.library.zipserv_dictionary_decode(
            *self.arguments, torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"Dictionary CUDA launch failed: status {status}")

    def verify(self) -> None:
        """Compare every reconstructed byte with the independently decoded CPU source."""
        actual = self.output.cpu().numpy().tobytes()
        if hashlib.sha256(actual).hexdigest() != self.expected_sha256:
            raise RuntimeError("GPU dictionary reconstruction changed source bits")


def load_decoder(path: Path) -> ctypes.CDLL:
    """Load the explicit experimental library without rebuilding or installing it."""
    library = ctypes.CDLL(str(path.resolve()))
    library.zipserv_dictionary_decode.argtypes = (
        [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p])
    library.zipserv_dictionary_decode.restype = ctypes.c_int
    return library


def capture(operation):
    """Capture a warmed operation on a side stream, with all buffers already allocated."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        operation()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def checks(library: ctypes.CDLL) -> list[dict]:
    """Check all bytes, raw tiles, dictionary hits/escapes, guards, streams and graphs."""
    rng = np.random.default_rng(173)
    patterns = [bytes(range(256)) * 32, b"\x88" * 8192, rng.bytes(8192)]
    mixed = bytearray((b"\x88\x78\x88\x88" * 2048))
    for first in (0, 32, 1000, 2040, 2048, 4092, 8184):
        mixed[first:first + 8] = rng.bytes(8)
    patterns.append(bytes(mixed))
    results = []
    for kind in ("vector4-warp", "vector4-direct", "vector8-warp", "vector8-direct"):
        for tile in (1024, 2048):
            for index, raw in enumerate(patterns):
                packed, accounting = encode_container(raw, "bytes", kind, tile)
                guard = torch.full((len(raw) + 128,), 0xa5, dtype=torch.uint8, device="cuda")
                decoder = Decoder(library, packed, guard[64:-64])
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    decoder.launch()
                torch.cuda.current_stream().wait_stream(stream)
                decoder.verify()
                graph = capture(decoder.launch)
                before = torch.cuda.memory_allocated()
                for _ in range(3):
                    decoder.output.zero_()
                    graph.replay()
                torch.cuda.synchronize()
                decoder.verify()
                if torch.cuda.memory_allocated() != before or not bool(
                    (guard[:64] == 0xa5).all() & (guard[-64:] == 0xa5).all()):
                    raise RuntimeError("Decoder changed guards or grew replay allocations")
                results.append({"codec": kind, "tile_bytes": tile, "pattern": index,
                                "bytes": len(raw), "exact": True, "guards": True,
                                "nondefault_stream": True, "graph_growth_bytes": 0,
                                "raw_tiles": accounting["raw_fallback_tiles"]})
                del graph, decoder, guard
    if library.zipserv_dictionary_decode(None, None, 0, 0, 0, 0, 1024, 4, 1, None) == 0:
        raise RuntimeError("CUDA decoder accepted invalid arguments")
    packed, _ = encode_container(b"\x88" * 1024, "bytes", "vector4-warp", 1024)
    guard = torch.empty(1025, dtype=torch.uint8, device="cuda")
    try:
        Decoder(library, packed, guard[1:])
    except ValueError:
        pass
    else:
        raise RuntimeError("CUDA wrapper accepted an unaligned output")
    decoder = Decoder(library, packed, guard[:1024])
    for index in (0, 1, 4, 5):
        arguments = list(decoder.arguments)
        arguments[index] += 1
        if library.zipserv_dictionary_decode(*arguments, torch.cuda.current_stream().cuda_stream) == 0:
            raise RuntimeError("CUDA entry point accepted an unaligned pointer or offset")
    return results


def timed_graph(operation, eviction: torch.Tensor, count: int, pressure: bool):
    """Capture device timing nodes around operations, excluding host submission gaps."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
        # Initialize timing events before capture. External events retain record nodes.
        start.record()
        end.record()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        if pressure:
            eviction.add_(1)
        else:
            for _ in range(3):
                operation()
        start.record()
        for _ in range(count):
            operation()
        end.record()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, start, end


def paired(operations: dict, eviction: torch.Tensor, repeats: int) -> dict:
    """Time device-only paired arms in both orders, with an identical hit control."""
    result = {}
    names = list(operations)
    for mode in ("warm", "cache_pressure"):
        # Pressure runs measure one call, not an average dominated by warm repeats.
        count = repeats if mode == "warm" else 1
        graphs = {name: timed_graph(operation, eviction, count, mode == "cache_pressure")
                  for name, operation in operations.items()}
        # First replay can release graph-capture scalar temporaries used by add_.
        # Check steady-state storage after this initialization in every arm.
        for graph, _, end in graphs.values():
            graph.replay()
            end.synchronize()
        before = torch.cuda.memory_allocated()
        samples = {name: [] for name in names}
        for trial in range(12):
            for name in names if trial % 2 == 0 else list(reversed(names)):
                graph, start, end = graphs[name]
                graph.replay()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000 / count)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        ratios = {name: [100 * (c / b - 1) for b, c in zip(samples["baseline"], values)]
                  for name, values in samples.items() if name != "decode_only"}
        result[mode] = {"samples_us": samples, "median_us": medians, "repeats_per_sample": count,
                        "slowdown_pct": {name: 100 * (value / medians["baseline"] - 1)
                                         for name, value in medians.items() if name != "decode_only"},
                        "paired_median_slowdown_pct": {name: statistics.median(values)
                                                       for name, values in ratios.items()},
                        "paired_p95_slowdown_pct": {name: percentile(values, .95)
                                                    for name, values in ratios.items()},
                        "cache_reuse_hints": [cache_reuse_hint(medians["baseline"], medians["cache_hit"],
                                                              medians["cache_miss"], budget)
                                              for budget in (0, 3, 5)],
                        "decoder_fraction_of_baseline_pct": 100 * medians["decode_only"] / medians["baseline"]}
        after = torch.cuda.memory_allocated()
        if after != before:
            raise RuntimeError(f"Timed replay changed CUDA tensor storage in {mode}: {before} -> {after}")
        del graphs
    return result


def load_moe_support(directory: Path):
    """Load the separately supplied MxWave reference bridge with recorded provenance."""
    path = directory / "moe_runtime.py"
    spec = importlib.util.spec_from_file_location("zipserv_reference_moe_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture(source: Path, layer: int, experts: list[int]):
    """Repack a bounded selection of real MXFP4 experts through stock vLLM operations."""
    from safetensors import safe_open
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import mxfp4_marlin_process_scales
    config = json.loads((source / "config.json").read_text())["quantization_config"]
    groups = [g for g in config["config_groups"].values() if g.get("format") == "mxfp4-pack-quantized"]
    if config["quant_method"] != "compressed-tensors" or len(groups) != 1:
        raise ValueError("Expected a declared MXFP4 compressed-tensors source")
    scheme = groups[0]["weights"]
    if (scheme["num_bits"], scheme["group_size"], scheme["type"]) != (4, 32, "float"):
        raise ValueError("Unexpected weight scheme")
    targets = set(groups[0]["targets"])
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    def tensor(expert, projection, suffix):
        module = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
        if module not in targets:
            raise ValueError("Expert is not explicitly declared MXFP4")
        key = module + "." + suffix
        shard = index[key]
        if Path(shard).name != shard:
            raise ValueError("Unsafe shard path")
        with safe_open(source / shard, framework="pt", device="cpu") as opened:
            value = opened.get_tensor(key)
            if value.dtype != torch.uint8:
                raise ValueError("Expected raw U8 MXFP4 checkpoint bytes")
            # TP=2, rank 0 component shapes, with no distributed communication.
            if projection == "down_proj":
                return value[:, :1024 // (32 if suffix == "weight_scale" else 2)].contiguous()
            return value[:1024].contiguous()
    output = []
    for family in ("w13", "w2"):
        n, k = (2048, 4096) if family == "w13" else (4096, 1024)
        weight = torch.empty((len(experts), k // 16, n * 2), dtype=torch.int32, device="cuda")
        scales = torch.empty((len(experts), k // 32, n), dtype=torch.float8_e8m0fnu, device="cuda")
        for i, expert in enumerate(experts):
            def projection(suffix):
                return torch.cat([tensor(expert, p, suffix) for p in ("gate_proj", "up_proj")]) \
                    if family == "w13" else tensor(expert, "down_proj", suffix)
            raw_weight = projection("weight_packed").cuda()
            qweight = raw_weight.view(torch.int32).T.contiguous()
            weight[i].copy_(ops.gptq_marlin_repack(qweight, k, n, 4, False))
            raw_scale = projection("weight_scale").cuda()
            if bool((raw_scale == 255).any()):
                raise ValueError("Reserved E8M0 NaN in expert scales")
            floating = raw_scale.view(torch.float8_e8m0fnu).to(torch.bfloat16)
            scales[i].copy_(mxfp4_marlin_process_scales(marlin_permute_scales(floating.T, k, n, 32)))
            del raw_weight, qweight, raw_scale, floating
        output.append((family, weight, scales))
    return output


def main() -> None:
    """Run decoder checks and optional real-weight GEMM comparisons under a GPU allocation cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--moe-library", type=Path)
    parser.add_argument("--moe-support", type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--tokens", default="1,16,128,512")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--gpu-cap-mib", type=int, default=512)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Preserve previous reports; use a fresh path")
    if not args.checks_only and any(p is None for p in (args.source, args.moe_library, args.moe_support)):
        parser.error("GEMM measurements require --source, --moe-library and --moe-support")
    if not 64 <= args.gpu_cap_mib <= 1024 or args.repeats < 1:
        parser.error("Require a bounded 64..1024 MiB GPU cap and positive repeat count")
    torch.set_num_threads(2)
    torch.manual_seed(174)
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(args.gpu_cap_mib * 2**20 / props.total_memory)
    library = load_decoder(args.library)
    report = {"status": "running", "device": props.name, "torch": torch.__version__,
              "cuda": torch.version.cuda, "scope": "component", "cases": [],
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "codec_sha256": hashlib.sha256(Path(encode_container.__code__.co_filename).read_bytes()).hexdigest(),
              "timing_method": {"events": "external CUDA event nodes captured inside each graph",
                                "paired_trials": 12, "order": "alternating forward/reverse",
                                "initialization_replays_per_graph": 1,
                                "shared_buffers": "All GEMM arms share scales, input, output, scratch and locks.",
                                "hit_control": "Identical GEMM callable and addresses; differences estimate timing noise.",
                                "cache_pressure_bytes": 64 * 2**20,
                                "cache_pressure_caveat": "64 MiB write before measurement, not guaranteed complete eviction."},
              "limits": ["Decoder plus unchanged MXFP4 Marlin component, not full-model serving.",
                         "Eight real experts; concentrated synthetic routing; TP2 rank-zero shapes on one GPU.",
                         "Forced misses and predecoded hits are separate arms, not measured cache hit rates.",
                         "Materialized decoded buffers spend memory; a cross-layer/expert cache is not implemented.",
                         "Comparison peak includes compressed storage and a shared decoded control/cache buffer.",
                         "Cache hit is a negative control; apparent speedups within its variation are noise."]}
    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(".incomplete")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.report)
    save()
    started = time.monotonic()
    try:
        report["decoder_checks"] = checks(library)
        save()
        if not args.checks_only:
            import inspect
            import vllm
            from vllm import _custom_ops as ops
            report["vllm"] = vllm.__version__
            report["repack_signature"] = str(inspect.signature(ops.gptq_marlin_repack))
            report["vllm_ops_sha256"] = hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest()
            runtime = load_moe_support(args.moe_support)
            native = runtime.NativeMoe(args.moe_library)
            if native.encoding != "aligned_nibbles":
                raise ValueError("Control must be the existing aligned-nibble consumer")
            report["moe_library_sha256"] = hashlib.sha256(args.moe_library.read_bytes()).hexdigest()
            report["moe_runtime_sha256"] = hashlib.sha256((args.moe_support / "moe_runtime.py").read_bytes()).hexdigest()
            report["source_checkpoint"] = str(args.source)
            report["source_metadata_sha256"] = {
                name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
                for name in ("config.json", "model.safetensors.index.json")}
            report["layer"] = args.layer
            experts = [0, 1, 2, 3, 143, 144, 286, 287]
            report["source_experts"] = experts
            tensors = fixture(args.source, args.layer, experts)
            eviction = torch.zeros(64 * 2**20, dtype=torch.uint8, device="cuda")
            for family, weight, raw in tensors:
                control = runtime.pack_scales(raw, Path("unused"), encoding="aligned_nibbles")
                cpu = control.records.view(torch.uint8).cpu().numpy().tobytes()
                weight_hash = hashlib.sha256(weight.view(torch.uint8).cpu().numpy()).hexdigest()
                for codec in ("vector4-warp", "vector4-direct"):
                    packed, accounting = encode_container(cpu, "bytes", codec, 1024)
                    # Use the exact same consumer addresses and GEMM scratch in every arm.
                    cache = control.records
                    decoder = Decoder(library, packed, cache.view(torch.uint8).reshape(-1))
                    decoder.launch()
                    decoder.verify()
                    for tokens in map(int, args.tokens.split(",")):
                        if not 1 <= tokens <= 512:
                            raise ValueError("Token counts must stay within 1..512")
                        e, groups, n = raw.shape
                        k = groups * 32
                        topk, m = (8, tokens) if family == "w13" else (1, tokens * 8)
                        ids = torch.arange(8, dtype=torch.int32, device="cuda").repeat(tokens, 1).reshape(m, topk)
                        scores = torch.full((m, topk), 1 / 8, dtype=torch.float32, device="cuda")
                        block = 8
                        capacity = min(m * topk + e * (block - 1), m * topk * block)
                        routing = (torch.empty(capacity, dtype=torch.int32, device="cuda"),
                                   torch.empty((capacity + block - 1) // block, dtype=torch.int32, device="cuda"),
                                   torch.empty(1, dtype=torch.int32, device="cuda"))
                        ops.moe_align_block_size(ids, e, block, *routing, None)
                        torch.cuda.manual_seed(174 + tokens + (1000 if family == "w2" else 0))
                        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                        plan = native.plan(m, n, k, block, topk, e, 0)
                        a = runtime.allocate_buffers(m, n, topk, block, capacity, x.device)
                        baseline = native.bind(x, weight, control, a, routing, scores, topk, block, family == "w2", plan)
                        def miss():
                            decoder.launch()
                            baseline()
                        baseline()
                        reference = a.output.clone()
                        baseline()
                        if not (torch.equal(reference.view(torch.uint8), a.output.view(torch.uint8))
                                and bool(torch.isfinite(a.output).all())):
                            raise RuntimeError("Control self-repeat or output finiteness failed")
                        miss()
                        if not torch.equal(reference.view(torch.uint8), a.output.view(torch.uint8)):
                            raise RuntimeError("Candidate output exactness failed")
                        measured = paired({"baseline": baseline, "cache_hit": baseline,
                                           "cache_miss": miss, "decode_only": decoder.launch},
                                          eviction, args.repeats)
                        decoder.verify()
                        if (not torch.equal(reference.view(torch.uint8), a.output.view(torch.uint8))
                                or bool(torch.count_nonzero(a.locks))):
                            raise RuntimeError("Replay changed outputs or left managed locks")
                        record = {"family": family, "tokens": tokens, "codec": codec,
                                  "marlin_weight_sha256": weight_hash,
                                  "native_scale_sha256": hashlib.sha256(cpu).hexdigest(),
                                  "input_sha256": hashlib.sha256(x.view(torch.uint8).cpu().numpy()).hexdigest(),
                                  "container_sha256": hashlib.sha256(packed).hexdigest(),
                                  "plan": plan, "exact_scale_bytes": True, "exact_gemm_outputs": True,
                                  "control_self_repeat_exact": True, "timings": measured,
                                  "baseline_scale_bytes": control.retained_bytes,
                                  "compressed_scale_bytes": len(packed), "decoded_cache_bytes": cache.numel() * cache.element_size(),
                                  "descriptor_bytes": control.descriptor.numel() * control.descriptor.element_size(),
                                  "cpu_container": accounting}
                        report["cases"].append(record)
                        save()
                        print(json.dumps({"family": family, "tokens": tokens, "codec": codec,
                                          "warm_us": measured["warm"]["median_us"],
                                          "cache_pressure_us": measured["cache_pressure"]["median_us"]}), flush=True)
                        del baseline, a, x, routing, ids, scores, reference
                    del decoder, cache
                del control
            del tensors, eviction
        report.update(status="passed", seconds=time.monotonic() - started,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
    except BaseException as error:
        report.update(status="failed", error=repr(error), seconds=time.monotonic() - started)
        save()
        raise


if __name__ == "__main__":
    main()
