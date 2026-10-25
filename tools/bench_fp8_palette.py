"""Bounded real-CT FP8 audit and GPU experiments; never installs serving hooks.

The stock-layout arm decodes exactly into Marlin's existing packed byte order.
The scalar fused probe uses row-major E4M3/E8M0 and is limited to M <= 8.
Tensor Core probes consume row-major or 64x128 tile-ordered bytes. Every probe
has an identical raw-layout arithmetic control and a stock Marlin comparison.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from bench_scale_dictionary import capture, timed_graph
from compression_policy import percentile
import fp8_palette as codec
import fp8_range as range_codec


def digest(tensor: torch.Tensor) -> str:
    """Hash exact contiguous bytes without converting the represented floats."""
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def load_library(path: Path):
    """Load only the explicitly supplied experimental CUDA library."""
    lib = ctypes.CDLL(str(path.resolve()))
    lib.zipserv_fp8_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.zipserv_fp8_decode.restype = ctypes.c_int
    lib.zipserv_fp8_gemv.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    lib.zipserv_fp8_gemv.restype = ctypes.c_int
    lib.zipserv_fp8_mma.argtypes = lib.zipserv_fp8_gemv.argtypes
    lib.zipserv_fp8_mma.restype = ctypes.c_int
    lib.zipserv_fp8_mma_tiled.argtypes = lib.zipserv_fp8_gemv.argtypes
    lib.zipserv_fp8_mma_tiled.restype = ctypes.c_int
    return lib


class Encoded:
    """Own validated immutable device input; the caller owns output through completion."""

    def __init__(self, lib, raw: bytes, bits: int, tile: int = 2048):
        start = time.perf_counter()
        encoder = range_codec if bits in (16, 32) else codec
        data, self.accounting = encoder.encode(raw, bits, tile)
        self.accounting["cpu_encode_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        if encoder.decode(data) != raw:
            raise RuntimeError("Independent CPU reconstruction failed")
        self.accounting["cpu_verify_seconds"] = time.perf_counter() - start
        self.data = torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy()).cuda()
        self.size, self.bits, self.tile, self.lib = len(raw), bits, tile, lib

    def bind(self, output: torch.Tensor):
        """Bind a preallocated byte output for allocation-free current-stream launches."""
        if (output.dtype != torch.uint8 or output.device != self.data.device
                or not output.is_contiguous() or output.numel() != self.size
                or output.data_ptr() % 4):
            raise ValueError("Require aligned contiguous byte output of exactly the source size")
        def run():
            status = self.lib.zipserv_fp8_decode(self.data.data_ptr(), output.data_ptr(),
                self.size, self.bits, self.tile, torch.cuda.current_stream().cuda_stream)
            if status:
                raise RuntimeError(f"FP8 decode launch failed: {status}")
        return run


def gemv(lib, weight, scales, x, output, bits=0, tile=2048, tensor_cores=False, tiled=False):
    """Bind the row-major probe, retaining all arguments for graph/stream lifetime."""
    n, k = scales.shape[0], x.shape[1]
    if (weight.device != x.device or scales.device != x.device or output.device != x.device
            or scales.dtype != torch.uint8 or scales.shape != (n, k // 32)
            or x.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
            or output.shape != (x.shape[0], n)
            or any(not t.is_contiguous() for t in (weight, scales, x, output))):
        raise ValueError("Invalid fused FP8 probe buffers")
    operation = lib.zipserv_fp8_mma_tiled if tiled else \
        lib.zipserv_fp8_mma if tensor_cores else lib.zipserv_fp8_gemv
    def run():
        status = operation(weight.data_ptr(), scales.data_ptr(), x.data_ptr(),
            output.data_ptr(), n, k, x.shape[0], bits, tile, torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"FP8 GEMV launch failed: {status}")
    return run


def tile_order(raw: bytes, n: int, k: int) -> bytes:
    """Reorder exact bytes to [N/64, K/128, 64, 128] for contiguous CTA loads."""
    if n % 64 or k % 128 or len(raw) != n * k:
        raise ValueError("Require complete 64x128 weight tiles")
    source = np.frombuffer(raw, dtype=np.uint8).reshape(n // 64, 64, k // 128, 128)
    result = source.transpose(0, 2, 1, 3).copy()
    if result.transpose(0, 2, 1, 3).copy().tobytes() != raw:
        raise RuntimeError("Tile-order inverse changed bits")
    return result.tobytes()


def checks(lib):
    """Exercise all patterns, mixed tiles, escapes, graphs, streams and guard bytes."""
    rng = np.random.default_rng(174)
    mixed = bytearray(b"\x38" * 4096)
    mixed[31:287] = bytes(range(256))
    mixed[1900:2156] = bytes(range(256))
    patterns = [bytes(range(256)) * 16, b"\xb8" * 4096,
                rng.bytes(4096), bytes(mixed), bytes(mixed[:2048]) + rng.bytes(2048)]
    results = []
    for bits in (2, 3, 16, 32):
        for tile in (1024, 2048):
            for i, raw in enumerate(patterns):
                encoded = Encoded(lib, raw, bits, tile)
                guard = torch.full((len(raw) + 128,), 0xa5, dtype=torch.uint8, device="cuda")
                output = guard[64:-64]
                run = encoded.bind(output)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                if output.cpu().numpy().tobytes() != raw:
                    raise RuntimeError("FP8 GPU reconstruction changed bytes")
                graph = capture(run)
                graph.replay()
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                for _ in range(3):
                    output.zero_()
                    graph.replay()
                torch.cuda.synchronize()
                if (output.cpu().numpy().tobytes() != raw
                        or not bool((guard[:64] == 0xa5).all() & (guard[-64:] == 0xa5).all())
                        or torch.cuda.memory_allocated() != before):
                    raise RuntimeError("FP8 replay exactness, guards or allocations failed")
                results.append({"bits": bits, "tile": tile, "pattern": i, "exact": True,
                                "guards": True, "nondefault_stream": True, "graph_growth_bytes": 0})
                del graph, run, encoded, guard, output
    # The arithmetic probe covers subnormals, signed zero and all finite exponents.
    codes = torch.from_numpy(rng.integers(0, 256, (4, 1024), dtype=np.uint8)).cuda()
    codes[(codes & 127) == 127] = 0
    scales = torch.full((4, 32), 119, dtype=torch.uint8, device="cuda")
    x = torch.randn((8, 1024), dtype=torch.bfloat16, device="cuda") * .125
    output = torch.empty((8, 4), dtype=torch.bfloat16, device="cuda")
    raw = gemv(lib, codes, scales, x, output)
    raw()
    reference = output.clone()
    expected = (x.float() @ (codes.view(torch.float8_e4m3fn).float() * 2**-8).T).bfloat16()
    relative = float((reference.float() - expected.float()).norm() / expected.float().norm())
    if not bool(torch.isfinite(reference).all()) or relative > .01:
        raise RuntimeError("Fused probe independent arithmetic check failed")
    for bits in (2, 3, 16, 32):
        encoded = Encoded(lib, codes.cpu().numpy().tobytes(), bits)
        run = gemv(lib, encoded.data, scales, x, output, bits)
        run()
        if not torch.equal(output.view(torch.uint8), reference.view(torch.uint8)):
            raise RuntimeError("Fused FP8 decode changed identical-control arithmetic")
        graph = capture(run)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(output.view(torch.uint8), reference.view(torch.uint8)):
            raise RuntimeError("Fused FP8 graph output changed")
        results.append({"fused_bits": bits, "exact_to_identical_raw": True,
                        "independent_relative_l2": relative})
    if lib.zipserv_fp8_decode(None, None, 0, 3, 2048, None) == 0:
        raise RuntimeError("Native decoder accepted invalid arguments")
    for k in (128, 1024):
        codes = torch.full((64, k), 0x38, dtype=torch.uint8, device="cuda")
        codes.reshape(-1)[:256] = torch.arange(256, dtype=torch.uint8, device="cuda")
        codes[(codes & 127) == 127] = 0
        scales = torch.full((64, k // 32), 119, dtype=torch.uint8, device="cuda")
        encoded = Encoded(lib, codes.cpu().numpy().tobytes(), 3)
        tiled = tile_order(codes.cpu().numpy().tobytes(), 64, k)
        tiled_codes = torch.from_numpy(np.frombuffer(tiled, dtype=np.uint8).copy()).cuda()
        tiled_encoded = Encoded(lib, tiled, 3)
        for m in (1, 7, 32, 33):
            x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * .125
            guard = torch.full((m * 64 + 128,), -7, dtype=torch.bfloat16, device="cuda")
            output = guard[64:-64].reshape(m, 64)
            raw = gemv(lib, codes, scales, x, output, tensor_cores=True)
            raw()
            reference = output.clone()
            expected = (x.float() @ (codes.view(torch.float8_e4m3fn).float() * 2**-8).T).bfloat16()
            relative = float((reference.float() - expected.float()).norm() / expected.float().norm())
            if not bool(torch.isfinite(reference).all()) or relative > .01:
                raise RuntimeError("Tensor Core independent arithmetic check failed")
            run = gemv(lib, encoded.data, scales, x, output, 3, tensor_cores=True)
            graph = capture(run)
            graph.replay()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            for _ in range(3):
                output.zero_()
                graph.replay()
            torch.cuda.synchronize()
            if (not torch.equal(output.view(torch.uint8), reference.view(torch.uint8))
                    or not bool((guard[:64] == -7).all() & (guard[-64:] == -7).all())
                    or before != torch.cuda.memory_allocated()):
                raise RuntimeError("Tensor Core graph output/guard/allocation check failed")
            results.append({"tensor_core_bits": 3, "shape_mnk": [m, 64, k],
                "exact_to_raw_probe": True, "independent_relative_l2": relative,
                "guards": True, "graph_growth_bytes": 0})
            del graph, run, raw, output, guard
            for bits, data in ((0, tiled_codes), (3, tiled_encoded.data)):
                output = torch.empty_like(reference)
                run = gemv(lib, data, scales, x, output, bits, tiled=True)
                graph = capture(run)
                graph.replay()
                torch.cuda.synchronize()
                if not torch.equal(output.view(torch.uint8), reference.view(torch.uint8)):
                    raise RuntimeError("Tiled Tensor Core bytes/arithmetic changed")
                results.append({"tiled_bits": bits, "shape_mnk": [m, 64, k],
                                "exact_to_row_major_raw_probe": True})
                del graph, run, output
        for group in (16, 32):
            candidate = Encoded(lib, tiled, group)
            output = torch.empty_like(reference)
            run = gemv(lib, candidate.data, scales, x, output, group, tiled=True)
            graph = capture(run)
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(output.view(torch.uint8), reference.view(torch.uint8)):
                raise RuntimeError("Range Tensor Core arithmetic differs from raw control")
            results.append({"range_group": group, "shape_mnk": [m, 64, k], "exact_to_raw": True})
            del graph, run, output
    return results


def measure(operations, pressure, trials=16, warm_repeats=20):
    """Rotate paired arms using external CUDA timing nodes inside captured graphs."""
    results = {}
    names = list(operations)
    for mode in ("warm", "cache_pressure"):
        repeats = warm_repeats if mode == "warm" else 1
        graphs = {name: timed_graph(operation, pressure, repeats, mode == "cache_pressure")
                  for name, operation in operations.items()}
        for graph, _, end in graphs.values():
            graph.replay()
            end.synchronize()
        before = torch.cuda.memory_allocated()
        samples = {name: [] for name in names}
        for trial in range(trials):
            order = names[trial % len(names):] + names[:trial % len(names)]
            for name in order if trial % 2 == 0 else list(reversed(order)):
                graph, start, end = graphs[name]
                graph.replay()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000 / repeats)
        if torch.cuda.memory_allocated() != before:
            raise RuntimeError("GPU replay allocation growth")
        ratios = {name: [100 * (v / b - 1) for v, b in zip(values, samples["baseline"])]
                  for name, values in samples.items()}
        results[mode] = {"samples_us": samples,
            "median_us": {name: statistics.median(v) for name, v in samples.items()},
            "paired_median_change_pct": {name: statistics.median(v) for name, v in ratios.items()},
            "paired_p95_change_pct": {name: percentile(v, .95) for name, v in ratios.items()},
            "repeats_per_sample": repeats, "graph_growth_bytes": 0}
    return results


def source_fixture(source, layer_index, rank, family):
    """Read complete TP2 shards only from explicitly declared group-32 MXFP8 modules."""
    from safetensors import safe_open
    model_config = json.loads((source / "config.json").read_text())
    config = model_config["quantization_config"]
    if config["quant_method"] != "compressed-tensors":
        raise ValueError("Expected declared compressed-tensors source")
    groups = [g for g in config["config_groups"].values() if g.get("format") == "mxfp8-quantized"]
    if len(groups) != 1 or any(groups[0]["weights"][k] != v for k, v in
                              (("num_bits", 8), ("group_size", 32), ("type", "float"))):
        raise ValueError("Expected declared group-32 E4M3/E8M0 source")
    targets = set(groups[0]["targets"])
    other_targets = {target for group in config["config_groups"].values()
                     if group is not groups[0] for target in group["targets"]}
    ignored = config.get("ignore", [])
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer_index}.mlp."
    text_config = model_config.get("text_config", model_config)
    if not 0 <= layer_index < text_config["num_hidden_layers"]:
        raise ValueError("Require a target model layer, excluding next-token prediction layers")
    if layer_index >= text_config["first_k_dense_replace"]:
        prefix += "shared_experts."
    weights, scales, sources = [], [], []
    for projection in (("gate_proj", "up_proj") if family == "gate_up" else ("down_proj",)):
        name = prefix + projection
        # This checkpoint declares a Linear fallback; explicit MXFP4 targets
        # and ignored modules take precedence. Reject patterns we cannot resolve.
        import re
        is_ignored = any(re.match(item[3:], name) if item.startswith("re:")
                         else item == name for item in ignored)
        if (is_ignored or name in other_targets
                or not (name in targets or targets == {"Linear"})):
            raise ValueError("Tensor is not explicitly declared MXFP8")
        values = []
        for suffix in ("weight", "weight_scale"):
            key = name + "." + suffix
            if Path(index[key]).name != index[key]:
                raise ValueError("Unsafe checkpoint shard path")
            with safe_open(source / index[key], framework="pt", device="cpu") as handle:
                values.append(handle.get_tensor(key))
        w, s = values
        if (w.dtype != torch.float8_e4m3fn or s.dtype != torch.uint8
                or s.shape != (w.shape[0], w.shape[1] // 32) or bool((s == 255).any())):
            raise ValueError("Unexpected FP8 source dtypes/shapes or reserved scale")
        if family == "gate_up":
            width = w.shape[0] // 2
            w, s = w[rank * width:(rank + 1) * width], s[rank * width:(rank + 1) * width]
        else:
            width = w.shape[1] // 2
            w, s = w[:, rank * width:(rank + 1) * width], \
                s[:, rank * width // 32:(rank + 1) * width // 32]
        weights.append(w.contiguous())
        scales.append(s.contiguous())
        sources.append({"name": name, "rank": rank, "shape_nk": list(w.shape),
                        "weight_sha256": digest(w), "scale_sha256": digest(s)})
    return torch.cat(weights), torch.cat(scales), sources


def experiment(lib, source, layer_index, rank, family, pressure, tokens):
    """Compare complete source and runtime layouts against unchanged stock Marlin."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as marlin
    weight, scale, sources = source_fixture(source, layer_index, rank, family)
    n, k = weight.shape
    raw = weight.view(torch.uint8).numpy().tobytes()
    row_major = weight.cuda().view(torch.uint8)
    raw_scales = scale.cuda()
    layer = torch.nn.Module()
    layer.input_size_per_partition, layer.output_size_per_partition = k, n
    layer.register_parameter("weight", torch.nn.Parameter(weight.cuda(), requires_grad=False))
    layer.register_parameter("weight_scale", torch.nn.Parameter(scale.cuda(), requires_grad=False))
    marlin.prepare_mxfp8_layer_for_marlin(layer)
    native_bytes = layer.weight.view(torch.uint8).cpu().numpy().tobytes()
    decoded = torch.empty_like(layer.weight)
    encodings = {bits: Encoded(lib, native_bytes, bits) for bits in (2, 3)}
    logical_encodings = {bits: Encoded(lib, raw, bits) for bits in (2, 3)}
    tiled_raw = tile_order(raw, n, k)
    tiled_weight = torch.from_numpy(np.frombuffer(tiled_raw, dtype=np.uint8).copy()).cuda()
    tiled_encoding = Encoded(lib, tiled_raw, 3)
    range_row = {g: Encoded(lib, raw, g) for g in (16, 32)}
    range_tiled = {g: Encoded(lib, tiled_raw, g) for g in (16, 32)}
    for value in (*range_row.values(), *range_tiled.values()):
        value.bind(decoded.view(torch.uint8).reshape(-1))()
        if digest(decoded) != value.accounting["source_sha256"]:
            raise RuntimeError("Range GPU byte reconstruction failed")
    tiled_encoding.bind(decoded.view(torch.uint8).reshape(-1))()
    if digest(decoded) != hashlib.sha256(tiled_raw).hexdigest():
        raise RuntimeError("Tiled GPU byte reconstruction failed")
    decoders = {bits: value.bind(decoded.view(torch.uint8).reshape(-1))
                for bits, value in encodings.items()}
    for run in decoders.values():
        run()
        if digest(decoded) != hashlib.sha256(native_bytes).hexdigest():
            raise RuntimeError("Actual Marlin bytes did not reconstruct exactly")
    result = {"layer": layer_index, "rank": rank, "family": family,
        "shape_nk": [n, k], "source": sources, "runtime_weight_shape": list(layer.weight.shape),
        "runtime_weight_sha256": digest(layer.weight), "runtime_scale_sha256": digest(layer.weight_scale),
        "runtime_encodings": {str(b): v.accounting for b, v in encodings.items()},
        "row_major_encodings": {str(b): v.accounting for b, v in logical_encodings.items()},
        "tensor_core_tile_encoding": tiled_encoding.accounting,
        "range_encodings": {str(g): {"row_major": range_row[g].accounting,
                                     "tiled": range_tiled[g].accounting} for g in (16, 32)},
        "unchanged_scale_bytes": layer.weight_scale.numel() * layer.weight_scale.element_size(),
        "per_layer_decoded_scratch_bytes": decoded.numel() * decoded.element_size(),
        "exact_gpu_runtime_bytes": True, "measurements": []}
    # Match the stock helper's existing optional workspace contract.
    for m in tokens:
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * .125
        def stock(w):
            return marlin.apply_mxfp8_marlin_linear(x, w, layer.weight_scale, None, n, k)
        reference = stock(layer.weight).clone()
        repeat = stock(layer.weight)
        if not torch.equal(reference.view(torch.uint8), repeat.view(torch.uint8)):
            raise RuntimeError("Stock Marlin output is not self-repeatable")
        operations = {"baseline": lambda: stock(layer.weight), "control": lambda: stock(layer.weight)}
        for bits, decode in decoders.items():
            def miss(decode=decode):
                decode()
                return stock(decoded)
            actual = miss()
            if not torch.equal(reference.view(torch.uint8), actual.view(torch.uint8)):
                raise RuntimeError("FP8 decompression changed stock GEMM outputs")
            operations[f"decode_stock_p{bits}"] = miss
            operations[f"decode_only_p{bits}"] = decode
        arithmetic = None
        if m <= 8:
            output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            fused_raw = gemv(lib, row_major, raw_scales, x, output)
            fused_raw()
            fused_reference = output.clone()
            operations["fused_raw"] = fused_raw
            for bits, value in logical_encodings.items():
                run = gemv(lib, value.data, raw_scales, x, output, bits)
                run()
                if not torch.equal(output.view(torch.uint8), fused_reference.view(torch.uint8)):
                    raise RuntimeError("Fused decode changed identical raw-probe outputs")
                operations[f"fused_p{bits}"] = run
            for group, value in range_row.items():
                run = gemv(lib, value.data, raw_scales, x, output, group)
                run()
                if not torch.equal(output.view(torch.uint8), fused_reference.view(torch.uint8)):
                    raise RuntimeError("Range scalar output differs from identical raw control")
                operations[f"fused_range{group}"] = run
            delta = fused_reference.float() - reference.float()
            relative = float(delta.norm() / reference.float().norm().clamp_min(1e-30))
            if not bool(torch.isfinite(fused_reference).all()) or relative > .01:
                raise RuntimeError("Fused probe differs excessively from stock arithmetic")
            arithmetic = {"exact_to_raw_probe": True, "relative_l2_to_stock": relative,
                          "bit_exact_to_stock": torch.equal(fused_reference.view(torch.uint8),
                                                             reference.view(torch.uint8))}
        mma_output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        mma_raw = gemv(lib, row_major, raw_scales, x, mma_output, tensor_cores=True)
        mma_raw()
        mma_reference = mma_output.clone()
        operations["mma_raw"] = mma_raw
        for bits, value in logical_encodings.items():
            run = gemv(lib, value.data, raw_scales, x, mma_output, bits, tensor_cores=True)
            run()
            if not torch.equal(mma_output.view(torch.uint8), mma_reference.view(torch.uint8)):
                raise RuntimeError("Tensor Core fused decode changed identical raw-control outputs")
            operations[f"mma_p{bits}"] = run
        for bits, data in ((0, tiled_weight), (3, tiled_encoding.data)):
            run = gemv(lib, data, raw_scales, x, mma_output, bits, tiled=True)
            run()
            if not torch.equal(mma_output.view(torch.uint8), mma_reference.view(torch.uint8)):
                raise RuntimeError("Tiled Tensor Core probe changed raw-control outputs")
            operations["mma_tiled_raw" if bits == 0 else "mma_tiled_p3"] = run
        for group, value in range_tiled.items():
            run = gemv(lib, value.data, raw_scales, x, mma_output, group, tiled=True)
            run()
            if not torch.equal(mma_output.view(torch.uint8), mma_reference.view(torch.uint8)):
                raise RuntimeError("Range tiled output differs from identical raw control")
            operations[f"mma_range{group}"] = run
        relative = float((mma_reference.float() - reference.float()).norm()
                         / reference.float().norm().clamp_min(1e-30))
        if not bool(torch.isfinite(mma_reference).all()) or relative > .01:
            raise RuntimeError("Tensor Core probe differs excessively from stock arithmetic")
        mma_arithmetic = {"exact_to_raw_probe": True, "relative_l2_to_stock": relative,
                         "bit_exact_to_stock": torch.equal(mma_reference.view(torch.uint8),
                                                            reference.view(torch.uint8))}
        timings = measure(operations, pressure)
        for run in decoders.values():
            run()
            if digest(decoded) != result["runtime_weight_sha256"]:
                raise RuntimeError("Post-timing runtime bytes changed")
        result["measurements"].append({"tokens": m, "stock_outputs_exact": True,
            "stock_self_repeat_exact": True, "fused_arithmetic": arithmetic,
            "mma_arithmetic": mma_arithmetic, "timings": timings})
        print(json.dumps({"layer": layer_index, "rank": rank, "family": family, "tokens": m,
                          "warm_us": timings["warm"]["median_us"]}), flush=True)
        del operations
    return result


def main():
    """Run under a fixed allocation cap and preserve failures as evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--cases", default="0:0:gate_up,3:0:gate_up,44:1:down")
    parser.add_argument("--tokens", default="1,8,32,128")
    args = parser.parse_args()
    if args.report.exists() or (not args.checks_only and args.source is None):
        parser.error("Use a fresh report path and supply --source for real-weight tests")
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(174)
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(1536 * 2**20 / properties.total_memory)
    report = {"status": "running", "scope": "component, not full-model serving",
        "device": properties.name, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_cap_bytes": 1536 * 2**20, "cases": [],
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "sources_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), Path(codec.__file__), Path(range_codec.__file__),
                                     Path(timed_graph.__code__.co_filename))},
        "timing": {"paired_trials": 16, "order": "rotating and alternating",
            "events": "external CUDA event nodes inside graphs", "pressure_bytes": 128 * 2**20,
            "pressure_caveat": "writes outside timed region; not guaranteed full cache eviction"},
        "limits": ["Fused row-major GEMV is a prototype, not the Marlin tensor-core kernel.",
            "The shared-tile Tensor Core probe uses synchronous WMMA and no Marlin pipeline.",
            "FP8 compression is lossless; fused arithmetic may round differently from stock.",
            "Synthetic activations; no model-quality or serving-throughput claim.",
            "Decoded Marlin buffers cost memory; encoded payload saving excludes this scratch.",
            "Both raw controls and candidates retained for comparisons; peak is not deployment RAM."]}
    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(".incomplete")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.report)
    start = time.monotonic()
    save()
    try:
        lib = load_library(args.library)
        report["checks"] = checks(lib)
        save()
        if not args.checks_only:
            import vllm
            from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as marlin
            report["vllm"] = vllm.__version__
            report["marlin_helper_sha256"] = hashlib.sha256(Path(inspect.getfile(marlin)).read_bytes()).hexdigest()
            report["source_metadata_sha256"] = {name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
                for name in ("config.json", "model.safetensors.index.json")}
            pressure = torch.zeros(128 * 2**20, dtype=torch.uint8, device="cuda")
            for case in args.cases.split(","):
                layer, rank, family = case.split(":")
                if rank not in ("0", "1") or family not in ("gate_up", "down"):
                    raise ValueError("Require TP2 rank 0/1 and gate_up/down family")
                report["cases"].append(experiment(lib, args.source, int(layer), int(rank), family,
                                                   pressure, list(map(int, args.tokens.split(",")))))
                save()
                gc.collect()
                torch.cuda.empty_cache()
        report.update(status="passed", seconds=time.monotonic() - start,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
    except BaseException as error:
        report.update(status="failed", error=repr(error), seconds=time.monotonic() - start)
        save()
        raise


if __name__ == "__main__":
    main()
