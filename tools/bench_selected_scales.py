"""Measure device-routed FP4 scale decoding without altering CT's fused consumer.

This experiment keeps original expert addresses in one full-size scratch arena.
It reduces decoding work, not scratch capacity. Routing workloads are synthetic;
the scale bytes are complete real CT groups when --source is supplied.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from bench_fp8_palette import measure
from bench_scale_dictionary import Decoder, capture, checks as full_checks, load_decoder
from low_precision_codec import encode_container


class Selected:
    """Own stable routing/selection buffers and use the existing full-address output."""

    def __init__(self, decoder, experts, routes):
        if (not 1 <= experts <= 1024 or decoder.output.numel() % (experts * decoder.arguments[6])
                or routes.dtype != torch.int32 or not routes.is_contiguous()
                or routes.device != decoder.output.device or not 1 <= routes.numel() <= 2**24):
            raise ValueError("Invalid expert geometry or CUDA int32 routing")
        self.decoder, self.experts, self.routes = decoder, experts, routes
        self.slots = min(experts, routes.numel())
        self.selected = torch.empty(self.slots, dtype=torch.int32, device=routes.device)
        lib = decoder.library
        lib.zipserv_dictionary_select_experts.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.zipserv_dictionary_select_experts.restype = ctypes.c_int
        lib.zipserv_dictionary_decode_selected.argtypes = [ctypes.c_void_p] * 2 + \
            [ctypes.c_size_t] * 4 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_void_p]
        lib.zipserv_dictionary_decode_selected.restype = ctypes.c_int

    def select(self):
        """Deduplicate current routing on device, with no host read or allocation."""
        status = self.decoder.library.zipserv_dictionary_select_experts(
            self.routes.data_ptr(), self.routes.numel(), self.experts, self.selected.data_ptr(),
            self.slots, torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"Expert selection launch failed: {status}")

    def decode(self):
        """Decode only the selected experts into their unchanged absolute addresses."""
        status = self.decoder.library.zipserv_dictionary_decode_selected(
            *self.decoder.arguments[:7], self.selected.data_ptr(), self.slots, self.experts,
            torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"Selective decoder launch failed: {status}")

    def launch(self):
        """Select from this replay's routing and decode the required expert union."""
        self.select()
        self.decode()


def check_output(selected, raw, sentinel=0xa5):
    """Check every selected byte, all untouched experts and selection uniqueness."""
    routes = selected.routes.cpu().numpy().reshape(-1)
    ids = np.unique(routes[(routes >= 0) & (routes < selected.experts)])
    actual_ids = selected.selected.cpu().numpy()
    if not np.array_equal(actual_ids[actual_ids >= 0], ids):
        raise RuntimeError("GPU routing deduplication failed")
    expected = np.full((selected.experts, len(raw) // selected.experts), sentinel, dtype=np.uint8)
    source = np.frombuffer(raw, dtype=np.uint8).reshape(expected.shape)
    expected[ids] = source[ids]
    if selected.decoder.output.cpu().numpy().tobytes() != expected.tobytes():
        raise RuntimeError("Selected bytes or untouched experts changed")
    return len(ids)


def checks(lib):
    """Check duplicate/padded/changed routing, boundaries, guards, streams and graphs."""
    results = full_checks(lib)
    rng = np.random.default_rng(176)
    for experts in (1, 8, 33, 288, 1024):
        size = experts * 2048
        raw = bytearray(b"\x88\x78\x88\x88" * (size // 4))
        for first in range(0, size, 2048):
            raw[first:first + 128] = rng.bytes(128)
        raw = bytes(raw)
        packed, _ = encode_container(raw, "bytes", "vector4-warp", 1024)
        guard = torch.full((size + 128,), 0xa5, dtype=torch.uint8, device="cuda")
        decoder = Decoder(lib, packed, guard[64:-64])
        routes = torch.tensor([0, experts - 1, 0, -1, experts - 1, -1, 0, -1],
                              dtype=torch.int32, device="cuda")
        selected = Selected(decoder, experts, routes)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            selected.launch()
        torch.cuda.current_stream().wait_stream(stream)
        check_output(selected, raw)
        graph = capture(selected.launch)
        graph.replay()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for values in ([experts // 2] * 8, [-1] * 8, [experts - 1] * 8):
            routes.copy_(torch.tensor(values, dtype=torch.int32))
            guard.fill_(0xa5)
            graph.replay()
            check_output(selected, raw)
        if not bool((guard[:64] == 0xa5).all() & (guard[-64:] == 0xa5).all()):
            raise RuntimeError("Selective decoder overwrote guards")
        if torch.cuda.memory_allocated() != before:
            raise RuntimeError("Selective graph allocations grew")
        results.append({"experts": experts, "exact": True, "guards": True,
                        "dynamic_graph_routing": True, "graph_growth_bytes": 0,
                        "duplicate_ids": True, "empty_routes": True, "nondefault_stream": True})
        del graph, selected, decoder, guard
    return results


def real_scales(source, layer, rank, family):
    """Reproduce the existing native aligned-nibble representation of all 288 experts."""
    from safetensors import safe_open
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import mxfp4_marlin_process_scales
    config = json.loads((source / "config.json").read_text())["quantization_config"]
    groups = [g for g in config["config_groups"].values() if g.get("format") == "mxfp4-pack-quantized"]
    if config["quant_method"] != "compressed-tensors" or len(groups) != 1:
        raise ValueError("Expected explicitly declared MXFP4 source")
    scheme = groups[0]["weights"]
    if (scheme["num_bits"], scheme["group_size"], scheme["type"]) != (4, 32, "float"):
        raise ValueError("Unexpected MXFP4 scheme")
    targets = set(groups[0]["targets"])
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    n, k = (2048, 4096) if family == "w13" else (4096, 1024)
    result = np.empty((288, k // 32, n), dtype=np.uint8)
    for expert in range(288):
        values = []
        for projection in (("gate_proj", "up_proj") if family == "w13" else ("down_proj",)):
            name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
            if name not in targets:
                raise ValueError("Expert not declared MXFP4")
            key = name + ".weight_scale"
            if Path(index[key]).name != index[key]:
                raise ValueError("Unsafe shard path")
            with safe_open(source / index[key], framework="pt", device="cpu") as opened:
                value = opened.get_tensor(key)
            if value.dtype != torch.uint8 or bool((value == 255).any()):
                raise ValueError("Unexpected source scale bytes")
            value = value[rank * 1024:(rank + 1) * 1024] if family == "w13" else \
                value[:, rank * 32:(rank + 1) * 32]
            values.append(value.contiguous())
        raw = torch.cat(values).cuda()
        floating = raw.view(torch.float8_e8m0fnu).to(torch.bfloat16)
        prepared = mxfp4_marlin_process_scales(marlin_permute_scales(floating.T, k, n, 32))
        result[expert] = prepared.view(torch.uint8).cpu().numpy()
    raw_sha = hashlib.sha256(result).hexdigest()
    base = int(result.min()) & ~15
    if int(result.max()) - base > 15:
        raise ValueError("Native nibble range exceeded")
    # Public native ABI: eight deltas per uint32 in order 0,1,4,5,2,3,6,7.
    delta = result.reshape(-1, 8).astype(np.uint32) - base
    order = [0, 1, 4, 5, 2, 3, 6, 7]
    words = np.bitwise_or.reduce(delta[:, order] << (np.arange(8, dtype=np.uint32) * 4), axis=1)
    raw = words.astype("<u4").tobytes()
    return raw, {"raw_e8m0_sha256": raw_sha, "native_sha256": hashlib.sha256(raw).hexdigest(),
                 "nibble_base": base, "shape_egn": [288, k // 32, n]}


def main():
    """Run complete-group decode comparisons without loading or serving the model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference-audit", type=Path)
    args = parser.parse_args()
    if args.report.exists() or (not args.checks_only and (args.source is None or args.reference_audit is None)):
        parser.error("Use fresh evidence paths and supply source/reference audit for real tests")
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_per_process_memory_fraction(512 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    report = {"status": "running", "scope": "selective scale decoder component",
        "device": torch.cuda.get_device_name(), "cases": [],
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": ["CT's existing fused scale decoder is unchanged.",
            "Synthetic routing, not measured model expert hit rates.",
            "Original expert addresses retained in a full-size scratch buffer; no scratch saving.",
            "Decode latency is not GEMM latency or serving throughput."]}
    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(".incomplete")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.report)
    start = time.monotonic()
    save()
    try:
        lib = load_decoder(args.library)
        report["checks"] = checks(lib)
        save()
        if not args.checks_only:
            import vllm
            report["vllm"] = vllm.__version__
            report["reference_audit_sha256"] = hashlib.sha256(args.reference_audit.read_bytes()).hexdigest()
            prior = json.loads(args.reference_audit.read_text())["cases"]
            pressure = torch.zeros(128 * 2**20, dtype=torch.uint8, device="cuda")
            for family in ("w13", "w2"):
                raw, metadata = real_scales(args.source, 23, 0, family)
                reference = next(c for c in prior if c["name"] == f"l23-r0-{family}"
                                 and c["codec"] == "vector4-warp")
                if metadata["native_sha256"] != reference["native_sha256"]:
                    raise RuntimeError("Reconstructed full scale group differs from prior CT audit")
                packed, accounting = encode_container(raw, "bytes", "vector4-warp", 1024)
                decoder = Decoder(lib, packed)
                decoder.launch()
                decoder.verify()
                row = {"family": family, "layer": 23, "rank": 0, "experts": 288,
                       **metadata, "encoded": accounting, "scratch_bytes": len(raw), "routing": []}
                report["cases"].append(row)
                workloads = {"one_token_8": [0, 1, 2, 3, 143, 144, 286, 287],
                             "eight_tokens_disjoint_64": list(range(64)),
                             "thirty_six_tokens_disjoint_288": list(range(288)),
                             "128_tokens_concentrated_8": [0, 1, 2, 3, 143, 144, 286, 287] * 128}
                for name, ids in workloads.items():
                    routes = torch.tensor(ids, dtype=torch.int32, device="cuda")
                    selected = Selected(decoder, 288, routes)
                    decoder.output.fill_(0xa5)
                    selected.launch()
                    unique = check_output(selected, raw)
                    timings = measure({"baseline": decoder.launch, "control": decoder.launch,
                        "selected": selected.launch, "selection_only": selected.select}, pressure)
                    decoder.output.fill_(0xa5)
                    selected.launch()
                    check_output(selected, raw)
                    row["routing"].append({"name": name, "route_count": len(ids), "unique_experts": unique,
                        "selected_slots": selected.slots, "selection_buffer_bytes": selected.selected.numel() * 4,
                        "exact_selected_bytes": True, "untouched_unselected_bytes": True, "timings": timings})
                    save()
                    print(json.dumps({"family": family, "routing": name,
                                      "warm_us": timings["warm"]["median_us"]}), flush=True)
                del decoder, selected, routes, raw, packed
        report.update(status="passed", seconds=time.monotonic() - start,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
    except BaseException as error:
        report.update(status="failed", error=repr(error), seconds=time.monotonic() - start)
        save()
        raise


if __name__ == "__main__":
    main()
