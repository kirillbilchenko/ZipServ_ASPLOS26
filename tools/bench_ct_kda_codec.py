# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Measure the existing byteplane codec on captured, unchanged CT KDA weights.

This is an isolated component experiment, not a serving adapter. Every captured
matrix is encoded and independently decoded on CPU. Selected matrices also
exercise the GPU decoder, legacy CT Marlin, and the new fused byteplane kernel.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time

import torch
from safetensors.torch import load_file

import fp8_byteplane
from fp8_marlin import Weights, load_library


HELPER_SHA = "72c324a9de072e60c1f126ad3d5e7f077f6a8752590435e088634f798683d12a"
VLLM_VERSION = "0.1.dev20051+g487ecf187"


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def tensor_bytes(value):
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def write_json(path, value):
    temporary = path.with_suffix(".incomplete")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_fixture(root, row):
    path = root / Path(row["path"]).name
    if digest_bytes(path.read_bytes()) != row["file_sha256"]:
        raise ValueError("Captured fixture changed: " + str(path))
    tensors = load_file(str(path), device="cpu")
    weight, scales = tensors["weight"], tensors["scale_bytes"]
    n, k = row["native_shape"]
    if (weight.dtype != torch.int32 or tuple(weight.shape) != (k // 16, n * 4)
            or scales.dtype != torch.uint8 or tuple(scales.shape) != (k // 32, n)):
        raise ValueError("Unexpected native KDA layout")
    raw = tensor_bytes(weight)
    if (digest_bytes(raw) != row["weight_sha256"]
            or digest_bytes(tensor_bytes(scales)) != row["scale_sha256"]):
        raise ValueError("Captured tensor bytes differ from the original CT audit")
    return weight, scales, raw


def scan(root, manifest, output):
    report = {"status": "running", "rank": manifest["rank"], "layers": [],
              "manifest_sha256": digest_bytes((root / "manifest.json").read_bytes()),
              "script_sha256": digest_bytes(Path(__file__).read_bytes()),
              "limits": ["Payload savings only; excludes GPU scratch and allocator effects.",
                         "No new quantization. KDA scales and code bytes reconstruct exactly."]}
    destination = output / "cpu-scan.json"
    for index, row in enumerate(manifest["layers"]):
        weight, scales, raw = load_fixture(root, row)
        start = time.perf_counter()
        encoded, accounting = fp8_byteplane.encode(raw)
        encode_seconds = time.perf_counter() - start
        start = time.perf_counter()
        if fp8_byteplane.decode(encoded) != raw:
            raise RuntimeError("Byteplane roundtrip changed CT KDA bytes")
        item = {"index": index, "name": row["name"], "native_shape": row["native_shape"],
                "logical_shape": row["logical_shape"], "scale_bytes": scales.nbytes,
                "cpu_encode_seconds": encode_seconds,
                "cpu_verify_seconds": time.perf_counter() - start,
                "exact_cpu_roundtrip": True, "compression": accounting}
        report["layers"].append(item)
        write_json(destination, report)
        print(json.dumps({"phase": "cpu", "index": index, "saving_pct": accounting["saving_pct"]}), flush=True)
        del weight, scales, raw, encoded
    report.update(status="complete", raw_weight_bytes=sum(r["weight_bytes"] for r in manifest["layers"]),
                  packed_weight_bytes=sum(r["compression"]["container_bytes"] for r in report["layers"]),
                  unchanged_scale_bytes=sum(r["scale_bytes"] for r in report["layers"]))
    report["saved_weight_bytes"] = report["raw_weight_bytes"] - report["packed_weight_bytes"]
    write_json(destination, report)
    return report


def graph_for(operation, repeats=1, pressure=None):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    stream.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        if pressure is not None:
            pressure.add_(1)
        start.record()
        for _ in range(repeats):
            operation()
        end.record()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, start, end


def timing(operations, pressure, trials=16):
    results = {}
    names = list(operations)
    for mode, repeats, evict in (("warm", 20, None), ("cache_pressure", 1, pressure)):
        graphs = {name: graph_for(operation, repeats, evict) for name, operation in operations.items()}
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
            raise RuntimeError("Captured component replay changed CUDA allocation")
        results[mode] = {"samples_us": samples,
            "median_us": {name: statistics.median(values) for name, values in samples.items()},
            "paired_median_change_pct": {name: statistics.median(
                100 * (a / b - 1) for a, b in zip(values, samples["legacy_ct"]))
                for name, values in samples.items()}, "graph_growth_bytes": 0}
    return results


def gpu_probe(root, manifest, library_path, indices, tokens, output,
              raw_library_path=None, candidate_library_paths=None):
    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as marlin
    from vllm.scalar_type import scalar_types

    helper = Path(inspect.getfile(marlin))
    if vllm.__version__ != VLLM_VERSION or digest_bytes(helper.read_bytes()) != HELPER_SHA:
        raise ValueError("This experiment requires CT's pinned legacy Marlin runtime")
    torch.cuda.set_device(0)
    torch.manual_seed(251025 + manifest["rank"])
    library = load_library(library_path, compressed=True, codec="byteplane")
    if library.zipserv_compact_only:
        raise ValueError("Require the mixed-tile decoder for arbitrary KDA matrices")
    raw_library = load_library(raw_library_path, compressed=False) if raw_library_path else None
    candidates = {name: load_library(path, compressed=True, codec="byteplane")
                  for name, path in (candidate_library_paths or {}).items()}
    pressure = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    report = {"status": "running", "rank": manifest["rank"], "cases": [],
              "helper_sha256": HELPER_SHA, "vllm_version": vllm.__version__,
              "library_sha256": digest_bytes(library_path.read_bytes()),
              "additional_libraries": {name: {"path": str(path), "sha256": digest_bytes(path.read_bytes())}
                  for name, path in ({**(candidate_library_paths or {}),
                      **({"native_raw": raw_library_path} if raw_library_path else {})}).items()},
              "device": torch.cuda.get_device_name(), "pressure_bytes": pressure.nbytes,
              "limits": ["Component timings are not end-to-end CT speed or resident memory savings.",
                         "FP32 reduction control is separate from CT's default atomic policy.",
                         "The forced-raw container isolates decoder effects; it is not unmodified stock Marlin."]}
    destination = output / "gpu-probe.json"
    write_json(destination, report)
    for index in indices:
        row = manifest["layers"][index]
        weight, scales, raw = load_fixture(root, row)
        n, k = row["native_shape"]
        logical_n, logical_k = row["logical_shape"]
        if logical_k != k:
            raise ValueError("This first component probe requires no input K padding")
        weight = weight.cuda()
        scales = scales.cuda().view(torch.float8_e8m0fnu)
        owners = {name: Weights(weight, scales, n, k, raw_library=None,
                  range_library=library, compression="on", codec="byteplane", force_raw=forced)
                  for name, forced in (("byteplane", False), ("forced_raw_container", True))}
        if raw_library is not None:
            owners["native_raw"] = Weights(weight, scales, n, k, raw_library=raw_library,
                                            range_library=None, compression="off")
        for name, candidate in candidates.items():
            owners[name] = owners["byteplane"].with_library(candidate)
        guard = torch.full((n * k + 128,), 0xa5, dtype=torch.uint8, device="cuda")
        for owner in owners.values():
            if not owner.compressed:
                continue
            owner.bind_decode(guard[64:-64])()
            torch.cuda.synchronize()
            if (tensor_bytes(guard[64:-64]) != raw
                    or not bool((guard[:64] == 0xa5).all() & (guard[-64:] == 0xa5).all())):
                raise RuntimeError("KDA GPU decoding failed exact reconstruction or guards")
        del guard, raw
        case = {"index": index, "name": row["name"], "native_shape": [n, k],
                "logical_shape": [logical_n, logical_k], "exact_gpu_bytes": True,
                "encoding": {key: owner.accounting for key, owner in owners.items()}, "tokens": []}
        report["cases"].append(case)
        write_json(destination, report)
        workspace = marlin.marlin_make_workspace_new(weight.device)
        controlled_workspace = marlin.marlin_make_workspace_new(weight.device)
        for m in tokens:
            x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * .125
            guards = {key: torch.full((m * n * 2 + 128,), 0xa5, dtype=torch.uint8, device="cuda") for key in owners}
            bound = {key: owner.bind(x, output=guards[key][64:-64].view(torch.bfloat16).reshape(m, n))
                     for key, owner in owners.items()}

            def legacy():
                return marlin.apply_mxfp8_marlin_linear(x, weight, scales, workspace, logical_n, logical_k)

            def controlled():
                return ops.marlin_gemm(a=x, c=None, b_q_weight=weight, b_bias=None, b_scales=scales,
                    a_scales=None, global_scale=None, b_zeros=None, g_idx=None, perm=None,
                    workspace=controlled_workspace, b_q_type=scalar_types.float8_e4m3fn,
                    size_m=m, size_n=n, size_k=k, use_atomic_add=False, use_fp32_reduce=True)[:, :logical_n]

            reference = controlled().clone()
            if not torch.equal(controlled().view(torch.uint8), reference.view(torch.uint8)):
                raise RuntimeError("Legacy controlled Marlin is not repeatable")
            comparisons = {}
            for key, operation in bound.items():
                actual = operation()[:, :logical_n]
                torch.cuda.synchronize()
                comparisons[key] = {"exact_legacy_control": torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)),
                    "relative_l2_legacy_control": float((actual.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-30))}
            if not torch.equal(bound["byteplane"].output.view(torch.uint8), bound["forced_raw_container"].output.view(torch.uint8)):
                raise RuntimeError("Compressed and raw tiles changed fused-kernel arithmetic")
            for name in candidates:
                if not torch.equal(bound[name].output.view(torch.uint8), bound["byteplane"].output.view(torch.uint8)):
                    raise RuntimeError(f"{name}: decoder variant changed GEMM arithmetic")
            production = legacy()
            if not bool(torch.isfinite(production).all() & torch.isfinite(reference).all()):
                raise RuntimeError("Non-finite component output")
            default_delta = float((production.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-30))
            operations = {"legacy_ct": legacy, "legacy_fp32_control": controlled, **bound}
            measurements = timing(operations, pressure)
            # Graph replay must reproduce the same candidate output, including native N padding.
            for key, operation in bound.items():
                expected = operation().clone()
                # External CUDA events must outlive every graph replay. Do not
                # bind the start event to a loop variable that will replace it.
                graph, start_event, end_event = graph_for(operation)
                before = torch.cuda.memory_allocated()
                for replay_index in range(3):
                    graph.replay()
                end_event.synchronize()
                if (not torch.equal(operation.output.view(torch.uint8), expected.view(torch.uint8))
                        or torch.cuda.memory_allocated() != before
                        or not bool((guards[key][:64] == 0xa5).all() & (guards[key][-64:] == 0xa5).all())):
                    raise RuntimeError("KDA graph output, allocation or guard check failed")
                del graph, start_event, end_event, expected
            case["tokens"].append({"m": m, "comparisons": comparisons,
                "default_legacy_relative_l2_to_control": default_delta,
                "exact_byteplane_vs_forced_raw": True, "graph_guards": True,
                "workspace_bytes_per_binding": bound["byteplane"].workspace_bytes, "timing": measurements})
            write_json(destination, report)
            print(json.dumps({"phase": "gpu", "index": index, "m": m,
                "exact_legacy_control": comparisons["byteplane"]["exact_legacy_control"],
                "cache_pressure_change_pct": measurements["cache_pressure"]["paired_median_change_pct"]["byteplane"]}), flush=True)
            del x, bound, guards, reference, actual, production, operation, operations
        del weight, scales, owners, owner, workspace, controlled_workspace
        gc.collect()
        torch.cuda.empty_cache()
    report["status"] = "complete"
    write_json(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 32, 33, 66, 67])
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--gpu-only", action="store_true",
                        help="Run selected GPU fixtures without repeating a separately recorded full CPU scan")
    parser.add_argument("--raw-library", type=Path)
    parser.add_argument("--candidate-library", action="append", default=[], metavar="NAME=PATH",
                        help="Compare a same-format decoder using the exact same immutable payload")
    args = parser.parse_args()
    candidate_paths = {}
    for value in args.candidate_library:
        name, separator, path = value.partition("=")
        if (not separator or not name.isidentifier() or not path or name in candidate_paths
                or name in ("byteplane", "forced_raw_container", "native_raw", "legacy_ct", "legacy_fp32_control")):
            parser.error("Candidate libraries need unique NAME=PATH entries with nonreserved identifiers")
        candidate_paths[name] = Path(path)
    if args.output.exists() or any(not 0 <= i < 68 for i in args.indices) or any(not 1 <= m <= 128 for m in args.tokens):
        parser.error("Choose fresh evidence output, KDA indices 0..67, and token counts 1..128")
    args.output.mkdir()
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    if manifest["status"] != "complete" or len(manifest["layers"]) != 68:
        raise ValueError("Require a complete original CT capture")
    state = {"status": "gpu_probe" if args.gpu_only else "cpu_scan", "started_epoch": time.time(),
             "cpu_scan_performed": not args.gpu_only}
    write_json(args.output / "state.json", state)
    try:
        if not args.gpu_only:
            scan(args.fixtures, manifest, args.output)
        state["status"] = "gpu_probe"
        write_json(args.output / "state.json", state)
        gpu_probe(args.fixtures, manifest, args.library, args.indices, args.tokens, args.output,
                  args.raw_library, candidate_paths)
        state.update(status="complete", finished_epoch=time.time())
    except BaseException as error:
        state.update(status="failed", error=repr(error), finished_epoch=time.time())
        raise
    finally:
        write_json(args.output / "state.json", state)


if __name__ == "__main__":
    main()
