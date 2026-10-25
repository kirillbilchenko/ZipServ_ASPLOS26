"""Check and measure optional FP4 scale expansion against frozen CT and rebuilt CT.

The component keeps all FP4 weight codes, packed scales, launch plans, scratch
and matrix arithmetic identical. It does not install a vLLM serving plugin.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from bench_fp8_palette import digest, measure
from bench_scale_dictionary import capture, fixture
from build_fp4_scale_expansion import CT_LIBRARY_SHA256


def decode_checks(runtime, natives, smoke: bool) -> list[dict]:
    """Verify normal, generic and exception records, guard bytes, streams and graphs."""
    cursor = np.arange(6561, dtype=np.uint32)
    selectors = np.empty((6561, 8), dtype=np.uint8)
    for column in range(8):
        selectors[:, column] = cursor % 3
        cursor //= 3
    patterns = np.concatenate((selectors, selectors[:, ::-1]), axis=1)
    bases = [0, 1, 126, 127, 128, 252, 253] if smoke else list(range(254))
    palettes = [(base, base + 1, base + 2) for base in bases] + [(0, 127, 255), (3, 17, 192)]
    output = []
    for palette in palettes:
        # Every byte value appears in the exact exception path, including 255.
        # These decoder-only cases do not interpret reserved E8M0 values as floats.
        source = np.concatenate((np.array(palette, dtype=np.uint8)[patterns].reshape(-1),
                                 np.arange(256, dtype=np.uint8)))
        raw = torch.from_numpy(source).cuda().reshape(1, 1, -1)
        packed = runtime.pack_scales(raw.view(torch.float8_e8m0fnu), Path("unused"),
                                     encoding="sparse_palette")
        if packed.affine_palette != (palette[1] == palette[0] + 1 and palette[2] == palette[0] + 2):
            raise RuntimeError("Palette selection differs from the intended check")
        graph_check = palette[0] in (0, 1, 3, 126, 127, 128, 252, 253)
        for name, native in natives.items():
            guard = torch.full((source.size + 128,), 0xa5, dtype=torch.uint8, device="cuda")
            decoded = guard[64:-64].reshape_as(raw)

            def run():
                if native.library.mxwave_moe_decode(
                        packed.descriptor.data_ptr(), decoded.data_ptr(), decoded.numel(),
                        torch.cuda.current_stream().cuda_stream):
                    raise RuntimeError("Native scale decode failed")

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            if graph_check:
                graph = capture(run)
                graph.replay()
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                for _ in range(3):
                    decoded.zero_()
                    graph.replay()
                torch.cuda.synchronize()
                if torch.cuda.memory_allocated() != before:
                    raise RuntimeError("Decode graph grew allocations")
                del graph
            if (not torch.equal(decoded, raw) or not bool((guard[:64] == 0xa5).all())
                    or not bool((guard[-64:] == 0xa5).all())):
                raise RuntimeError(f"Scale bytes or guard changed: {name}, {palette}")
        output.append({"palette": palette, "bytes": source.size, "exact": True,
                       "exceptions": int(packed.exceptions.numel() // 16),
                       "nondefault_stream": True, "guards": True, "graph": graph_check})
    return output


def bind_case(runtime, natives, weight, scales, x, ids, multiply: bool, cpu_routing: bool = False):
    """Use one routing result, one launch plan and guarded scratch across all arms."""
    from vllm import _custom_ops as ops
    experts, groups, n = scales.shape
    m, topk = ids.shape
    k, block = groups * 32, 8
    capacity = min(m * topk + experts * (block - 1), m * topk * block)
    routing = (torch.empty(capacity, dtype=torch.int32, device="cuda"),
               torch.empty((capacity + block - 1) // block, dtype=torch.int32, device="cuda"),
               torch.empty(1, dtype=torch.int32, device="cuda"))
    if cpu_routing:
        # Independent fixture for native-kernel sanitizer isolation. The installed
        # vLLM small-batch router itself fails synccheck on the tiny check shapes.
        # Real-model timing always uses vLLM's unchanged device router.
        flat = ids.cpu().numpy().reshape(-1)
        positions, routed_experts = [], []
        for expert in range(experts):
            selected = np.flatnonzero(flat == expert).tolist()
            if selected:
                selected += [flat.size] * (-len(selected) % block)
                positions.extend(selected)
                routed_experts.extend([expert] * (len(selected) // block))
        if len(positions) > capacity:
            raise RuntimeError("CPU route fixture exceeded the stock allocation bound")
        routing[0].fill_(flat.size)
        routing[1].fill_(-1)
        routing[0][:len(positions)].copy_(torch.tensor(positions, dtype=torch.int32))
        routing[1][:len(routed_experts)].copy_(torch.tensor(routed_experts, dtype=torch.int32))
        routing[2].fill_(len(positions))
    else:
        ops.moe_align_block_size(ids, experts, block, *routing, None)
    scores = torch.full((m, topk), 1 / 8, dtype=torch.float32, device="cuda")
    buffers = runtime.allocate_buffers(m, n, topk, block, capacity, x.device)
    guard = torch.full((m * topk * n + 64,), -7, dtype=torch.bfloat16, device="cuda")
    buffers.output = guard[32:-32].reshape(m * topk, n)
    plan = natives["baseline"].plan(m, n, k, block, topk, experts, 0)
    runs = {}
    for name, native in natives.items():
        if native.plan(m, n, k, block, topk, experts, 0) != plan:
            raise RuntimeError("Control and candidate selected different plans")
        runs[name] = native.bind(x, weight, scales, buffers, routing, scores,
                                 topk, block, multiply, plan)
    return runs, buffers, guard, plan


def exact_gemms(runs, buffers, guard, graphs: bool) -> str:
    """Require bit-exact CT outputs, clean locks and untouched output guards."""
    runs["baseline"]()
    reference = buffers.output.clone()
    if not bool(torch.isfinite(reference).all()):
        raise RuntimeError("CT baseline produced nonfinite values")
    for name, run in runs.items():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        if graphs:
            graph = capture(run)
            graph.replay()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            for _ in range(3):
                buffers.output.zero_()
                graph.replay()
            torch.cuda.synchronize()
            if torch.cuda.memory_allocated() != before:
                raise RuntimeError("GEMM graph grew allocations")
            del graph
        if (not torch.equal(reference.view(torch.uint8), buffers.output.view(torch.uint8))
                or bool(torch.count_nonzero(buffers.locks))
                or not bool((guard[:32] == -7).all()) or not bool((guard[-32:] == -7).all())):
            raise RuntimeError(f"GEMM exactness, lock or guard check failed: {name}")
    return digest(reference)


def gemm_checks(runtime, natives, cpu_routing: bool = False) -> list[dict]:
    """Exercise producer synchronization with mixed escapes and several route sizes."""
    results = []
    for m, topk, n, k in ((1, 8, 256, 256), (8, 8, 256, 256),
                          (32, 1, 256, 256), (1, 1, 512, 128)):
        experts = 8
        # All E2M1 code nibbles are valid, so random prepared bytes cover their full domain.
        weight = torch.randint(-(2**31), 2**31 - 1, (experts, k // 16, n * 2),
                               dtype=torch.int32, device="cuda")
        raw = torch.randint(120, 123, (experts, k // 32, n), dtype=torch.uint8, device="cuda")
        raw.reshape(-1)[::257] = 118
        raw.reshape(-1)[::431] = 125
        packed = runtime.pack_scales(raw.view(torch.float8_e8m0fnu), Path("unused"),
                                     encoding="sparse_palette")
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * .125
        ids = (torch.arange(m * topk, dtype=torch.int32, device="cuda") % experts).reshape(m, topk)
        for multiply in (False, True):
            runs, buffers, guard, plan = bind_case(runtime, natives, weight, packed, x, ids,
                                                 multiply, cpu_routing)
            result = exact_gemms(runs, buffers, guard, graphs=True)
            results.append({"shape_mnk": [m, n, k], "topk": topk, "multiply": multiply,
                            "plan": plan, "output_sha256": result, "exact": True,
                            "graph": True, "nondefault_stream": True, "guards": True})
    return results


def main() -> None:
    """Run bounded correctness checks or paired full-expert CT component timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct-library", type=Path, required=True)
    parser.add_argument("--palette-library", type=Path, required=True)
    parser.add_argument("--affine-library", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--check-arm", choices=("baseline", "palette", "affine"),
                        help="Isolate one native library for sanitizer diagnosis")
    parser.add_argument("--check-cpu-routing", action="store_true",
                        help="Use independent routing fixtures to isolate native GEMM checks")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile-arm", choices=("baseline", "palette", "affine"))
    parser.add_argument("--family", choices=("w13", "w2"))
    parser.add_argument("--tokens", type=int, choices=(1, 8, 32))
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Preserve existing evidence; choose a fresh report path")
    if not args.checks_only and args.source is None:
        parser.error("Timings require the real checkpoint")
    if args.profile_arm and (args.family is None or args.tokens is None or args.checks_only):
        parser.error("Profiling needs one real family/token count")
    if (args.check_arm or args.check_cpu_routing) and not args.checks_only:
        parser.error("Isolated check arms/routing cannot be used for real timings")
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(174)
    torch.cuda.set_per_process_memory_fraction(3072 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    paths = {"baseline": args.ct_library, "palette": args.palette_library, "affine": args.affine_library}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    if hashes["baseline"] != CT_LIBRARY_SHA256:
        raise ValueError("Frozen CT baseline differs from the qualified library")
    report = {"status": "running", "libraries_sha256": hashes,
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "device": torch.cuda.get_device_name(), "cases": [],
              "scope": "MXFP4 E8M0 group32 shared-U8 scale expansion; code bytes unchanged",
              "additional_storage_savings_bytes": 0, "extra_decode_scratch_bytes": 0,
              "limits": ["Component with synthetic activations/routing; no serving-speed claim.",
                         "Single GPU, full layer-23 TP2 rank-zero expert shards; no communication.",
                         "Both candidates retain CT's existing compression and tensor-core arithmetic.",
                         "No serving plugin is enabled by this benchmark."]}
    started = time.monotonic()

    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(".incomplete")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.report)

    save()
    try:
        import vllm
        from mxwave_glm53._runtime import moe_runtime as runtime
        report["vllm"] = vllm.__version__
        report["runtime_sha256"] = hashlib.sha256(Path(inspect.getfile(runtime)).read_bytes()).hexdigest()
        natives = {name: runtime.NativeMoe(path) for name, path in paths.items()}
        report["skip_unused_prefetch"] = {}
        report["tail_policy"] = {}
        for name, native in natives.items():
            if native.encoding != "sparse_palette" or not native.affine_only or not native.shared_stage:
                raise ValueError("Require the same CT affine-palette shared-U8 consumer")
            if name != "baseline":
                native.library.zipserv_fp4_scale_expansion.restype = ctypes.c_int
                if native.library.zipserv_fp4_scale_expansion() != int(name == "affine"):
                    raise ValueError("Library expansion flag differs from its selected arm")
            skip = False
            if hasattr(native.library, "zipserv_fp4_skip_unused_prefetch"):
                native.library.zipserv_fp4_skip_unused_prefetch.restype = ctypes.c_int
                skip = bool(native.library.zipserv_fp4_skip_unused_prefetch())
            report["skip_unused_prefetch"][name] = skip
            policy = int(skip)
            if hasattr(native.library, "zipserv_fp4_tail_policy"):
                native.library.zipserv_fp4_tail_policy.restype = ctypes.c_int
                policy = native.library.zipserv_fp4_tail_policy()
            report["tail_policy"][name] = {0: "keep", 1: "skip", 2: "barrier"}[policy]
        if report["skip_unused_prefetch"]["palette"] != report["skip_unused_prefetch"]["affine"]:
            raise ValueError("The palette and affine controls must use the same tail-prefetch policy")
        if report["tail_policy"]["palette"] != report["tail_policy"]["affine"]:
            raise ValueError("The palette and affine controls must use the same shared-reuse policy")
        if args.checks_only:
            checked = {"baseline": natives[args.check_arm]} if args.check_arm else natives
            report["isolated_check_arm"] = args.check_arm
            report["check_routing"] = "cpu" if args.check_cpu_routing else "vllm"
            report["decode_checks"] = decode_checks(runtime, checked, args.smoke)
            report["gemm_checks"] = gemm_checks(runtime, checked, args.check_cpu_routing)
        else:
            report["source_metadata_sha256"] = {
                name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
                for name in ("config.json", "model.safetensors.index.json")}
            tensors = fixture(args.source, 23, list(range(288)))
            pressure = torch.zeros(128 * 2**20, dtype=torch.uint8, device="cuda")
            for family, weight, raw in tensors:
                if args.family and family != args.family:
                    continue
                packed = runtime.pack_scales(raw, Path("unused"), encoding="sparse_palette")
                for native in natives.values():
                    if not torch.equal(native.decode(packed), raw.view(torch.uint8)):
                        raise RuntimeError("Real source scale bytes changed")
                row = {"family": family, "weight_sha256": digest(weight), "scale_sha256": digest(raw),
                       "unchanged_fp4_code_bytes": weight.numel() * weight.element_size(),
                       "raw_scale_bytes": raw.numel(), "ct_scale_bytes": packed.retained_bytes,
                       "candidate_scale_bytes": packed.retained_bytes,
                       "exception_vectors": packed.exceptions.numel() // 16, "measurements": []}
                report["cases"].append(row)
                for tokens, pattern in ((1, "concentrated"), (8, "concentrated"),
                                        (8, "spread"), (32, "spread")):
                    if args.tokens and (tokens != args.tokens or (tokens == 8 and pattern != "spread")):
                        continue
                    topk, m = (8, tokens) if family == "w13" else (1, tokens * 8)
                    values = ([0, 1, 2, 3, 143, 144, 286, 287] * tokens if pattern == "concentrated"
                              else [(i * 5) % 288 for i in range(tokens * 8)])
                    ids = torch.tensor(values, dtype=torch.int32, device="cuda").reshape(m, topk)
                    x = torch.randn((m, raw.shape[1] * 32), dtype=torch.bfloat16, device="cuda") * .125
                    runs, buffers, guard, plan = bind_case(runtime, natives, weight, packed, x,
                                                         ids, family == "w2")
                    output_hash = exact_gemms(runs, buffers, guard, graphs=True)
                    if args.profile_arm:
                        run = runs[args.profile_arm]
                        for _ in range(3):
                            run()
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_push("zipserv_fp4_scale_expansion")
                        run()
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                        timings = None
                    else:
                        timings = measure({**runs, "control": runs["baseline"]}, pressure)
                    if exact_gemms(runs, buffers, guard, graphs=False) != output_hash:
                        raise RuntimeError("Post-timing outputs changed")
                    row["measurements"].append({"tokens": tokens, "routing": pattern,
                                                "unique_experts": len(set(values)), "plan": plan,
                                                "exact_outputs_to_ct": True, "timings": timings})
                    save()
                    print(json.dumps({"family": family, "tokens": tokens, "routing": pattern,
                                      "warm_us": timings["warm"]["median_us"] if timings else None}), flush=True)
                    del runs, buffers, guard
        report.update(status="passed", seconds=time.monotonic() - started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
    except BaseException as error:
        report.update(status="failed", error=repr(error), seconds=time.monotonic() - started)
        save()
        raise


if __name__ == "__main__":
    main()
