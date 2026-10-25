"""Complete real MoE-layer comparison against CT's existing fused scale kernel.

All 288 experts' code weights are loaded unchanged. Only scale representation
and decoding vary. Synthetic routing/activations do not qualify model quality
or end-to-end serving; a shared full-scale workspace remains necessary.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch

from bench_fp8_palette import digest, measure
from bench_scale_dictionary import Decoder, fixture, load_decoder
from bench_selected_scales import Selected, check_output
from low_precision_codec import encode_container


def main():
    """Compare CT palette, native nibbles and full/routed dictionary decode plus GEMM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--aligned-library", type=Path, required=True)
    parser.add_argument("--ct-library", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Preserve previous evidence; use a fresh report path")
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(174)
    torch.cuda.set_per_process_memory_fraction(3072 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    report = {"status": "running", "scope": "complete layer-23, TP2 rank-zero component",
        "cases": [], "experts": 288, "source": str(args.source),
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "libraries": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
            (("decoder", args.library), ("aligned", args.aligned_library), ("ct", args.ct_library))},
        "source_metadata_sha256": {name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
                                  for name in ("config.json", "model.safetensors.index.json")},
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": ["Synthetic activations and routing; no end-to-end model speed claim.",
            "No distributed communication; complete TP2 rank-zero expert shards on one GPU.",
            "Dictionary misses use a full-size scratch buffer, intended to be shared across layers.",
            "No decoded cache; routed decoding is repeated on every invocation.",
            "CT's separate production recipe is not started or modified."]}
    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(".incomplete")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.report)
    save()
    started = time.monotonic()
    try:
        import vllm
        from vllm import _custom_ops as ops
        from mxwave_glm53._runtime import moe_runtime as runtime
        report["vllm"] = vllm.__version__
        report["consumer_wrapper_sha256"] = hashlib.sha256(Path(inspect.getfile(runtime)).read_bytes()).hexdigest()
        decoder_library = load_decoder(args.library)
        native = runtime.NativeMoe(args.aligned_library)
        ct_native = runtime.NativeMoe(args.ct_library)
        if native.encoding != "aligned_nibbles" or ct_native.encoding != "sparse_palette":
            raise ValueError("Require exact existing aligned and CT sparse-palette consumers")
        tensors = fixture(args.source, 23, list(range(288)))
        pressure = torch.zeros(128 * 2**20, dtype=torch.uint8, device="cuda")
        for family, weight, raw in tensors:
            aligned = runtime.pack_scales(raw, Path("unused"), encoding="aligned_nibbles")
            ct = runtime.pack_scales(raw, Path("unused"), encoding="sparse_palette")
            for consumer, packed in ((native, aligned), (ct_native, ct)):
                if not torch.equal(consumer.decode(packed), raw.view(torch.uint8)):
                    raise RuntimeError("Existing native scale consumer changed bytes")
            cpu = aligned.records.view(torch.uint8).cpu().numpy().tobytes()
            blob, accounting = encode_container(cpu, "bytes", "vector4-warp", 1024)
            decoder = Decoder(decoder_library, blob, aligned.records.view(torch.uint8).reshape(-1))
            row = {"family": family, "weight_sha256": digest(weight), "scale_sha256": digest(raw),
                "native_sha256": hashlib.sha256(cpu).hexdigest(), "encoding": accounting,
                "ct_scale_bytes": ct.retained_bytes, "aligned_scale_bytes": aligned.retained_bytes,
                "dictionary_plus_descriptor_bytes": len(blob) + 16,
                "unchanged_fp4_code_bytes": weight.numel() * weight.element_size(),
                "required_full_address_scratch_bytes": len(cpu), "measurements": []}
            report["cases"].append(row)
            save()
            e, groups, n = raw.shape
            k = groups * 32
            for tokens, pattern in ((1, "concentrated"), (8, "concentrated"),
                                    (8, "spread"), (32, "spread")):
                topk, m = (8, tokens) if family == "w13" else (1, tokens * 8)
                values = ([0, 1, 2, 3, 143, 144, 286, 287] * tokens if pattern == "concentrated"
                          else [(i * 5) % 288 for i in range(tokens * 8)])
                ids = torch.tensor(values, dtype=torch.int32, device="cuda").reshape(m, topk)
                scores = torch.full((m, topk), 1 / 8, dtype=torch.float32, device="cuda")
                block = 8
                capacity = min(m * topk + e * (block - 1), m * topk * block)
                routing = (torch.empty(capacity, dtype=torch.int32, device="cuda"),
                    torch.empty((capacity + block - 1) // block, dtype=torch.int32, device="cuda"),
                    torch.empty(1, dtype=torch.int32, device="cuda"))
                ops.moe_align_block_size(ids, e, block, *routing, None)
                x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * .125
                buffers = runtime.allocate_buffers(m, n, topk, block, capacity, x.device)
                plan = native.plan(m, n, k, block, topk, e, 0)
                ct_plan = ct_native.plan(m, n, k, block, topk, e, 0)
                aligned_gemm = native.bind(x, weight, aligned, buffers, routing, scores,
                                            topk, block, family == "w2", plan)
                ct_gemm = ct_native.bind(x, weight, ct, buffers, routing, scores,
                                         topk, block, family == "w2", ct_plan)
                selected = Selected(decoder, e, ids)
                def full():
                    decoder.launch()
                    aligned_gemm()
                def routed():
                    selected.launch()
                    aligned_gemm()
                ct_gemm()
                reference = buffers.output.clone()
                ct_gemm()
                if (not bool(torch.isfinite(reference).all()) or not torch.equal(
                        reference.view(torch.uint8), buffers.output.view(torch.uint8))):
                    raise RuntimeError("CT baseline finiteness/self-repeat failed")
                for run in (full, routed):
                    decoder.output.fill_(0xa5)
                    run()
                    if not torch.equal(reference.view(torch.uint8), buffers.output.view(torch.uint8)):
                        raise RuntimeError("Dictionary decode plus GEMM changed CT outputs")
                check_output(selected, cpu)
                timings = measure({"baseline": ct_gemm, "control": ct_gemm,
                    "aligned_predecoded": aligned_gemm, "full_dictionary": full,
                    "routed_dictionary": routed}, pressure)
                decoder.output.fill_(0xa5)
                routed()
                check_output(selected, cpu)
                if (not torch.equal(reference.view(torch.uint8), buffers.output.view(torch.uint8))
                        or bool(torch.count_nonzero(buffers.locks))):
                    raise RuntimeError("Post-timing output exactness/lock check failed")
                row["measurements"].append({"tokens": tokens, "routing": pattern,
                    "unique_experts": len(set(values)), "route_slots": selected.slots,
                    "ct_plan": ct_plan, "aligned_plan": plan, "exact_outputs_to_ct": True,
                    "timings": timings})
                save()
                print(json.dumps({"family": family, "tokens": tokens, "routing": pattern,
                                  "warm_us": timings["warm"]["median_us"]}), flush=True)
                del ct_gemm, aligned_gemm, full, routed, selected, buffers, x, reference
            del decoder, aligned, ct
        report.update(status="passed", seconds=time.monotonic() - started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
    except BaseException as error:
        report.update(status="failed", error=repr(error), seconds=time.monotonic() - started)
        save()
        raise


if __name__ == "__main__":
    main()
