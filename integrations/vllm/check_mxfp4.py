# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Bounded checks for the bundled MXFP4 runtime and actual vLLM GEMM hook."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


def checksum(tensor: torch.Tensor) -> str:
    """Hash bytes without converting their represented floating-point values."""
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def capture(run):
    """Capture after warming the opaque operation on a non-default CUDA stream."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def routing(ids: torch.Tensor, experts: int, block: int):
    """Independently construct padded routing to isolate native synchronization checks."""
    flat = ids.cpu().numpy().reshape(-1)
    positions, expert_ids = [], []
    for expert in range(experts):
        selected = np.flatnonzero(flat == expert).tolist()
        if selected:
            selected += [flat.size] * (-len(selected) % block)
            positions.extend(selected)
            expert_ids.extend([expert] * (len(selected) // block))
    return (torch.tensor(positions, dtype=torch.int32, device="cuda"),
            torch.tensor(expert_ids, dtype=torch.int32, device="cuda"),
            torch.tensor([len(positions)], dtype=torch.int32, device="cuda"))


def layer_fixture(palette, experts=8, n=256, k=256):
    """Create a real torch module and vLLM quant descriptors with controlled scale entropy."""
    from vllm.model_executor.layers.fused_moe.config import mxfp4_moe_quant_config
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
    layer = torch.nn.Module()
    layer.prefix = "independent-mxfp4-check"
    for name in ("w13_weight_scale", "w2_weight_scale"):
        if palette is None:
            raw = torch.randint(1, 255, (experts, k // 32, n), dtype=torch.uint8, device="cuda")
        else:
            raw = torch.tensor(palette, dtype=torch.uint8, device="cuda")[
                torch.randint(0, len(palette), (experts, k // 32, n), device="cuda")]
            raw.reshape(-1)[::257] = 117  # Exact whole-vector exceptions.
        layer.register_parameter(name, torch.nn.Parameter(raw.view(torch.float8_e8m0fnu),
                                                          requires_grad=False))
    config = mxfp4_moe_quant_config(layer.w13_weight_scale, layer.w2_weight_scale)
    method = SimpleNamespace(use_cutlass_mxfp4=False, experts_cls=MarlinExperts, group_size=32,
        moe=SimpleNamespace(has_bias=False, layer_name=layer.prefix), moe_quant_config=config,
        moe_kernel=None)
    return layer, method


def main() -> None:
    """Exercise generic palette decoding, ownership, fallback and captured hook execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Preserve prior results; choose a fresh report path")
    torch.set_num_threads(2)
    torch.manual_seed(174)
    torch.cuda.set_per_process_memory_fraction(2**30 / torch.cuda.get_device_properties(0).total_memory)
    os.environ.update(ZIPSERV_MXFP4_SCALES="on", ZIPSERV_MXFP4_LIBRARY=str(args.library.resolve()))
    from zipserv_vllm.mxfp4_plugin import register
    register()
    register()  # Idempotent registration must not wrap or allocate twice.
    from zipserv_vllm import mxfp4_backend as backend
    from zipserv_vllm.mxfp4_runtime import NativeMoe, pack_scales
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    native = NativeMoe(args.library)
    result = {"status": "running", "cases": [], "native_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "routing": "independent CPU fixture; installed vLLM router is outside these checks",
              "scope": "Native/plugin component; full-model qualification is separate"}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    try:
        config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16), lora_config=None,
                                 parallel_config=SimpleNamespace(enable_expert_parallel=False))
        for palette in ((120, 121, 122), (118, 121, 125)):
            for m, topk in ((1, 8), (8, 8), (32, 1)):
                experts, n, k, block = 8, 256, 256, 8
                layer, method = layer_fixture(palette, experts, n, k)
                ids = (torch.arange(m * topk, dtype=torch.int32, device="cuda") % experts).reshape(m, topk)
                sorted_ids, expert_ids, padded = routing(ids, experts, block)
                x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * .125
                weight = torch.randint(-(2**31), 2**31-1, (experts, k // 16, n * 2), dtype=torch.int32, device="cuda")
                weight_sha = checksum(weight)
                guard = torch.full((m * topk * n + 64,), -7, dtype=torch.bfloat16, device="cuda")
                output = guard[32:-32].reshape(m * topk, n)
                workspace = torch.zeros(torch.cuda.get_device_properties(0).multi_processor_count * 4,
                                         dtype=torch.int32, device="cuda")
                weights = torch.full((m, topk), .125, dtype=torch.float32, device="cuda")
                plan = native.plan(m, n, k, block, topk, experts, 0)

                def run():
                    return ops.moe_wna16_marlin_gemm(input=x, output=output, b_qweight=weight,
                        b_bias=None, b_scales=layer.w13_weight_scale, a_scales=None, global_scale=None,
                        b_qzeros=None, workspace=workspace, sorted_token_ids=sorted_ids, expert_ids=expert_ids,
                        num_tokens_past_padded=padded, topk_weights=weights, moe_block_size=block,
                        top_k=topk, mul_topk_weights=False, b_q_type=scalar_types.float4_e2m1f,
                        size_m=m, size_n=n, size_k=k, use_atomic_add=False, use_fp32_reduce=True,
                        is_zp_float=False, thread_k=plan[0], thread_n=plan[1], blocks_per_sm=plan[2])

                run()  # The unregistered raw tensor must take the unchanged stock path.
                reference = output.clone()
                if not bool(torch.isfinite(reference).all()):
                    raise RuntimeError("Stock control returned nonfinite output")
                raw_scales = checksum(layer.w13_weight_scale)
                with patch("vllm.config.get_current_vllm_config", return_value=config):
                    backend.prepare_layer(method, layer, "on")
                if not all(r["compressed"] for r in layer.zipserv_mxfp4_qualification["scales"]):
                    raise RuntimeError("An eligible test scale unexpectedly fell back")
                if method.moe_quant_config.w1_scale is not layer.w13_weight_scale:
                    raise RuntimeError("Quant descriptor retained the raw scale owner")
                run()
                candidate = output.clone()
                relative = float((reference.float() - candidate.float()).norm() / reference.float().norm().clamp_min(1e-20))
                exact_to_stock = torch.equal(reference.view(torch.uint8), candidate.view(torch.uint8))
                if not bool(torch.isfinite(candidate).all()) or relative > .005:
                    raise RuntimeError("Native arithmetic diverged from the unchanged stock control")
                graph = capture(run)
                graph.replay()
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                for _ in range(3):
                    output.zero_()
                    graph.replay()
                torch.cuda.synchronize()
                if (not torch.equal(output.view(torch.uint8), candidate.view(torch.uint8))
                        or bool(torch.count_nonzero(workspace))
                        or not bool((guard[:32] == -7).all()) or not bool((guard[-32:] == -7).all())
                        or torch.cuda.memory_allocated() != before or checksum(weight) != weight_sha):
                    raise RuntimeError("Captured hook output, guards, locks, weights or allocations changed")
                audit = backend.report(verify=True)
                result["cases"].append({"palette": palette, "m": m, "topk": topk,
                    "stock_output_bits_exact": exact_to_stock, "stock_relative_l2": relative,
                    "source_scale_sha256": raw_scales, "source_weights_unchanged": True,
                    "graph_replay_exact": True, "graph_growth_bytes": 0, "guards": True,
                    "raw_owners_released": all(v["raw_owner_released"] for v in audit["ownership"])})
                save()
                del graph, run, layer, method, candidate, reference, weight, x, output, guard, workspace
        layer, method = layer_fixture(None)
        owners = (layer.w13_weight_scale, layer.w2_weight_scale)
        with patch("vllm.config.get_current_vllm_config", return_value=config):
            backend.prepare_layer(method, layer, "on")
        if (any(r["compressed"] for r in layer.zipserv_mxfp4_qualification["scales"])
                or layer.w13_weight_scale is not owners[0] or layer.w2_weight_scale is not owners[1]):
            raise RuntimeError("Nonshrinking scale fallback changed stock ownership")
        # Keep this fallback module alive while its worker audit is checked.
        result["fallback_preserves_stock"] = True
        fallback_layers = [layer]
        result["unsupported_configuration_fallbacks"] = []
        for reason in ("other_backend", "weight_offload", "lora", "expert_parallel"):
            layer, method = layer_fixture((120, 121, 122))
            owners = (layer.w13_weight_scale, layer.w2_weight_scale)
            configured = SimpleNamespace(model_config=config.model_config, lora_config=None,
                parallel_config=SimpleNamespace(enable_expert_parallel=False))
            if reason == "other_backend":
                method.experts_cls = object
            elif reason == "weight_offload":
                configured.offload_config = SimpleNamespace(
                    uva=SimpleNamespace(cpu_offload_gb=1),
                    prefetch=SimpleNamespace(offload_group_size=0))
            elif reason == "lora":
                configured.lora_config = object()
            else:
                configured.parallel_config.enable_expert_parallel = True
            with patch("vllm.config.get_current_vllm_config", return_value=configured):
                backend.prepare_layer(method, layer, "on")
            if (any(r["compressed"] for r in layer.zipserv_mxfp4_qualification["scales"])
                    or layer.w13_weight_scale is not owners[0] or layer.w2_weight_scale is not owners[1]):
                raise RuntimeError("Unsupported configuration changed stock scale ownership")
            fallback_layers.append(layer)
            result["unsupported_configuration_fallbacks"].append(reason)
        bad = torch.full((1, 1, 16), 255, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
        try:
            pack_scales(bad)
        except ValueError:
            result["reserved_scale_rejected"] = True
        else:
            raise RuntimeError("Reserved E8M0 scales were accepted")
        gc.collect()
        probe = torch.nn.Module()
        probe.register_parameter('first', torch.nn.Parameter(
            torch.ones(128, dtype=torch.bfloat16, device='cuda'), requires_grad=False))
        probe.register_parameter('alias', torch.nn.Parameter(probe.first.detach(), requires_grad=False))
        inventory = backend.ScaleWorker.zipserv_weight_inventory(
            SimpleNamespace(model_runner=SimpleNamespace(get_model=lambda: probe)))
        if (inventory['unique_cuda_storage_bytes'] != 256 or len(inventory['storages']) != 1
                or len(inventory['storages'][0]['owners']) != 2):
            raise RuntimeError('Model inventory double-counted tied tensor storage')
        result['inventory_counts_aliases_once'] = True
        del probe
        result.update(status="passed", audit=backend.report(verify=True),
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        save()
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
