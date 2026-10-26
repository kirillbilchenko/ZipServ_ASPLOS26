# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Bounded checks for mixed BF16/FP8 storage and actual vLLM consumers."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-library", type=Path, required=True)
    parser.add_argument("--fp8-library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Preserve prior reports")
    torch.set_num_threads(2)
    torch.manual_seed(481)
    torch.cuda.set_per_process_memory_fraction(2 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    os.environ.update(ZIPSERV_LIBRARY=str(args.bf16_library.resolve()),
        ZIPSERV_FP8_LIBRARY=str(args.fp8_library.resolve()), ZIPSERV_MIXED_BF16="on",
        ZIPSERV_MIXED_FP8="on", ZIPSERV_BF16_TARGETS='["*"]', ZIPSERV_FP8_TARGETS='["*"]')
    from zipserv_vllm.mixed_plugin import register, settings
    register()
    register()
    from zipserv_vllm.mixed_backend import _prepare, MixedWorker
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (init_distributed_environment, initialize_model_parallel,
                                 destroy_model_parallel, destroy_distributed_environment)
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_mxfp8 import CompressedTensorsW8A8Mxfp8
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel

    result = {"status": "running", "cases": [], "scope": "Small fixtures, not full-model qualification"}
    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    save()
    try:
        with tempfile.TemporaryDirectory() as temporary, set_current_vllm_config(VllmConfig()):
            init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl",
                                         distributed_init_method="file://" + temporary + "/rendezvous")
            initialize_model_parallel(backend="nccl")
            model = torch.nn.Module()
            for name, n in (("bf16", 256), ("bf16_padded", 224), ("fp8", 256), ("fp8_padded", 130), ("alias_a", 256), ("alias_b", 256), ("tiny", 1)):
                layer = ReplicatedLinear(256, n, bias=False, params_dtype=torch.bfloat16).cuda()
                layer.weight.data = (torch.randn(n, 256, device="cuda") * .125).to(torch.bfloat16)
                if name.startswith("fp8"):
                    layer.weight.data = layer.weight.to(torch.float8_e4m3fn)
                    layer.register_parameter("weight_scale", torch.nn.Parameter(
                        torch.full((n, 8), 127, dtype=torch.uint8, device="cuda"), requires_grad=False))
                    layer.input_size_per_partition = 256
                    layer.output_size_per_partition = n
                    layer.scheme = CompressedTensorsW8A8Mxfp8()
                    # These two methods have no constructor state; use the actual
                    # installed class and its stock prepare/apply implementations.
                    layer.scheme.kernel = object.__new__(MarlinMxfp8LinearKernel)
                    layer.quant_method = CompressedTensorsLinearMethod(None)
                    layer.quant_method.process_weights_after_loading(layer)
                model.add_module(name, layer)
            model.head = ParallelLMHead(256, 256, params_dtype=torch.bfloat16).cuda()
            model.head.weight.data = (torch.randn_like(model.head.weight) * .125)
            model.alias_b.weight = model.alias_a.weight
            # Multiple module paths for the same layer must not be mistaken for
            # two independently owning parameters.
            model.same_bf16_module = model.bf16
            inputs = {m: torch.randn(m, 256, device="cuda", dtype=torch.bfloat16) * .125
                      for m in (0, 1, 4, 8, 128, 129, 197, 769)}
            references = {(name, m): getattr(model, name).quant_method.apply(getattr(model, name), x).clone()
                          for name in ("bf16", "bf16_padded", "head", "fp8", "fp8_padded") for m, x in inputs.items()}
            config = SimpleNamespace(lora_config=None, weight_transfer_config=None,
                model_config=SimpleNamespace(enable_sleep_mode=False, dtype=torch.bfloat16),
                compilation_config=SimpleNamespace(mode=0, cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY")),
                offload_config=SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=0),
                                               prefetch=SimpleNamespace(offload_group_size=0)))
            worker = SimpleNamespace(vllm_config=config,
                model_runner=SimpleNamespace(get_model=lambda: model, model_memory_usage=0))
            _prepare(worker, settings(os.environ))
            assert all(getattr(model, name).weight is None for name in ("bf16", "bf16_padded", "head", "fp8", "fp8_padded"))
            assert model.alias_a.weight is model.alias_b.weight
            assert model.tiny.weight is not None
            for name in ("bf16", "bf16_padded", "head", "fp8", "fp8_padded"):
                layer = getattr(model, name)
                for m, x in inputs.items():
                    expected = references[name, m]
                    actual = layer.quant_method.apply(layer, x)
                    exact = torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
                    error = float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30))
                    if (name.startswith("fp8") and not exact) or not torch.isfinite(actual).all() or error > .005:
                        raise RuntimeError(f"Consumer mismatch: {name}, M={m}, exact={exact}, error={error}")
                    graph_exact = None
                    if m in (1, 4, 128):
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3): layer.quant_method.apply(layer, x)
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            captured = layer.quant_method.apply(layer, x)
                        graph.replay()
                        torch.cuda.synchronize()
                        before = torch.cuda.memory_allocated()
                        for _ in range(3): graph.replay()
                        torch.cuda.synchronize()
                        graph_exact = torch.equal(captured.view(torch.uint8), actual.view(torch.uint8))
                        if not graph_exact or torch.cuda.memory_allocated() != before:
                            raise RuntimeError("CUDA graph output or live allocation changed")
                        del graph, captured
                    result["cases"].append({"kind": name, "tokens": m, "stock_bits_exact": exact,
                                             "relative_l2": error, "graph_exact": graph_exact})
                    save()
            # Reproduce a serving-only boundary: a shared-expert stream sees
            # only fused small batches during warmup, then needs stock Marlin
            # for a mixed prefill/decode batch after workspace allocation locks.
            # Do not manually allocate stock locks here; the adapter must do it.
            from vllm.v1.worker.workspace import (init_workspace_manager, lock_workspace,
                                                 reset_workspace_manager, use_workspace_lane)
            init_workspace_manager(torch.device("cuda:0"), num_lanes=2)
            streams = [torch.cuda.current_stream(), torch.cuda.Stream()]
            for stream in streams:
                stream.wait_stream(torch.cuda.current_stream())
                for lane in (0, 1):
                    with torch.cuda.stream(stream), use_workspace_lane(lane):
                        for name in ("fp8", "fp8_padded"):
                            layer = getattr(model, name)
                            layer.quant_method.apply(layer, inputs[1])
            torch.cuda.synchronize()
            lock_workspace()
            result["locked_workspace_transitions"] = []
            try:
                for stream_index, stream in enumerate(streams):
                    for lane in (0, 1):
                        with torch.cuda.stream(stream), use_workspace_lane(lane):
                            for name in ("fp8", "fp8_padded"):
                                layer = getattr(model, name)
                                actual = layer.quant_method.apply(layer, inputs[197])
                                exact = torch.equal(actual.view(torch.uint8), references[name, 197].view(torch.uint8))
                                if not exact:
                                    raise RuntimeError("Locked-workspace FP8 fallback changed output bits")
                                result["locked_workspace_transitions"].append(
                                    {"stream": stream_index, "lane": lane, "kind": name, "stock_bits_exact": exact})
            finally:
                torch.cuda.synchronize()
                reset_workspace_manager()
            result["audit"] = MixedWorker.zipserv_mixed_report(worker, True)
            assert result["audit"]["raw_owners_released"]
            assert result["audit"]["per_type"]["bf16"]["compressed_layers"] == 3
            assert result["audit"]["per_type"]["fp8"]["compressed_layers"] == 2
            result.update(status="passed", shared_owner_fallback=True)
            destroy_model_parallel()
            destroy_distributed_environment()
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
