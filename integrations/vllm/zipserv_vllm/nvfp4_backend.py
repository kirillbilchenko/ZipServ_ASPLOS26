# Copyright MxWave contributors.
# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Experimental ModelOpt W4A16 NVFP4 scale storage for the Marlin backend.

The original loader, FP4 codes, global scales, routing and activation stay in
vLLM. Only qualified post-Marlin scale byte loads use the native consumer.
CSF/B12X checkpoints and activation-quantized NVFP4 are outside this adapter.
"""
from __future__ import annotations

import functools
import gc
import hashlib
import inspect
import logging
import weakref
from typing import Any

import torch

from .mxfp4_backend import _replace_config_scales
from .nvfp4_runtime import NativeMoe, layout_rejection, pack_scales
from .scale_ownership import scale_metadata, storage_identity

logger = logging.getLogger(__name__)
_INSTALLED = False
_SETTINGS = None
_NATIVE = None
_STORAGE: dict[int, dict] = {}
_DESCRIPTOR_STORAGES: set[tuple[str, int]] = set()
_SCRATCH: dict[tuple[int, int], torch.Tensor] = {}
_LAYERS: list[dict] = []


def native() -> NativeMoe:
    global _NATIVE
    if _NATIVE is None:
        candidate = NativeMoe(_SETTINGS.library)
        if not candidate.compressed:
            raise ValueError("NVFP4 serving requires the delta5 library; raw is a benchmark control")
        _NATIVE = candidate
    return _NATIVE


def _launch(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
            global_scale: torch.Tensor, output: torch.Tensor, temporary: torch.Tensor,
            locks: torch.Tensor, ids: torch.Tensor, experts: torch.Tensor,
            padded: torch.Tensor, routing_weights: torch.Tensor,
            m: int, n: int, k: int, e: int, topk: int, block: int, multiply: bool,
            tk: int, tn: int, bpsm: int) -> None:
    native().launch((x, weight, scales, global_scale, output, temporary, locks, ids,
                     experts, padded, routing_weights),
                    (m, n, k, e, topk, block, int(multiply), tk, tn, bpsm))


_LIBRARY = torch.library.Library("zipserv_nvfp4", "DEF")
_LIBRARY.define("moe_gemm" + torch.library.infer_schema(
    _launch, mutates_args=("output", "temporary", "locks")))
_LIBRARY.impl("moe_gemm", _launch, "CUDA")


@torch.library.register_fake("zipserv_nvfp4::moe_gemm")
def _fake(*args: Any, **kwargs: Any) -> None:
    return None


_GEMM = torch.ops.zipserv_nvfp4.moe_gemm.default


@functools.lru_cache(maxsize=1024)
def _plan(m: int, n: int, k: int, block: int, topk: int, e: int, device: int):
    return native().plan(m, n, k, block, topk, e, device)


def _temporary(x: torch.Tensor) -> torch.Tensor:
    key = (x.device.index, torch.cuda.current_stream(x.device).cuda_stream)
    if key not in _SCRATCH:
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        _SCRATCH[key] = torch.empty(sms * 4 * 64 * 256, device=x.device, dtype=torch.float32)
    return _SCRATCH[key]


def prepare_layer(method: Any, layer: torch.nn.Module, mode: str) -> None:
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        get_marlin_input_dtype,
    )
    config = get_current_vllm_config()
    label = str(getattr(layer, "prefix", getattr(method.moe, "layer_name", "unknown")))
    entry = {"layer": label, "mode": mode, "scales": []}
    reason = None
    # ModelOpt's use_a16 records the checkpoint recipe, not Marlin's effective
    # activation path. Stock vLLM also serves plain NVFP4 checkpoints with BF16
    # Marlin activations; its oracle drops their activation scales. Validate the
    # selected consumer here, and the actual tensor dtype again in the GEMM.
    if method.experts_cls is not MarlinExperts:
        reason = "Only ModelOpt NVFP4 with the BF16 Marlin consumer is supported"
    elif getattr(method.quant_config, "weight_scale_encoding", None):
        reason = "Already encoded scales require their original consumer"
    elif (config.model_config.dtype != torch.bfloat16
          or get_marlin_input_dtype(prefix="") not in (None, torch.bfloat16)):
        reason = "Native consumer requires BF16 activations"
    elif config.lora_config is not None or config.parallel_config.enable_expert_parallel:
        reason = "Expert LoRA and expert parallelism have not been qualified"
    elif getattr(method.moe, "has_bias", False):
        reason = "Biased expert GEMMs have not been qualified"
    elif (getattr(config, "weight_transfer_config", None) is not None
          or getattr(config.model_config, "enable_sleep_mode", False)):
        reason = "Reload/sleep cannot relocate native pointer descriptors"
    offload = getattr(config, "offload_config", None)
    if offload is not None and (offload.uva.cpu_offload_gb > 0
                               or offload.prefetch.offload_group_size > 0):
        reason = "Offloading cannot relocate native pointer descriptors"
    compilation = getattr(config, "compilation_config", None)
    if compilation is not None and int(compilation.mode) != 0:
        reason = "NVFP4 adapter currently qualifies eager/CUDA graphs with compilation mode 0"
    for name in ("w13_weight_scale", "w2_weight_scale"):
        raw = getattr(layer, name)
        row = {"name": name, "shape": list(raw.shape), "raw_bytes": raw.nbytes,
               "retained_bytes": raw.nbytes, "allocated_bytes": (raw.nbytes + 511) // 512 * 512,
               "compressed": False}
        entry["scales"].append(row)
        if mode == "off":
            row["fallback"] = "disabled"
            continue
        parts = 2 if name.startswith("w13") and method.moe.is_act_and_mul else 1
        fallback = reason or layout_rejection(tuple(raw.shape), parts)
        if raw.dtype != torch.float8_e4m3fn or raw.device.type != "cuda" or not raw.is_contiguous():
            fallback = fallback or "Unsupported post-Marlin scale storage"
        if fallback:
            row["fallback"] = fallback
            continue
        packed = pack_scales(raw, parts)
        saving = 100 * (1 - packed.allocated_bytes / row["allocated_bytes"])
        if saving < _SETTINGS.min_saving_percent:
            row.update(fallback="Below requested saving including metadata/allocation padding",
                       candidate_bytes=packed.retained_bytes, candidate_saving_percent=saving)
            continue
        decoded = native().decode(packed)
        if not torch.equal(decoded, raw.view(torch.uint8)):
            raise RuntimeError(f"{label}: GPU NVFP4 scale reconstruction changed bytes")
        del decoded
        replacement = torch.nn.Parameter(packed.descriptor, requires_grad=False)
        references = _replace_config_scales(method, raw, replacement)
        if not references:
            raise RuntimeError("No prepared quant descriptor owned the NVFP4 scale tensor")
        # Registered buffers keep native pointer targets visible to inventories.
        # Runtime metadata also holds them strongly throughout graph replay.
        for i, buffer in enumerate(packed.buffers[:-1]):
            layer.register_buffer(f"zipserv_{name}_{i}", buffer)
        setattr(layer, name, replacement)
        metadata = {"replacement": replacement, "shape": packed.shape, "packed": packed,
                    "source_sha256": packed.source_sha256, "raw_owner": weakref.ref(raw),
                    "layer": label, "name": name, "eager_or_capture_calls": 0}
        _STORAGE[id(replacement)] = metadata
        _DESCRIPTOR_STORAGES.add(storage_identity(replacement))
        row.update(compressed=True, retained_bytes=packed.retained_bytes,
                   allocated_bytes=packed.allocated_bytes, scale_saving_percent=saving,
                   source_sha256=packed.source_sha256, gpu_bytes_exact=True,
                   raw_parts=sum(r["raw_fallback"] for r in packed.records),
                   quant_descriptor_references=references)
    _LAYERS.append(entry)
    layer.zipserv_nvfp4_qualification = entry
    for row in entry["scales"]:
        if not row["compressed"]:
            logger.info("ZipServ NVFP4 keeps %s.%s stock: %s", label, row["name"], row["fallback"])


def report(verify: bool = False) -> dict:
    if verify:
        torch.cuda.synchronize()
        gc.collect()
    checks = []
    for item in _STORAGE.values():
        released = item["raw_owner"]() is None
        row = {"layer": item["layer"], "name": item["name"], "raw_owner_released": released,
               "eager_or_capture_calls": item["eager_or_capture_calls"]}
        if verify:
            decoded = native().decode(item["packed"])
            digest = hashlib.sha256(decoded.cpu().numpy()).hexdigest()
            del decoded
            if digest != item["source_sha256"] or not released:
                raise RuntimeError("NVFP4 scale byte/ownership audit failed")
            row["gpu_bytes_exact"] = True
        checks.append(row)
    raw = sum(r["raw_bytes"] for layer in _LAYERS for r in layer["scales"])
    kept = sum(r["allocated_bytes"] for layer in _LAYERS for r in layer["scales"])
    scratch = sum(t.nbytes for t in _SCRATCH.values())
    return {"mode": _SETTINGS.mode, "layers": _LAYERS, "ownership": checks,
            "compressed_scale_groups": len(_STORAGE), "scale_raw_bytes": raw,
            "scale_retained_allocation_bytes": kept, "scratch_bytes": scratch,
            "saving_after_scratch_bytes": raw - kept - scratch,
            "cuda_allocated_bytes": torch.cuda.memory_allocated()}


class ScaleWorker:
    def zipserv_nvfp4_report(self, verify: bool = False) -> dict:
        return report(verify)


def install(selected) -> None:
    global _INSTALLED, _SETTINGS
    if _INSTALLED:
        if selected != _SETTINGS:
            raise RuntimeError("Cannot change NVFP4 mode after installation")
        return
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
    from vllm.scalar_type import scalar_types
    original_prepare = ModelOptNvFp4FusedMoE.process_weights_after_loading
    if (original_prepare.__module__ != ModelOptNvFp4FusedMoE.__module__
            or hasattr(original_prepare, "__wrapped__")):
        raise RuntimeError("Another plugin owns NVFP4 preparation")
    _SETTINGS = selected
    if selected.mode == "on":
        native()  # Validate ABI before installing hooks or releasing any scales.

    @functools.wraps(original_prepare)
    def prepare(method, layer):
        original_prepare(method, layer)
        prepare_layer(method, layer, selected.mode)

    ModelOptNvFp4FusedMoE.process_weights_after_loading = prepare
    _INSTALLED = True
    if selected.mode == "off":
        return
    stock = ops.moe_wna16_marlin_gemm
    signature = inspect.signature(stock)
    position = list(signature.parameters).index("b_scales")

    @functools.wraps(stock)
    def gemm(*args, **kwargs):
        candidate = kwargs.get("b_scales", args[position] if len(args) > position else None)
        metadata = scale_metadata(candidate, _STORAGE, _DESCRIPTOR_STORAGES)
        if metadata is None:
            return stock(*args, **kwargs)
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        p = call.arguments
        x, weight, output, gs = p["input"], p["b_qweight"], p["output"], p["global_scale"]
        if (any(p[key] is not None and p[key].numel() for key in ("b_bias", "a_scales", "b_qzeros"))
                or p["use_atomic_add"] or not p["use_fp32_reduce"] or p["is_zp_float"]
                or x.dtype != torch.bfloat16 or output is None or gs is None
                or p["b_q_type"] != scalar_types.float4_e2m1f):
            raise ValueError("GEMM is outside the W4A16 NVFP4 runtime contract")
        m, n, k = (int(p[key]) for key in ("size_m", "size_n", "size_k"))
        e, groups, expected_n = metadata["shape"]
        topk, block = int(p["top_k"]), int(p["moe_block_size"])
        if (m < 0 or topk < 1 or m * topk * max(n, k) >= 2**31
                or n != expected_n or k != groups * 16
                or tuple(weight.shape) != (e, k // 16, n * 2) or weight.dtype != torch.int32
                or output.dtype != torch.bfloat16 or output.numel() < m * topk * n
                or x.ndim != 2 or x.shape[1] != k or x.shape[0] < m
                or gs.dtype != torch.float32 or gs.numel() != e
                or block not in (8, 16, 32, 48, 64)):
            raise ValueError("NVFP4 scale ownership does not match the GEMM geometry")
        if m == 0:
            return output
        integers = tuple(p[key] for key in ("workspace", "sorted_token_ids", "expert_ids", "num_tokens_past_padded"))
        tensors = (x, weight, candidate, gs, output, p["topk_weights"], *integers)
        if (x.device.type != "cuda" or any(t.device != x.device or not t.is_contiguous() for t in tensors)
                or any(t.dtype != torch.int32 for t in integers)
                or p["topk_weights"].dtype != torch.float32 or p["topk_weights"].numel() < m * topk
                or p["sorted_token_ids"].numel() < m * topk
                or p["expert_ids"].numel() * block < m * topk
                or p["num_tokens_past_padded"].numel() != 1
                or p["workspace"].numel() < torch.cuda.get_device_properties(x.device).multi_processor_count * 4):
            raise ValueError("NVFP4 GEMM buffers violate the native device/layout contract")
        plan = _plan(m, n, k, block, topk, e, x.device.index)
        if p["thread_k"] > 0 or p["thread_n"] > 0:
            plan = (p["thread_k"], p["thread_n"], p["blocks_per_sm"])
        metadata["eager_or_capture_calls"] += 1
        _GEMM(x, weight, candidate, gs, output, _temporary(x), *integers,
              p["topk_weights"], m, n, k, e, topk, block, p["mul_topk_weights"], *plan)
        return output

    ops.moe_wna16_marlin_gemm = gemm
