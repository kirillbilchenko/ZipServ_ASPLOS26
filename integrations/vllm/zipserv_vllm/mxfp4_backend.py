# Copyright MxWave contributors.
# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
# Adapted for ZipServ; see NOTICE and mxfp4-provenance.json.
"""Opt-in MXFP4 scale storage with stock MoE routing and activation.

The default is stock execution. Only explicitly prepared MXFP4 expert tensors
use the bundled fused GEMM operation; mixed MXFP8 linears stay on vLLM.
"""

from __future__ import annotations

import functools
import gc
import importlib
import hashlib
import weakref
from typing import Any

import torch

from .mxfp4_runtime import NativeMoe, pack_scales

_INSTALLED = False
_NATIVE: dict[int, NativeMoe] = {}
_SETTINGS: Any = None
_STORAGE: dict[int, dict[str, Any]] = {}
_SCRATCH: dict[tuple[int, int], torch.Tensor] = {}
_LAYERS: list[dict[str, Any]] = []
_RAW_STORAGE: list[dict[str, Any]] = []


def native(kind: int) -> NativeMoe:
    """Load the explicitly selected native library inside a GPU worker."""
    if kind not in _NATIVE:
        if kind != 1:
            raise ValueError("Only prepared ZipServ MXFP4 descriptors use the native consumer")
        _NATIVE[kind] = NativeMoe(_SETTINGS.library)
    return _NATIVE[kind]


def _launch_moe(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
    output: torch.Tensor, temporary: torch.Tensor, locks: torch.Tensor,
    sorted_ids: torch.Tensor, expert_ids: torch.Tensor, padded: torch.Tensor,
    routing_weights: torch.Tensor, m: int, n: int, k: int, experts: int,
    topk: int, block: int, multiply: bool, tile_k: int, tile_n: int,
    blocks_per_sm: int, kind: int,
) -> None:
    """Launch one exact grouped GEMM on the caller's CUDA stream without allocating."""
    tensors = (x, weight, scales, output, temporary, locks, sorted_ids, expert_ids,
               padded, routing_weights)
    library = native(kind).library
    status = library.mxwave_moe_run(
        *(t.data_ptr() for t in tensors), m, n, k, experts, topk, block, int(multiply),
        x.device.index, tile_k, tile_n, blocks_per_sm,
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if status:
        raise RuntimeError(library.mxwave_moe_last_error().decode())


_MUTATIONS = ("output", "temporary", "locks")
# Keep the Library alive for the lifetime of captured graphs and model workers.
_DISPATCH_LIBRARY = torch.library.Library("zipserv_mxfp4", "DEF")
_DISPATCH_LIBRARY.define(
    "moe_gemm" + torch.library.infer_schema(_launch_moe, mutates_args=_MUTATIONS))
_DISPATCH_LIBRARY.impl("moe_gemm", _launch_moe, "CUDA")


@torch.library.register_fake("zipserv_mxfp4::moe_gemm")
def _fake_moe_gemm(*args: Any, **kwargs: Any) -> None:
    """Preserve mutation aliases without inspecting storage during compilation."""
    return None


_GEMM_OP = torch.ops.zipserv_mxfp4.moe_gemm.default


@functools.lru_cache(maxsize=1024)
def _plan(m: int, n: int, k: int, block: int, topk: int, experts: int,
          device: int) -> tuple[int, int, int]:
    return native(1).plan(m, n, k, block, topk, experts, device)


def _temporary(x: torch.Tensor) -> torch.Tensor:
    stream = torch.cuda.current_stream(x.device).cuda_stream
    key = (x.device.index, stream)
    if key not in _SCRATCH:
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        # Maximum stock non-atomic FP32 reduction scratch for every supported
        # block (the eight-token case doubles its smaller scratch requirement).
        _SCRATCH[key] = torch.empty(sms * 4 * 64 * 256, dtype=torch.float32, device=x.device)
    return _SCRATCH[key]


def _replace_config_scales(method: Any, old: torch.Tensor, replacement: torch.Tensor) -> int:
    """Replace every shared/cloned quant descriptor without retaining the expanded tensor."""
    queue, seen, count = [method.moe_quant_config, method.moe_kernel], set(), 0
    pointer = old.data_ptr()
    while queue:
        value = queue.pop()
        if id(value) in seen or value is None:
            continue
        seen.add(id(value))
        if len(seen) > 512:
            raise RuntimeError("Unexpected MoE descriptor object graph")
        if isinstance(value, (torch.Tensor, str, bytes, int, float, bool, type)):
            continue
        if isinstance(value, dict):
            queue.extend(value.values())
        elif isinstance(value, (tuple, list)):
            queue.extend(value)
        elif type(value).__module__.startswith("vllm.model_executor.layers.fused_moe"):
            for key, child in vars(value).items():
                if key == "scale" and isinstance(child, torch.Tensor) and child.data_ptr() == pointer:
                    setattr(value, key, replacement)
                    count += 1
                elif not isinstance(child, torch.Tensor):
                    queue.append(child)
    return count


def prepare_layer(method: Any, layer: torch.nn.Module, mode: str) -> None:
    """Replace eligible prepared scales and every owning descriptor, or retain stock storage."""
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
    config = get_current_vllm_config()
    label = getattr(layer, "prefix", getattr(method.moe, "layer_name", "unknown"))
    entry: dict[str, Any] = {"layer": str(label), "ordinal": len(_LAYERS), "mode": mode, "scales": []}
    common_fallback = None
    if method.use_cutlass_mxfp4 or method.experts_cls is not MarlinExperts:
        common_fallback = "Only the Marlin backend has a fused scale consumer"
    elif method.group_size != 32:
        common_fallback = "Native consumer requires group-32 E8M0 scales"
    elif config.model_config.dtype != torch.bfloat16:
        common_fallback = "Native consumer requires BF16 activations"
    elif config.lora_config is not None:
        common_fallback = "Expert LoRA has not been qualified"
    elif config.parallel_config.enable_expert_parallel:
        common_fallback = "Expert parallelism has not been qualified"
    elif getattr(method.moe, "has_bias", False):
        common_fallback = "Biased expert GEMMs have not been qualified"
    offload = getattr(config, "offload_config", None)
    if offload is not None and (offload.uva.cpu_offload_gb > 0
                               or offload.prefetch.offload_group_size > 0):
        common_fallback = "Weight offloading cannot relocate native scale pointer descriptors"
    for name in ("w13_weight_scale", "w2_weight_scale"):
        raw = getattr(layer, name)
        record: dict[str, Any] = {"name": name, "shape": list(raw.shape),
            "raw_bytes": raw.numel() * raw.element_size(),
            "retained_bytes": raw.numel() * raw.element_size(), "compressed": False}
        raw_sha = hashlib.sha256(raw.contiguous().view(torch.uint8).cpu().numpy()).hexdigest()
        record["source_sha256"] = raw_sha
        entry["scales"].append(record)
        if mode == "off":
            continue
        if common_fallback:
            record["fallback"] = common_fallback
            continue
        if (raw.dtype != torch.float8_e8m0fnu or raw.ndim != 3 or not raw.is_contiguous()
                or min(raw.shape) < 1 or raw.shape[2] % 64 or (raw.shape[1] * 32) % 64
                or raw.numel() >= 2**31):
            record["fallback"] = "Scale layout is outside the supported E8M0 group-32 contract"
            continue
        try:
            packed = pack_scales(raw)
        except ValueError as error:
            record["fallback"] = str(error)
            continue
        raw_allocated = (raw.numel() + 511) // 512 * 512
        retained_allocated = sum((t.numel() * t.element_size() + 511) // 512 * 512
                                 for t in (packed.records, packed.exceptions, packed.descriptor))
        saving = 100 * (1 - retained_allocated / raw_allocated)
        if saving < _SETTINGS.min_saving_percent:
            record.update(fallback="Packed allocation does not meet the requested saving",
                          candidate_retained_bytes=packed.retained_bytes,
                          estimated_scale_saving_percent=saving)
            continue
        decoded = native(1).decode(packed)
        if not torch.equal(decoded, raw.view(torch.uint8)):
            raise RuntimeError("Prepared expert scales changed bytes in the fused native decoder")
        del decoded
        replacement = torch.nn.Parameter(packed.descriptor, requires_grad=False)
        references = _replace_config_scales(method, raw, replacement)
        if references < 1:
            raise RuntimeError("No prepared quant descriptor owned the original expert scales")
        layer.register_buffer("zipserv_" + name + "_records", packed.records)
        layer.register_buffer("zipserv_" + name + "_exceptions", packed.exceptions)
        setattr(layer, name, replacement)
        key = id(replacement)
        _STORAGE[key] = {"shape": tuple(raw.shape), "kind": 1, "packed": packed,
                         "source_sha256": raw_sha, "raw_owner": weakref.ref(raw),
                         "layer_ordinal": entry["ordinal"], "name": name,
                         "replacement": replacement, "eager_or_capture_calls": 0}
        record.update(compressed=True, retained_bytes=packed.retained_bytes,
                      gpu_bytes_exact=True, source_sha256=raw_sha,
                      quant_descriptor_references=references,
                      estimated_scale_saving_percent=saving)
    for record in entry["scales"]:
        if not record["compressed"]:
            _RAW_STORAGE.append({"ordinal": entry["ordinal"], "name": record["name"],
                                 "owner": weakref.ref(getattr(layer, record["name"])),
                                 "source_sha256": record["source_sha256"]})
    _LAYERS.append(entry)
    layer.zipserv_mxfp4_qualification = entry


def report(verify: bool = False) -> dict[str, Any]:
    """Report complete per-worker coverage, ownership, memory and optional byte rechecks."""
    if verify:
        torch.cuda.synchronize()
        gc.collect()  # Audit live ownership after reclaiming unreachable descriptor cycles.
    checks = []
    raw_checks = []
    for metadata in _RAW_STORAGE:
        tensor = metadata["owner"]()
        row = {"ordinal": metadata["ordinal"], "name": metadata["name"],
               "stock_owner_alive": tensor is not None}
        if verify:
            if tensor is None:
                raise RuntimeError("Stock fallback scale owner disappeared")
            actual = hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy()).hexdigest()
            if actual != metadata["source_sha256"]:
                raise RuntimeError("Stock fallback scale bytes changed")
            row["gpu_bytes_exact"] = True
        raw_checks.append(row)
    for metadata in _STORAGE.values():
        raw_released = metadata["raw_owner"]() is None
        row = {"ordinal": metadata["layer_ordinal"], "name": metadata["name"],
               "raw_owner_released": raw_released,
               "eager_or_capture_calls": metadata["eager_or_capture_calls"]}
        if verify:
            decoded = native(1).decode(metadata["packed"])
            actual = hashlib.sha256(decoded.cpu().numpy()).hexdigest()
            del decoded
            if actual != metadata["source_sha256"] or not raw_released:
                raise RuntimeError(
                    f"MXFP4 audit failed for layer {metadata['layer_ordinal']} {metadata['name']}: "
                    f"bytes_exact={actual == metadata['source_sha256']}, raw_released={raw_released}")
            row["gpu_bytes_exact"] = True
        checks.append(row)
    result = {"mode": _SETTINGS.mode, "layers": _LAYERS, "layers_seen": len(_LAYERS),
              "compressed_scale_groups": len(_STORAGE), "ownership": checks,
              "stock_scale_groups": len(_RAW_STORAGE), "stock_ownership": raw_checks,
              "scale_raw_bytes": sum(x["raw_bytes"] for e in _LAYERS for x in e["scales"]),
              "scale_retained_bytes": sum(x["retained_bytes"] for e in _LAYERS for x in e["scales"]),
              "scratch_bytes": sum(x.numel() * x.element_size() for x in _SCRATCH.values()),
              "scratch_streams": len(_SCRATCH), "cuda_allocated_bytes": torch.cuda.memory_allocated(),
              "cuda_reserved_bytes": torch.cuda.memory_reserved(),
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    result["scale_saving_after_scratch_bytes"] = (result["scale_raw_bytes"]
        - result["scale_retained_bytes"] - result["scratch_bytes"])
    return result


class ScaleWorker:
    """Optional vLLM worker RPC extension for read-only FP4 audits."""

    def zipserv_mxfp4_report(self, verify: bool = False) -> dict[str, Any]:
        """Return complete local-rank coverage and exact scale verification."""
        return report(verify)

    def zipserv_weight_inventory(self) -> dict[str, Any]:
        """Inventory every registered tensor, counting aliased storage only once."""
        model = self.model_runner.get_model()
        modules, storages = [], {}
        for prefix, module in model.named_modules():
            tensors = [*(('parameter', name, value) for name, value in
                         module.named_parameters(recurse=False, remove_duplicate=False)),
                       *(('buffer', name, value) for name, value in
                         module.named_buffers(recurse=False, remove_duplicate=False))]
            if not tensors:
                continue
            method = getattr(module, 'quant_method', None)
            scheme = getattr(module, 'scheme', getattr(method, 'scheme', None))
            kernel = getattr(scheme, 'kernel', getattr(method, 'kernel', None))
            row = {'prefix': prefix, 'class': type(module).__module__ + '.' + type(module).__name__,
                   'method': type(method).__name__ if method is not None else None,
                   'scheme': type(scheme).__name__ if scheme is not None else None,
                   'kernel': type(kernel).__name__ if kernel is not None else None,
                   'tensors': []}
            for attribute in ('input_size_per_partition', 'output_size_per_partition'):
                value = getattr(module, attribute, None)
                if isinstance(value, int):
                    row[attribute] = value
            for kind, name, tensor in tensors:
                storage = tensor.untyped_storage()
                key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
                if key not in storages:
                    storages[key] = {'id': len(storages), 'device': str(tensor.device),
                                     'bytes': storage.nbytes(), 'owners': []}
                storages[key]['owners'].append(prefix + '.' + name)
                row['tensors'].append({'name': name, 'kind': kind, 'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype), 'logical_bytes': tensor.numel() * tensor.element_size(),
                    'storage_id': storages[key]['id'], 'contiguous': tensor.is_contiguous()})
            modules.append(row)
        return {'modules': modules, 'storages': list(storages.values()),
                'unique_cuda_storage_bytes': sum(row['bytes'] for row in storages.values()
                                                 if row['device'].startswith('cuda')),
                'cuda_allocated_bytes': torch.cuda.memory_allocated(),
                'scope': 'Registered model parameters and buffers; allocator total also includes KV, graphs and workspaces.'}


def install(settings: Any) -> None:
    """Install explicit on/off observation; only on-mode owns compressed scale tensors."""
    global _INSTALLED, _SETTINGS
    if _INSTALLED:
        if settings != _SETTINGS:
            raise RuntimeError("Cannot change active MXFP4 settings within a model process")
        return
    _SETTINGS = settings
    mode = settings.mode
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    if mode == "on":
        native(1)  # Reject incompatible libraries before any layer releases raw scales.

    module = importlib.import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe."
        "compressed_tensors_moe_w4a4_mxfp4")
    original_prepare = module.CompressedTensorsW4A4Mxfp4MoEMethod.process_weights_after_loading
    if (original_prepare.__module__ != module.__name__
            or hasattr(original_prepare, "__wrapped__")):
        raise RuntimeError("Another plugin already replaced MXFP4 preparation; avoid double ownership")

    @functools.wraps(original_prepare)
    def prepare(method: Any, layer: torch.nn.Module) -> None:
        original_prepare(method, layer)
        prepare_layer(method, layer, mode)

    module.CompressedTensorsW4A4Mxfp4MoEMethod.process_weights_after_loading = prepare
    _INSTALLED = True
    if mode == "off":
        return
    import inspect

    original_gemm = ops.moe_wna16_marlin_gemm
    signature = inspect.signature(original_gemm)
    scale_position = list(signature.parameters).index("b_scales")

    @functools.wraps(original_gemm)
    def gemm(*args: Any, **kwargs: Any) -> torch.Tensor:
        candidate = kwargs.get("b_scales", args[scale_position] if len(args) > scale_position else None)
        if id(candidate) not in _STORAGE:
            return original_gemm(*args, **kwargs)
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        p = call.arguments
        scales = p["b_scales"]
        if id(scales) not in _STORAGE:
            return original_gemm(*args, **kwargs)
        metadata = _STORAGE[id(scales)]
        x, output, weight = p["input"], p["output"], p["b_qweight"]
        if (any(p[key] is not None and p[key].numel()
                for key in ("b_bias", "a_scales", "global_scale", "b_qzeros"))
                or p["use_atomic_add"] or not p["use_fp32_reduce"] or p["is_zp_float"]
                or x.dtype != torch.bfloat16 or output is None
                or p["b_q_type"] != scalar_types.float4_e2m1f):
            raise ValueError("Grouped GEMM is outside the exact-scale runtime contract")
        m, n, k = (int(p[key]) for key in ("size_m", "size_n", "size_k"))
        experts = metadata["shape"][0]
        block, topk = int(p["moe_block_size"]), int(p["top_k"])
        if (m < 0 or m * topk >= 2**31
                or n != metadata["shape"][2] or k != metadata["shape"][1] * 32
                or tuple(weight.shape) != (experts, k // 16, n * 2)
                or weight.dtype != torch.int32 or output.dtype != torch.bfloat16
                or x.ndim != 2 or x.shape[1] != k or x.shape[0] < m
                or output.numel() < m * topk * n or topk < 1
                or block not in (8, 16, 32, 48, 64)):
            raise ValueError("Grouped scale ownership does not match the GEMM weight")
        if m == 0:
            return output
        integer_buffers = (p["workspace"], p["sorted_token_ids"], p["expert_ids"],
                           p["num_tokens_past_padded"])
        tensors = (x, weight, scales, output, p["topk_weights"], *integer_buffers)
        if (x.device.type != "cuda" or any(t.device != x.device or not t.is_contiguous() for t in tensors)
                or any(t.dtype != torch.int32 for t in integer_buffers)
                or p["topk_weights"].dtype != torch.float32
                or p["topk_weights"].numel() < m * topk
                or p["sorted_token_ids"].numel() < m * topk
                or p["expert_ids"].numel() * block < m * topk
                or p["num_tokens_past_padded"].numel() != 1
                or p["workspace"].numel() < torch.cuda.get_device_properties(x.device).multi_processor_count * 4):
            raise ValueError("Grouped GEMM buffers do not satisfy the native device/layout contract")
        plan = _plan(m, n, k, block, topk, experts, x.device.index)
        if p["thread_k"] > 0 or p["thread_n"] > 0:
            plan = (p["thread_k"], p["thread_n"], p["blocks_per_sm"])
        metadata["eager_or_capture_calls"] += 1  # Graph replays do not re-enter Python.
        _GEMM_OP(x, weight, scales, output, _temporary(x), p["workspace"],
                 p["sorted_token_ids"], p["expert_ids"], p["num_tokens_past_padded"],
                 p["topk_weights"], m, n, k, experts, topk, block,
                 p["mul_topk_weights"], *plan, metadata["kind"])
        return output

    ops.moe_wna16_marlin_gemm = gemm
