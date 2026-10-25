# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Experimental post-load mixed storage: explicit targets, exact bytes, source release."""
from __future__ import annotations

from collections import Counter
import fnmatch
import functools
import gc
import hashlib
import weakref

import numpy as np
import torch

from .mxfp4_backend import ScaleWorker
from .runtime import BUFFER_NAMES, PackedWeight, compress, decompress, linear

_INSTALLED = False
_SETTINGS = None


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


class BF16Method:
    """Use fused decoding for small batches; preserve stock GEMM for large prefills."""

    def __init__(self, packed: PackedWeight, stock_gemm, logical_n: int):
        self.packed, self.stock_gemm, self.logical_n = packed, stock_gemm, logical_n
        self.calls = Counter()

    def apply(self, layer, x, bias=None):
        m = x.numel() // x.shape[-1]
        if m > 768:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("The BF16 mixed experiment qualifies decode-only CUDA graphs")
            self.calls["decode_then_stock"] += 1
            return self.stock_gemm(layer, x, decompress(self.packed)[:self.logical_n], bias)
        self.calls["fused"] += 1
        desired = 8 if m <= 32 else 4 if m <= 128 else 1
        split = next(s for s in (desired, 4, 2, 1) if s <= desired and (self.packed.k // 64) % s == 0)
        output = linear(x, self.packed, split_k=split)
        if self.logical_n != self.packed.m:
            output = output[..., :self.logical_n].contiguous()
        return output if bias is None else output + bias


def _prepare(worker, selected) -> None:
    from vllm import envs
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod, dispatch_unquantized_gemm
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
    from . import fp8_codec, fp8_runtime

    config = worker.vllm_config
    offload = config.offload_config
    compilation = config.compilation_config
    if (config.lora_config is not None or config.weight_transfer_config is not None
            or offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0
            or config.model_config.enable_sleep_mode or envs.VLLM_BATCH_INVARIANT):
        raise ValueError("Mixed experiment does not support LoRA, hot reload, offload, sleep mode or batch-invariant GEMM")
    if (config.model_config.dtype != torch.bfloat16 or int(compilation.mode) != 0
            or compilation.cudagraph_mode.name not in ("NONE", "FULL_DECODE_ONLY")):
        raise ValueError("Mixed experiment currently requires BF16 activations and eager or full-decode-only graphs with compilation mode 0")
    model = worker.model_runner.get_model()
    storage_counts = Counter(t.untyped_storage()._cdata for module in model.modules()
        for t in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)))
    storage_ids = {key: i for i, key in enumerate(storage_counts)}
    rows, owners = [], []
    allocated_before = torch.cuda.memory_allocated()
    worker._zipserv_mixed = {"rows": rows, "owners": owners, "settings": selected,
                            "allocated_before": allocated_before, "status": "preparing"}
    for name, layer in model.named_modules():
        raw = getattr(layer, "weight", None)
        if not isinstance(raw, torch.Tensor):
            continue
        method = getattr(layer, "quant_method", None)
        scheme = getattr(method, "scheme", None)
        # CompressedTensorsLinearMethod stores the scheme on the layer in vLLM;
        # retain both supported ownership paths for an explicit type check.
        scheme = getattr(layer, "scheme", scheme)
        kernel = getattr(scheme, "kernel", None)
        bf16 = ((isinstance(layer, LinearBase) and type(method) is UnquantizedLinearMethod)
                or (type(layer) is ParallelLMHead and type(method) is UnquantizedEmbeddingMethod))
        fp8 = type(kernel) is MarlinMxfp8LinearKernel
        if not (bf16 or fp8):
            continue
        kind = "bf16" if bf16 else "fp8"
        mode = getattr(selected, kind)
        patterns = getattr(selected, kind + "_targets")
        row = {"name": name, "kind": kind, "shape": list(raw.shape), "raw_bytes": raw.nbytes,
               "retained_bytes": raw.nbytes, "compressed": False, "device": str(raw.device),
               "source_storage_id": storage_ids[raw.untyped_storage()._cdata]}
        rows.append(row)
        reason = None
        if mode == "off":
            reason = "disabled"
        elif not any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            reason = "not in the reviewed target list"
        elif raw.device.type != "cuda" or not raw.is_contiguous() or raw.ndim != 2:
            reason = "unsupported source layout"
        elif storage_counts[raw.untyped_storage()._cdata] != 1:
            reason = "shared registered storage"
        elif isinstance(layer, GateLinear) or getattr(layer, "_use_min_latency_gemm", False):
            reason = "specialized forward reads raw weights directly"
        if reason:
            row["fallback"] = reason
            continue
        row["source_sha256"] = tensor_hash(raw)
        if bf16:
            n, k = raw.shape
            padded_n = (n + 63) // 64 * 64
            if raw.dtype != torch.bfloat16 or min(n, k) <= 0 or k % 64 or padded_n * k > 2**31 - 4097:
                row["fallback"] = "unsupported BF16 shape/dtype"
                continue
            cpu = raw.detach().cpu()
            if n != padded_n:
                cpu = torch.nn.functional.pad(cpu, (0, 0, 0, padded_n - n))
            packed = compress(cpu)
            del cpu
            row["padded_shape"] = [padded_n, k]
            allocated = sum((t.nbytes + 511) // 512 * 512 for t in packed.buffers)
            row["candidate_bytes"] = packed.storage_bytes
        else:
            k, n = raw.shape[0] * 16, raw.shape[1] // 4
            scales = layer.weight_scale
            if (raw.dtype != torch.int32 or raw.shape[1] % 4 or min(n, k) < 64 or n % 64 or k % 64
                    or (n % 128 and k % 128) or n * k >= 2**31
                    or scales.dtype != torch.float8_e8m0fnu or tuple(scales.shape) != (k // 32, n)
                    or not scales.is_contiguous() or bool((scales.view(torch.uint8) == 255).any())):
                row["fallback"] = "unsupported MXFP8 Marlin layout"
                continue
            raw_bytes = raw.detach().view(torch.uint8).cpu().numpy().tobytes()
            encoded, accounting = fp8_codec.encode(raw_bytes)
            if fp8_codec.decode(encoded) != raw_bytes:
                raise RuntimeError(f"{name}: CPU FP8 roundtrip changed source bytes")
            row["codec"] = accounting
            allocated = (len(encoded) + 511) // 512 * 512
            row["candidate_bytes"] = len(encoded)
            del raw_bytes
        saving = 100 * (1 - allocated / ((raw.nbytes + 511) // 512 * 512))
        if saving < selected.min_saving_percent:
            row["fallback"] = "below requested allocation saving"
            if bf16:
                del packed
            else:
                del encoded
            continue
        if bf16:
            packed = packed.to(raw.device)
            restored = decompress(packed)[:n]
        else:
            data = torch.from_numpy(np.frombuffer(encoded, dtype=np.uint8).copy()).to(raw.device)
            del encoded
            restored = fp8_runtime.decode(data, n, k)
        if not torch.equal(restored.view(torch.uint8), raw.view(torch.uint8)):
            raise RuntimeError(f"{name}: GPU decoder changed source bytes")
        del restored
        owners.append((name, weakref.ref(raw)))
        if bf16:
            for field, buffer in zip(BUFFER_NAMES, packed.buffers, strict=True):
                layer.register_buffer("zipserv_mixed_bf16_" + field, buffer)
            stock_gemm = method._gemm_impl if type(method) is UnquantizedLinearMethod else dispatch_unquantized_gemm()
            layer.quant_method = BF16Method(packed, stock_gemm, n)
            row["retained_bytes"] = packed.storage_bytes
            del packed
        else:
            layer.register_buffer("zipserv_fp8_data", data)
            layer.zipserv_fp8_shape = (n, k)
            layer.zipserv_fp8_calls = Counter()
            row["retained_bytes"] = data.nbytes
            row["largest_per_call_decoded_weight_bytes"] = n * k
            del data
        layer.register_parameter("weight", None)
        row.update(compressed=True, gpu_bytes_exact=True)
        del raw
    # Do not leave the final loop variable holding a compressed source alive.
    raw = None
    gc.collect()
    torch.cuda.synchronize()
    if any(ref() is not None for _, ref in owners):
        raise RuntimeError("A compressed layer still has a live original weight owner")
    after = torch.cuda.memory_allocated()
    worker.model_runner.model_memory_usage += after - allocated_before
    worker._zipserv_mixed.update(status="prepared", allocated_after=after)


def install(selected) -> None:
    global _INSTALLED, _SETTINGS
    if _INSTALLED:
        if selected != _SETTINGS:
            raise RuntimeError("Cannot change mixed compression settings in an active worker")
        return
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
    original_load = Worker.load_model
    original_apply = MarlinMxfp8LinearKernel.apply_weights

    @functools.wraps(original_load)
    def load(worker, *args, **kwargs):
        result = original_load(worker, *args, **kwargs)
        _prepare(worker, selected)
        return result

    @functools.wraps(original_apply)
    def apply(kernel, layer, x, bias=None):
        if hasattr(layer, "zipserv_fp8_data"):
            from .fp8_runtime import apply as compressed
            return compressed(layer, x, bias)
        return original_apply(kernel, layer, x, bias)

    Worker.load_model = load
    MarlinMxfp8LinearKernel.apply_weights = apply
    _INSTALLED = True
    _SETTINGS = selected


class MixedWorker(ScaleWorker):
    def zipserv_mixed_report(self, verify: bool = False):
        from . import fp8_runtime
        state = self._zipserv_mixed
        model = self.model_runner.get_model()
        rows = [dict(row) for row in state["rows"]]
        if verify:
            for row in rows:
                if not row["compressed"]:
                    continue
                layer = model.get_submodule(row["name"])
                value = (decompress(layer.quant_method.packed) if row["kind"] == "bf16"
                         else fp8_runtime.decode(layer.zipserv_fp8_data, *layer.zipserv_fp8_shape))
                if row["kind"] == "bf16":
                    value = value[:row["shape"][0]]
                row["gpu_bytes_exact"] = tensor_hash(value) == row["source_sha256"]
                del value
                if not row["gpu_bytes_exact"]:
                    raise RuntimeError("Live compressed weight failed source verification")
        torch.cuda.synchronize()
        gc.collect()
        for row in rows:
            if row["compressed"] and row["kind"] == "bf16":
                row["eager_or_capture_calls"] = dict(model.get_submodule(row["name"]).quant_method.calls)
            elif row["compressed"]:
                row["eager_or_capture_calls"] = dict(model.get_submodule(row["name"]).zipserv_fp8_calls)
        unique = {row["source_storage_id"]: row for row in rows if row["device"].startswith("cuda")}
        return {"status": state["status"], "rows": rows,
            "raw_owners_released": all(ref() is None for _, ref in state["owners"]),
            "fp8_shared_scratch_bytes": fp8_runtime.scratch_bytes(),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "per_type": {kind: {
                "compressed_layers": sum(r["compressed"] for r in rows if r["kind"] == kind),
                "raw_bytes": sum(r["raw_bytes"] for r in unique.values() if r["kind"] == kind),
                "retained_bytes": sum(r["retained_bytes"] for r in unique.values() if r["kind"] == kind),
            } for kind in ("bf16", "fp8")},
            "limits": ["Raw fallback and other tensor types remain in the full weight inventory.",
                       "FP8 scales are unchanged. Large-prefill decoded weights are temporary.",
                       "CUDA peak includes loading; use a separate inference peak window."]}
