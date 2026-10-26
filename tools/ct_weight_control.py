# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Observe optional weight representations on the exact, legacy CT runtime.

This private experiment imports the original CT observer and adapters from
read-only mounts. It does not install the vLLM 0.32 ZipServ integration.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

VERSION = "0.1.dev20051+g487ecf187"
DISABLED = (
    "GLM_DS_DRAFT_HEAD_FP8", "GLM_DS_DRAFT_NVFP4", "GLM_DS_DRAFT_HEAD_FP4",
    "GLM_CERT_HEAD", "GLM_CERT_HEAD_DRAFT", "GLM_MHC_BF16W",
)
_ORIGINAL_REPORT = None


def tensor_record(value):
    """Read exact bytes in bounded host chunks without a decoded GPU copy."""
    import torch

    if not value.is_contiguous():
        raise RuntimeError("Audited resident weights must be contiguous")
    raw = value.detach().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, raw.numel(), 16 * 1024**2):
        digest.update(raw[start:start + 16 * 1024**2].cpu().numpy().tobytes())
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": raw.numel(), "sha256": digest.hexdigest()}


def validate_environment():
    import vllm

    mode = os.environ.get("ZIPSERV_CT_CONTROL_MODE")
    if vllm.__version__ != VERSION or mode not in ("off", "ct"):
        raise RuntimeError("Require the exact CT runtime and an explicit control mode")
    if os.environ.get("GLM_LV_DRAFT_FP8_KV") != "1":
        raise RuntimeError("Keep the block-FP8 drafter context-KV correctness fix")
    if mode == "off":
        required = {"MXWAVE_COMPARE_ARM": "A", "MXWAVE_PALETTE_MODEL": "0",
                    "MXWAVE_MOE_MODE": "stock", "MXWAVE_GLM53_VOCAB_MODE": "stock",
                    "MXWAVE_GLM53_KDA_MODE": "stock", **dict.fromkeys(DISABLED, "0")}
        if any(os.environ.get(key) != value for key, value in required.items()):
            raise RuntimeError("An optional weight representation is still enabled")
        if os.environ.get("GLM_CERT_HEAD_DUMP"):
            raise RuntimeError("Disable target-head dump hooks in the control")
    elif os.environ.get("MXWAVE_COMPARE_ARM") != "B":
        raise RuntimeError("The CT arm must use its original adapter recipe")
    if any(name.startswith("zipserv_vllm") for name in sys.modules):
        raise RuntimeError("The modern ZipServ plugin must not be loaded in this control")
    return mode


def report_worker(worker):
    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    mode = validate_environment()
    if _ORIGINAL_REPORT is None:
        raise RuntimeError("CT observation hooks are not installed")
    recipe = _ORIGINAL_REPORT(worker)
    runner = worker.model_runner
    model = runner.model
    language = getattr(model, "language_model", model)
    if not recipe["speculation"] or not recipe["physical_graph_token_counts"]:
        raise RuntimeError("The CT speculative decoder and captured graphs must be active")
    if recipe["draft_context_dtype"] != "torch.bfloat16":
        raise RuntimeError("The drafter context-KV buffer must use scaled BF16 weights")
    result = {"status": "verified", "mode": mode, "rank": recipe["rank"],
              "ct_recipe": recipe, "vllm_version": VERSION,
              "checkpoint_quantization_retained": True,
              "optional_weight_flags": {key: os.environ.get(key) for key in DISABLED},
              "correctness_fix_draft_fp8_kv": os.environ["GLM_LV_DRAFT_FP8_KV"]}
    if mode == "off":
        modules = dict(model.named_modules())
        reference_bytes = Path("/kda-runtime/rounding-reference.json").read_bytes()
        reference = json.loads(reference_bytes)["ranks"][str(recipe["rank"])]
        if len(reference) != 68 or len({row["name"] for row in reference}) != 68:
            raise RuntimeError("Expected all 68 audited KDA matrices per rank")
        kda = []
        for expected in reference:
            layer = modules[expected["name"]]
            if (type(layer.quant_method) is not UnquantizedLinearMethod
                    or layer.weight.dtype != torch.bfloat16
                    or hasattr(layer, "weight_scale")):
                raise RuntimeError("KDA must retain its original dense BF16 weights")
            row = {"name": expected["name"], **tensor_record(layer.weight)}
            if row["sha256"] != expected["source_sha256"] or row["shape"] != expected["shape"]:
                raise RuntimeError("KDA source bytes differ: " + expected["name"])
            kda.append(row)
        scales = []
        for name, module in modules.items():
            for attribute in ("w13_weight_scale", "w2_weight_scale"):
                value = getattr(module, attribute, None)
                if value is not None:
                    if value.dtype != torch.float8_e8m0fnu or value.ndim != 3:
                        raise RuntimeError("Expert scales must retain native E8M0 storage")
                    scales.append({"name": name + "." + attribute, **tensor_record(value)})
            method = getattr(module, "quant_method", None)
            origin = type(method).__module__
            if any(text in origin for text in ("zipserv", "kda_mxfp8_runtime", "model_plugin")):
                raise RuntimeError("Unexpected weight adapter: " + name + ": " + origin)
        embedding = language.model.embed_tokens.weight
        if len(scales) != 84 or embedding is None or embedding.dtype != torch.bfloat16:
            raise RuntimeError("Require all 84 raw scale groups and the dense BF16 embedding")
        draft = runner.speculator.model
        owners = (runner, draft, draft.model, language.lm_head)
        for owner in owners:
            for name in ("_glm_ds_head_fp8_twin", "_glm_head_twin", "_glm_cert"):
                value = getattr(owner, name, None)
                if value is not None and not isinstance(value, str):
                    raise RuntimeError("An optional compressed head copy remains: " + name)
        mhc = sys.modules.get("glm_mhc_bf16w")
        if mhc is not None and mhc.S.get("twins"):
            raise RuntimeError("mHC BF16 weight twins remain allocated")
        if hasattr(worker, "_mxwave_same_runtime_kda"):
            raise RuntimeError("Post-load KDA conversion unexpectedly ran")
        result.update(kda=kda, raw_scales=scales, embedding=tensor_record(embedding),
                      head=tensor_record(language.lm_head.weight),
                      kda_source_reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                      optional_head_twins=0, mhc_bf16_twins=0)
    torch.cuda.synchronize()
    result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                  cuda_reserved_bytes=torch.cuda.memory_reserved(),
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    worker._zipserv_ct_control_report = result
    return result


def install():
    global _ORIGINAL_REPORT
    validate_environment()
    if _ORIGINAL_REPORT is not None:
        return
    # This is checkpoint interpretation metadata, not compression. CT normally
    # installs it inside the arm-B adapter path; the raw control needs it too.
    if os.environ.get("MXWAVE_GLM53_ATTENTION_MAPPING") != "1":
        raise RuntimeError("Keep CT's fused-attention checkpoint mapping repair")
    if os.environ["ZIPSERV_CT_CONTROL_MODE"] == "off":
        import attention_mapping
        attention_mapping.install()
    import same_runtime_plugin

    _ORIGINAL_REPORT = same_runtime_plugin.SameRuntimeWorker.mxwave_same_runtime_report
    same_runtime_plugin.SameRuntimeWorker.mxwave_same_runtime_report = report_worker
    same_runtime_plugin.install()


class ControlWorker:
    """Private RPC methods; the server is bound to loopback only."""

    def ct_weight_control_report(self):
        return report_worker(self)
