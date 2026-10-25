# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Explicit experimental BF16/FP8 additions to a mixed quantized checkpoint."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Settings:
    bf16: str
    fp8: str
    bf16_targets: tuple[str, ...]
    fp8_targets: tuple[str, ...]
    min_saving_percent: float


def settings(environment: Mapping[str, str]) -> Settings | None:
    if not any(k in environment for k in ("ZIPSERV_MIXED_BF16", "ZIPSERV_MIXED_FP8")):
        return None
    modes = [environment.get(k, "off") for k in ("ZIPSERV_MIXED_BF16", "ZIPSERV_MIXED_FP8")]
    if any(mode not in ("off", "on") for mode in modes):
        raise ValueError("Mixed BF16/FP8 modes must be off or on")
    targets = []
    for key in ("ZIPSERV_BF16_TARGETS", "ZIPSERV_FP8_TARGETS"):
        value = json.loads(environment.get(key, "[]"))
        if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
            raise ValueError(f"{key} must be a JSON list of reviewed module-name patterns")
        targets.append(tuple(value))
    for mode, selected, key in zip(modes, targets, ("ZIPSERV_LIBRARY", "ZIPSERV_FP8_LIBRARY"), strict=True):
        if mode == "on":
            if not selected:
                raise ValueError("Explicit compression requires reviewed target patterns; inspect direct weight readers first")
            if not Path(environment.get(key, "")).is_file():
                raise ValueError(f"Set {key} to its native library")
    minimum = float(environment.get("ZIPSERV_MIXED_MIN_SAVING_PERCENT", "1"))
    if not 0 < minimum < 100:
        raise ValueError("Minimum saving must be finite and between zero and 100 percent")
    return Settings(*modes, *targets, minimum)


def register() -> None:
    selected = settings(os.environ)
    if selected is None:
        return
    from .mxfp4_plugin import check_upstream
    check_upstream()
    spec = importlib.util.find_spec("vllm")
    root = Path(spec.origin).parent
    contract = {
        "model_executor/layers/linear.py": "7a9b90937865fa35d955ffc2017f23d3997bc4fba954e2667870b1b4854a38e0",
        "model_executor/layers/vocab_parallel_embedding.py": "10e17024bd5a5915e0d486e7a78a96d62e77bce803e1afd27feae7ad2042edc0",
        "model_executor/kernels/linear/mxfp8/marlin.py": "299c549e3e3830fe3b52ef83eca565356d6bd42e2a85805300a1bd0fe6a67e0c",
        "model_executor/layers/quantization/utils/marlin_utils_fp8.py": "a5cc60882dc32851b2cb7d573e2b28980319a99ef816fd6ed9925b6fabfffd9a",
        "model_executor/layers/quantization/utils/marlin_utils.py": "e60713c15080347b01425f90eaf03840ca80a1cd226ec57e526e8f6286eeeb22",
        "v1/worker/workspace.py": "bcf9ea207279c38a476be4ca74137a9d3a973f1fd97d85639a383357f881eb06",
        "model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py":
            "ec5de2adc6b7452377a08e685522224aee8c345dc3312fb15cece662b40d3a60",
        "v1/worker/gpu_worker.py": "e970aa29a738aa15cd963d9f9202ca6d3d504e98fd344959fd998e940472a5a0",
    }
    for name, expected in contract.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Installed mixed-weight API differs from reviewed source: {name}")
    from .mixed_backend import install
    install(selected)
