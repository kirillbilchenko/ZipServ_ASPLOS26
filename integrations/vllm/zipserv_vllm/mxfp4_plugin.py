# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Inert-by-default vLLM entry point for lossless MXFP4 expert-scale storage."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Settings:
    """Immutable, worker-inherited controls with no hidden model-name decisions."""

    mode: str
    library: Path | None
    min_saving_percent: float


def settings(environment: Mapping[str, str]) -> Settings | None:
    """Parse an explicit mode without importing Torch, vLLM or a CUDA library."""
    mode = environment.get("ZIPSERV_MXFP4_SCALES")
    if mode is None:
        return None
    if mode not in ("off", "on"):
        raise ValueError("ZIPSERV_MXFP4_SCALES must be off or on")
    minimum = float(environment.get("ZIPSERV_MXFP4_MIN_SAVING_PERCENT", "1"))
    if not 0 < minimum < 100:
        raise ValueError("MXFP4 minimum scale saving must be finite and between 0 and 100 percent")
    path = environment.get("ZIPSERV_MXFP4_LIBRARY")
    library = Path(path).expanduser().resolve() if path else None
    if mode == "on" and (library is None or not library.is_file()):
        raise ValueError("Set ZIPSERV_MXFP4_LIBRARY to the built native scale consumer")
    if (environment.get("ZIPSERV_SCALE_DICTIONARY") == "1"
            or environment.get("MXWAVE_GLM53_ENABLE") == "1"
            or environment.get("MXWAVE_MOE_MODE", "stock") != "stock"):
        raise ValueError("Select one expert-scale owner; disable the MxWave/cache scale adapter")
    return Settings(mode, library, minimum)


def check_upstream() -> None:
    """Check the supported vLLM API files before wrapping internal loader and GEMM functions."""
    from packaging.version import Version
    if Version(importlib.metadata.version("vllm")).release != (0, 32, 0):
        raise RuntimeError("ZipServ MXFP4 currently supports vLLM 0.32.0")
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate installed vLLM")
    root = Path(spec.origin).parent
    contract = {
        "_custom_ops.py": "36428c27bad9bf5ffe5debf9e34bccbc888adb68ad3eb972f6c9ff080e7f683e",
        "model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/"
        "compressed_tensors_moe_w4a4_mxfp4.py":
            "c26a798139a6a24ae7802ae5e5c8cca180d8a3ad75ad1c06b0e69062d47d8eef",
        "model_executor/layers/fused_moe/experts/marlin_moe.py":
            "e048731e130ecc3100be0ab4a168777a7697131b41538f9d9b6a73c4f66e14ff",
    }
    for name, expected in contract.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Installed vLLM MXFP4 API differs from the reviewed 0.32.0 source: {name}")


def register() -> None:
    """Install only after an explicit on/off selection and a verified runtime API."""
    selected = settings(os.environ)
    if selected is None:
        return
    check_upstream()
    from .mxfp4_backend import install
    install(selected)
