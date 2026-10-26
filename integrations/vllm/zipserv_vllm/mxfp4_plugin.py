# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Inert-by-default vLLM entry point for lossless MXFP4 expert-scale storage."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


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
    """Check the reviewed loader and native-GEMM source contract."""
    from .compatibility import check_upstream as check
    check("mxfp4")


def register() -> None:
    """Install only after an explicit on/off selection and a verified runtime API."""
    selected = settings(os.environ)
    if selected is None:
        return
    check_upstream()
    from .mxfp4_backend import install
    install(selected)
