# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Explicit experimental BF16/FP8 additions to a mixed quantized checkpoint."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


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
    from .compatibility import check_upstream
    # A BF16-only experiment must not depend on unrelated FP4/FP8 hooks.
    adapters = ["mixed_bf16"]
    if selected.fp8 == "on":
        adapters.append("mixed_fp8")
    check_upstream(*adapters)
    from .mixed_backend import install
    install(selected)
