# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in NVFP4 scale adapter; importing this entry point is inert."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    mode: str
    library: Path | None
    min_saving_percent: float


def settings(environment: Mapping[str, str]) -> Settings | None:
    mode = environment.get("ZIPSERV_NVFP4_SCALES")
    if mode is None:
        return None
    if mode not in ("off", "on"):
        raise ValueError("ZIPSERV_NVFP4_SCALES must be off or on")
    minimum = float(environment.get("ZIPSERV_NVFP4_MIN_SAVING_PERCENT", "1"))
    if not 0 < minimum < 100:
        raise ValueError("NVFP4 minimum scale saving must be finite and between 0 and 100 percent")
    path = environment.get("ZIPSERV_NVFP4_LIBRARY")
    library = Path(path).expanduser().resolve() if path else None
    if mode == "on" and (library is None or not library.is_file()):
        raise ValueError("Set ZIPSERV_NVFP4_LIBRARY to the built delta5 scale consumer")
    if (environment.get("ZIPSERV_MXFP4_SCALES", "off") != "off"
            or environment.get("ZIPSERV_SCALE_DICTIONARY") == "1"
            or environment.get("MXWAVE_GLM53_ENABLE") == "1"
            or environment.get("MXWAVE_MOE_MODE", "stock") != "stock"):
        raise ValueError("Select one expert-scale owner; disable the other FP4/cache adapters")
    return Settings(mode, library, minimum)


def checkpoint_recipes(config: dict) -> dict[str, list[str]]:
    """Screen storage recipes; this is not a claim of model/runtime support.

    In particular, CSF checkpoints need their own loader and consumer. Expanding
    CSF and then storing five-bit offsets can increase resident scale memory.
    """
    q = config.get("quantization_config", {})
    if not isinstance(q, dict):
        raise TypeError("quantization_config must be an object")
    recipes = q.get("quantized_layers", {"*": q})
    if not isinstance(recipes, dict):
        raise TypeError("quantized_layers must be an object")
    result = {"plain_w4a16_group16": [], "already_encoded": [],
              "activation_quantized_or_backend_dependent": [], "other": []}
    for name, recipe in recipes.items():
        if not isinstance(recipe, dict):
            raise TypeError(f"Invalid quantization recipe: {name}")
        algorithm = recipe.get("quant_algo", recipe.get("quant_method"))
        if recipe.get("weight_scale_encoding"):
            category = "already_encoded"
        elif algorithm == "W4A16_NVFP4" and recipe.get("group_size", 16) == 16:
            category = "plain_w4a16_group16"
        elif algorithm == "NVFP4":
            category = "activation_quantized_or_backend_dependent"
        else:
            category = "other"
        result[category].append(name)
    return result


def register() -> None:
    selected = settings(os.environ)
    if selected is None:
        return
    from .compatibility import check_upstream
    check_upstream("nvfp4")
    from .nvfp4_backend import install
    install(selected)
