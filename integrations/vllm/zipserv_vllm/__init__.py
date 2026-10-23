# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Optional vLLM plugin: importing the package does not load CUDA or vLLM."""

from __future__ import annotations


def register() -> None:
    """Register explicitly selected methods in each vLLM 0.32.0 worker."""
    from importlib.metadata import version

    from packaging.version import Version

    if Version(version("vllm")).release != (0, 32, 0):
        raise RuntimeError("zipserv-vllm currently targets vLLM 0.32.0")
    from . import hybrid, vllm_plugin  # noqa: F401
