# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Optional vLLM plugin: importing the package does not load CUDA or vLLM."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Register supported methods without breaking unrelated vLLM installations."""
    from importlib.metadata import PackageNotFoundError, version

    from .compatibility import SUPPORTED_RELEASES, check_upstream, release_profile

    try:
        installed = version("vllm")
    except PackageNotFoundError:
        logger.warning("Skipping ZipServ registration: vLLM is not installed.")
        return
    if release_profile(installed) is None:
        logger.warning(
            "Skipping ZipServ registration: vLLM %s is unsupported (reviewed: %s).",
            installed, ", ".join(SUPPORTED_RELEASES),
        )
        return
    try:
        check_upstream("bf16")
    except RuntimeError as error:
        logger.warning("Skipping ZipServ registration: %s", error)
        return
    from . import vllm_plugin  # noqa: F401

    try:
        check_upstream("hybrid")
    except RuntimeError as error:
        logger.warning("Skipping ZipServ MXFP4-head registration: %s", error)
        return
    from . import hybrid  # noqa: F401
