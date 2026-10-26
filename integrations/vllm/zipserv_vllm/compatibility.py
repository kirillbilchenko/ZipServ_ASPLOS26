# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Reviewed vLLM API profiles, independent of installing the optional plugin.

Matching a release number is necessary, but not sufficient: internal loaders
and kernel interfaces must also match a reviewed source profile. Local version
suffixes are allowed; silently accepting an arbitrary fork is not.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from packaging.version import InvalidVersion, Version

SUPPORTED_RELEASES = ("0.31.0", "0.32.0")


def release_profile(version: str) -> str | None:
    """Allow stable reviewed releases and their local build tags."""
    try:
        public = Version(version).public
    except InvalidVersion:
        return None
    return public if public in SUPPORTED_RELEASES else None


def check_source(root: Path, version: str, *adapters: str) -> str:
    """Validate every file used by selected adapters before altering any weights."""
    profile = release_profile(version)
    if profile is None:
        raise RuntimeError(
            f"Unsupported vLLM {version}; reviewed releases: {', '.join(SUPPORTED_RELEASES)}")
    manifest = json.loads(Path(__file__).with_name("vllm-compatibility.json").read_text())
    files: set[str] = set()
    for adapter in adapters:
        if adapter not in manifest["adapters"]:
            raise ValueError(f"Unknown ZipServ compatibility adapter: {adapter}")
        files.update(manifest["adapters"][adapter])
    if not files:
        raise ValueError("Select at least one ZipServ compatibility adapter")
    for name in sorted(files):
        path = root / name
        expected = manifest["releases"][profile]["sha256"][name]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(
                f"Installed vLLM {version} API differs from reviewed {profile}: {name}. "
                "A custom fork needs its own reviewed profile; disabling this check is unsafe.")
    return profile


def check_upstream(*adapters: str) -> str:
    """Inspect installed package sources without importing vLLM or initializing CUDA."""
    version = importlib.metadata.version("vllm")
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate installed vLLM")
    return check_source(Path(spec.origin).parent, version, *adapters)
