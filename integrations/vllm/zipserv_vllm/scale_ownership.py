# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Keep compressed pointer descriptors out of stock scale consumers."""
from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Any

import torch


def storage_identity(tensor: torch.Tensor) -> tuple[str, int]:
    """Recognize detached, sliced and dtype-reinterpreted views without reading GPU data."""
    return str(tensor.device), tensor.untyped_storage()._cdata


def scale_metadata(
    candidate: Any,
    registered: Mapping[int, dict[str, Any]],
    descriptor_storages: AbstractSet[tuple[str, int]],
) -> dict[str, Any] | None:
    """Return a known descriptor's owner, allow raw scales, reject unknown descriptors."""
    metadata = registered.get(id(candidate))
    if metadata is not None:
        if candidate is not metadata["replacement"]:
            raise RuntimeError("FP4 compressed descriptor identity does not match its owner")
        return metadata
    if isinstance(candidate, torch.Tensor) and (
        candidate.dtype == torch.int64 or storage_identity(candidate) in descriptor_storages
    ):
        # The compressed format is an int64 pointer descriptor; stock scale
        # tensors are floating-point. Never interpret a copied descriptor or
        # an alias of its storage as a raw scale tensor.
        raise RuntimeError(
            "Unrecognised FP4 compressed scale descriptor; stock GEMM fallback is unsafe"
        )
    return None
