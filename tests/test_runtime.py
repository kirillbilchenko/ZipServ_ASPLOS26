# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the optional PyTorch integration, without vLLM or CUDA."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
from zipserv_vllm import runtime


@pytest.mark.parametrize(
    "weight",
    [
        torch.empty(64, 64, dtype=torch.float16),
        torch.empty(64, dtype=torch.bfloat16),
        torch.empty(63, 64, dtype=torch.bfloat16),
        torch.empty(64, 65, dtype=torch.bfloat16),
        torch.empty(0, 64, dtype=torch.bfloat16),
    ],
)
def test_invalid_weights_rejected_before_library_load(weight):
    with pytest.raises(ValueError):
        runtime.compress(weight)


def test_missing_library(monkeypatch):
    runtime._library.cache_clear()
    monkeypatch.delenv("ZIPSERV_LIBRARY", raising=False)
    with pytest.raises(RuntimeError, match="Build the ZipServ bridge"):
        runtime.compress(torch.empty(64, 64, dtype=torch.bfloat16))


def test_storage_accounts_for_all_buffers():
    packed = runtime.PackedWeight(
        tuple(torch.empty(i, dtype=torch.uint8) for i in range(7)),
        64,
        64,
        4096,
        0,
        120,
    )
    assert packed.storage_bytes == 21
    assert packed.to("cpu").storage_bytes == 21
    with pytest.raises(ValueError, match="CUDA buffers"):
        runtime.decompress(packed)


@pytest.mark.parametrize("shape", [(128,), (0, 128), (2, 3, 128), (129, 128)])
def test_fake_shape_contract(shape):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        packed = runtime.PackedWeight(
            tuple(torch.empty(0, dtype=torch.uint8, device="cuda") for _ in range(7)),
            256,
            128,
            4096,
            0,
            120,
        )
        output = runtime.linear(x, packed)
        assert output.shape == (*shape[:-1], 256)
        assert output.dtype == x.dtype
        assert output.device == x.device
