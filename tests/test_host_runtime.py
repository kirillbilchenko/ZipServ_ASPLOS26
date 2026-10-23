# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real CPU bridge and native allocation ownership without CUDA."""

from __future__ import annotations

import ctypes
import gc
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations/vllm"))
from zipserv_vllm import runtime


@pytest.fixture(scope="session")
def host_library_path(tmp_path_factory):
    directory = tmp_path_factory.mktemp("zipserv-host-bridge")
    path = directory / ("bridge.dylib" if sys.platform == "darwin" else "bridge.so")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "integrations/vllm/build_native.py"),
            "--cpu-only",
            "--output",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


@pytest.fixture
def host_library(host_library_path, monkeypatch):
    monkeypatch.setenv("ZIPSERV_LIBRARY", str(host_library_path))
    runtime._library.cache_clear()
    yield runtime._library()
    runtime._library.cache_clear()
    gc.collect()


def decode_words(packed):
    """Invert bitmap tile indices independently of the native encoding loops."""
    common, rare, b1, b2, b3, medium, global_offsets = packed.buffers
    common = common.tolist()
    rare = rare.view(torch.uint16).tolist()
    bitmaps = [b.view(torch.int64).tolist() for b in (b1, b2, b3)]
    medium = medium.view(torch.int32).tolist()
    offsets = global_offsets.view(torch.int32).tolist()
    result = [0] * (packed.m * packed.k)
    for tile in range((packed.m // 64) * (packed.k // 64)):
        common_pos, rare_pos = offsets[2 * tile : 2 * tile + 2]
        for small in range(64):
            if small % 16 == 0:
                index = tile * 8 + (small // 16) * 2
                assert medium[index] == common_pos - offsets[2 * tile]
                assert medium[index + 1] == rare_pos - offsets[2 * tile + 1]
            row = (tile // (packed.k // 64)) * 64 + (small // 16) * 16 + (small % 2) * 8
            col = (
                (tile % (packed.k // 64)) * 64
                + ((small // 4) % 4) * 16
                + ((small // 2) % 2) * 8
            )
            bitmap_index = tile * 64 + small
            for bit in range(64):
                code = sum(
                    ((bitmap[bitmap_index] >> bit) & 1) << i
                    for i, bitmap in enumerate(bitmaps)
                )
                if code:
                    sm = common[common_pos]
                    common_pos += 1
                    exponent = (packed.start_exp + code) & 255
                    word = ((sm & 128) << 8) | (exponent << 7) | (sm & 127)
                else:
                    word = rare[rare_pos]
                    rare_pos += 1
                result[(row + bit // 8) * packed.k + col + bit % 8] = word
        assert (common_pos + 15) // 16 * 16 == offsets[2 * tile + 2]
        assert (rare_pos + 7) // 8 * 8 == offsets[2 * tile + 3]
    return result


def track_free(lib, monkeypatch):
    calls = []
    original = lib.zipserv_free

    def free(descriptor):
        native = ctypes.cast(descriptor, ctypes.POINTER(runtime._NativePacked)).contents
        calls.append(tuple(native.buffers))
        original(descriptor)

    monkeypatch.setattr(lib, "zipserv_free", free)
    return calls


def test_cpu_build_manifest(host_library_path, host_library):
    manifest = json.loads(host_library_path.with_suffix(".json").read_text())
    assert manifest["abi_version"] == 2
    assert manifest["cuda"] is False
    assert manifest["arch"] is None
    assert "csrc/HostPacker.h" in manifest["sources_sha256"]
    with pytest.raises(RuntimeError, match="no CUDA support"):
        runtime._cuda_library()


@pytest.mark.parametrize(
    "mode", ["all_bits", "zero", "negative_zero", "nan", "normal", "transposed"]
)
def test_real_bridge_bitwise_round_trip(host_library, mode):
    if mode in ("all_bits", "transposed"):
        bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16).reshape(256, 256)
        weight = bits.view(torch.bfloat16)
        if mode == "transposed":
            weight = weight.T
    elif mode == "normal":
        generator = torch.Generator().manual_seed(42)
        weight = torch.randn(128, 192, dtype=torch.bfloat16, generator=generator)
    else:
        word = {"zero": 0, "negative_zero": 0x8000, "nan": 0xFFFF}[mode]
        weight = torch.full((64, 64), word, dtype=torch.uint16).view(torch.bfloat16)
    expected = weight.contiguous().view(torch.uint16).flatten().tolist()
    packed = runtime.compress(weight)
    weight.zero_()  # The packed allocation must be independent of its input.
    assert decode_words(packed) == expected
    assert all(buffer.is_contiguous() for buffer in packed.buffers)


def test_tensors_share_native_addresses(host_library, monkeypatch):
    native_pointers = []
    original = host_library.zipserv_compress

    def compress(source, m, k, descriptor):
        status = original(source, m, k, descriptor)
        native = ctypes.cast(descriptor, ctypes.POINTER(runtime._NativePacked)).contents
        native_pointers.extend(native.buffers)
        return status

    monkeypatch.setattr(host_library, "zipserv_compress", compress)
    packed = runtime.compress(torch.ones(64, 64, dtype=torch.bfloat16))
    for pointer, buffer in zip(native_pointers, packed.buffers, strict=True):
        if buffer.numel():
            assert buffer.data_ptr() == pointer


def test_last_storage_view_owns_all_native_allocations(host_library, monkeypatch):
    frees = track_free(host_library, monkeypatch)
    packed = runtime.compress(torch.ones(64, 64, dtype=torch.bfloat16))
    retained = packed.buffers[2][1:4].detach()
    expected = retained.clone()
    same_device = packed.to("cpu")
    del packed
    gc.collect()
    assert not frees
    del same_device
    runtime._library.cache_clear()
    gc.collect()
    assert not frees
    assert torch.equal(retained, expected)
    del retained
    gc.collect()
    assert len(frees) == 1
    assert all(frees[0])


def test_empty_stream_does_not_retain_native_memory(host_library, monkeypatch):
    frees = track_free(host_library, monkeypatch)
    packed = runtime.compress(torch.zeros(64, 64, dtype=torch.bfloat16))
    empty = packed.buffers[1]
    assert empty.numel() == 0
    del packed
    gc.collect()
    assert len(frees) == 1
    assert empty.numel() == 0


@pytest.mark.parametrize("failure_at", [1, 3])
def test_partial_tensor_wrapping_failure_releases_allocations(
    host_library, monkeypatch, failure_at
):
    frees = track_free(host_library, monkeypatch)
    original = torch.frombuffer
    calls = 0

    def frombuffer(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise RuntimeError("simulated tensor wrapping failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "frombuffer", frombuffer)
    with pytest.raises(RuntimeError, match="simulated tensor wrapping failure"):
        runtime.compress(torch.ones(64, 64, dtype=torch.bfloat16))
    gc.collect()
    assert len(frees) == 1


def test_owner_creation_failure_releases_allocations(host_library, monkeypatch):
    frees = track_free(host_library, monkeypatch)

    def fail(*args):
        raise MemoryError("simulated owner allocation failure")

    monkeypatch.setattr(runtime, "_NativeBufferOwner", fail)
    with pytest.raises(MemoryError, match="owner allocation"):
        runtime.compress(torch.ones(64, 64, dtype=torch.bfloat16))
    gc.collect()
    assert len(frees) == 1


def test_native_validation_returns_empty_descriptor(host_library):
    native = runtime._NativePacked()
    assert host_library.zipserv_compress(None, 64, 64, ctypes.byref(native)) != 0
    assert not any(native.buffers)
    assert not any(native.lengths)
    host_library.zipserv_free(ctypes.byref(native))
