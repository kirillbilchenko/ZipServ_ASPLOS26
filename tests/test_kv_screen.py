# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Bit-pattern and valid-page coverage for the experimental KV screen."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from capture_kv_vllm import select_slots
from screen_kv_pages import HEADER, codebook, decode, encode, exponents
from kv_transfer_buffers import validate_header


@pytest.mark.parametrize("kind,bits,width,patterns", [
    ("bf16", 4, 2, 65536), ("fp8_e4m3", 3, 1, 256), ("fp8_e5m2", 4, 1, 256)])
def test_every_bit_pattern_and_sparse_escapes(kind, bits, width, patterns):
    raw = np.arange(patterns, dtype="<u2" if width == 2 else "u1").tobytes()
    table = codebook(np.bincount(exponents(raw, kind)), bits)
    packed, stats = encode(raw, kind, table, bits, raw_fallback=False)
    assert stats["escapes"] > 0
    assert decode(packed) == raw  # Includes signed zero, subnormals and NaN payloads.
    fallback, stats = encode(raw, kind, table, bits)
    assert stats["raw_fallback"]
    assert len(fallback) == HEADER.size + len(raw)
    assert decode(fallback) == raw


@pytest.mark.parametrize("kind,bits,width", [("bf16", 4, 2), ("fp8_e4m3", 3, 1), ("fp8_e5m2", 4, 1)])
def test_odd_lengths_and_chunk_boundaries(kind, bits, width):
    rng = np.random.default_rng(91)
    for size in (0, 1, 3, 7, 1023, 1025, 2051):
        raw = rng.integers(0, 256, size * width, dtype=np.uint8).tobytes()
        table = np.arange(1 << bits, dtype=np.uint8)
        encoded, _ = encode(raw, kind, table, bits, raw_fallback=False)
        assert decode(encoded) == raw
        with pytest.raises(ValueError):
            decode(encoded[:-1])


def test_sampling_excludes_padding_and_selects_complete_pages():
    slots = [-1] + list(range(32, 32 + 20)) + [-1]
    picked = select_slots(slots, block_size=4, max_full_pages=3)
    assert [slots[i] for i in picked] == list(range(32, 36)) + list(range(40, 44)) + list(range(48, 52))
    assert select_slots([-1, 4, 5, -1], 4, 3) == [1, 2]
    assert select_slots([-1, -1], 4, 3) == []
    with pytest.raises(ValueError, match="Repeated"):
        select_slots([4, 4], 4, 3)
    with pytest.raises(ValueError, match="positive"):
        select_slots([1], 4, 0)


def test_pipeline_header_checks_actual_frame_bounds_and_raw_fallback():
    assert validate_header([1024, 0, 2048, 0], 1024, 2048) == (1024, 0, 2048, 0)
    assert validate_header([1024, 2, 1566, 1], 1024, 2048) == (1024, 2, 1566, 1)
    for fields in ([1024, 2, 1565, 1], [1024, 1, 2048, 0], [1024, 100, 2252, 1],
                   [1023, 0, 2046, 0], [1024, 0, 2048, 2], [1024, -1, 0, 1],
                   [1024, 0, 2048, False], [1024, 0, 2048]):
        with pytest.raises(ValueError):
            validate_header(fields, 1024, 2048)
