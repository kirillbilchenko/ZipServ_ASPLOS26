# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""CSF checkpoint decoder fixtures, including malformed metadata boundaries."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from csf32_scales import decode


def hand():
    # The independent TensorFold decoder's published Hand fixture.
    return {"bases": bytes([10] * 32),
            "codes": bytes(r % 16 | (r + 1) % 16 << 4 for r in range(32)),
            "page_bases": bytes(4), "offsets": bytes([0, 0, 2, 0]),
            "exceptions": bytes([1, 200, 62, 3])}


def test_published_hand_fixture():
    wanted = np.asarray([[10 + (r + c) % 16 for c in range(2)] for r in range(32)], dtype=np.uint8)
    wanted[0, 1], wanted[31, 0] = 200, 3
    assert np.array_equal(decode(hand(), 32, 2), wanted)


@pytest.mark.parametrize("key,value", [
    ("bases", bytes([241] * 32)), ("codes", bytes(31)),
    ("page_bases", bytes([1, 0, 0, 0])), ("offsets", bytes([1, 0, 2, 0])),
    ("offsets", bytes([0, 0, 1, 0])), ("offsets", bytes([0, 0, 3, 0])),
    ("exceptions", bytes([1, 200, 1, 3])), ("exceptions", bytes([62, 3, 1, 200])),
    ("exceptions", bytes([1, 200, 64, 3])), ("exceptions", bytes(3)),
])
def test_corrupt_components_fail(key, value):
    parts = hand()
    parts[key] = value
    with pytest.raises(ValueError):
        decode(parts, 32, 2)


def test_tile_order_and_page_boundary():
    # 33 full tiles, each containing eight 32x2 blocks: the last tile opens
    # a second page. Codes deliberately differ by block, not just by row.
    rows, columns = 128, 132
    codes = bytearray()
    wanted = np.empty((rows, columns), np.uint8)
    for tile in range(33):
        for pair in range(2):
            for br in range(4):
                code = (tile * 8 + pair * 4 + br) % 16
                codes.extend(bytes([code * 17] * 32))
                wanted[br * 32:(br + 1) * 32, tile * 4 + pair * 2:tile * 4 + pair * 2 + 2] = code
    parts = {"bases": bytes(rows), "codes": bytes(codes), "page_bases": bytes(8),
             "offsets": bytes((264 + 2) * 2), "exceptions": b""}
    assert np.array_equal(decode(parts, rows, columns), wanted)
    parts["page_bases"] = bytes(4) + bytes([2, 0, 0, 0])
    with pytest.raises(ValueError):
        decode(parts, rows, columns)


def test_one_column_stream_and_invalid_geometry():
    parts = {"bases": bytes([240] * 32), "codes": bytes([0xff] * 16),
             "page_bases": bytes(4), "offsets": bytes(4), "exceptions": b""}
    assert np.array_equal(decode(parts, 32, 1), np.full((32, 1), 255, np.uint8))
    for rows, columns in ((0, 1), (48, 2), (32, 0)):
        with pytest.raises(ValueError):
            decode(parts, rows, columns)
