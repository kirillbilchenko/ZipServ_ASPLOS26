# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Decode existing csf_32x2_v1 NVFP4 scale bytes, without requantization.

Format reference: TensorFold nvfp4_csf32.zig, commit
67cc8fbce879e2d0dd74b94ebb8848b134478ee4 (Apache-2.0).
https://github.com/ashhart/TensorFold/blob/67cc8fbce879e2d0dd74b94ebb8848b134478ee4/zig/src/core/nvfp4_csf32.zig

This is a checkpoint decoder, not an inference kernel. MXFP4's one-bit CSF
representation is a different contract and is deliberately not accepted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import numpy as np

PARTS = ("bases", "codes", "page_bases", "offsets", "exceptions")


@dataclass(frozen=True)
class Geometry:
    rows: int
    columns: int
    block_rows: np.ndarray
    block_columns: np.ndarray
    widths: np.ndarray
    pages: np.ndarray


@lru_cache(maxsize=8)
def geometry(rows: int, columns: int) -> Geometry:
    if (type(rows) is not int or type(columns) is not int or rows < 32
            or rows % 32 or columns < 1 or rows * columns >= 2**31):
        raise ValueError("CSF requires positive rows divisible by 32 and an int32-sized matrix")
    records = []
    tile = 0
    for row in range(0, rows // 32, 4):
        height = min(4, rows // 32 - row)
        for pair in range(0, (columns + 1) // 2, 2):
            width = min(2, (columns + 1) // 2 - pair)
            for local_pair in range(width):
                column = 2 * (pair + local_pair)
                for local_row in range(height):
                    records.append(((row + local_row) * 32, column,
                                    min(2, columns - column), tile // 32))
            tile += 1
    arrays = np.asarray(records, dtype=np.int64).T.copy()
    arrays.flags.writeable = False
    return Geometry(rows, columns, *arrays)


def decode(parts: Mapping[str, bytes], rows: int, columns: int) -> np.ndarray:
    """Return exact E4M3 storage bytes in row-major [rows, columns] order.

    All component sizes, base ranges, page origins, monotonic prefixes,
    exception bounds/order, and complete stream consumption are checked.
    """
    g = geometry(rows, columns)
    blocks, pages = g.pages.size, int(g.pages[-1]) + 1
    if set(parts) != set(PARTS):
        raise ValueError("Require exactly the five CSF scale components")
    expected = {"bases": rows, "codes": rows * columns // 2,
                "page_bases": 4 * pages, "offsets": 2 * (blocks + pages)}
    if any(len(parts[key]) != size for key, size in expected.items()) or len(parts["exceptions"]) % 2:
        raise ValueError("CSF component lengths do not match the matrix geometry")
    bases = np.frombuffer(parts["bases"], dtype=np.uint8)
    if np.any(bases > 240):
        raise ValueError("CSF base plus four-bit offset can exceed a byte")
    page_bases = np.frombuffer(parts["page_bases"], dtype="<u4").astype(np.int64)
    offsets = np.frombuffer(parts["offsets"], dtype="<u2").astype(np.int64)
    exceptions = np.frombuffer(parts["exceptions"], dtype="<u2")
    entries = np.arange(blocks) + g.pages
    lo, hi = offsets[entries], offsets[entries + 1]
    origins = page_bases[g.pages] // 2
    starts = np.r_[True, g.pages[1:] != g.pages[:-1]]
    new_pages = np.flatnonzero(starts)[1:]
    if (np.any(page_bases % 2) or origins[0] != 0 or np.any(lo[starts])
            or np.any(hi < lo) or np.any(origins + hi > exceptions.size)
            or np.any(origins[new_pages] != (origins + hi)[new_pages - 1])
            or origins[-1] + hi[-1] != exceptions.size):
        raise ValueError("Invalid CSF page origins or exception prefixes")

    # Decode the fixed stream using its tile ordering, before exact exceptions.
    codes = np.frombuffer(parts["codes"], dtype=np.uint8)
    values = np.empty(rows * columns, dtype=np.uint8)
    values[0::2], values[1::2] = codes & 15, codes >> 4
    if rows % 128 == 0 and columns % 4 == 0:
        result = values.reshape(rows // 128, columns // 4, 2, 4, 32, 2)
        result = result.transpose(0, 3, 4, 1, 2, 5).reshape(rows, columns).copy()
        result += bases[:, None]
    else:
        result = np.empty((rows, columns), dtype=np.uint8)
        position = 0
        for r, c, width in zip(g.block_rows, g.block_columns, g.widths, strict=True):
            count = 32 * width
            result[r:r + 32, c:c + width] = (
                values[position:position + count].reshape(32, width) + bases[r:r + 32, None])
            position += count
        if position != values.size:
            raise ValueError("CSF fixed stream was not completely consumed")

    if exceptions.size:
        owner = np.repeat(np.arange(blocks), hi - lo)
        position = exceptions & 255
        if (owner.size != exceptions.size or np.any(position // 2 >= 32)
                or np.any(position % 2 >= g.widths[owner])
                or np.any((owner[1:] == owner[:-1]) & (position[1:] <= position[:-1]))):
            raise ValueError("Invalid CSF exception position or record order")
        r = g.block_rows[owner] + position // 2
        c = g.block_columns[owner] + position % 2
        result[r, c] = (exceptions >> 8).astype(np.uint8)
    return result
