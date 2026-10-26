# Copyright MxWave contributors.
# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Lossless five-bit byte offsets for post-Marlin NVFP4 scales.

These are S0E5M3 bytes in a float8 storage tensor, not ordinary E4M3 values to
requantize. Only integer operations are used. Wide ranges remain raw.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EncodedPart:
    low: NDArray[np.uint32]
    high: NDArray[np.uint16]
    palette: NDArray[np.uint8]
    raw: NDArray[np.uint8]
    size: int

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.low, self.high, self.palette, self.raw))


def encode(raw: NDArray[np.uint8]) -> EncodedPart:
    """Use exact offsets in a 32-byte range, or a private copy of raw storage."""
    if raw.dtype != np.uint8 or not raw.size or raw.size % 16:
        raise ValueError("Require nonempty uint8 bytes in complete 16-byte vectors")
    values = np.ascontiguousarray(raw).reshape(-1)
    lo, hi = int(values.min()), int(values.max())
    if hi - lo >= 32:
        return EncodedPart(np.empty((0, 2), np.uint32), np.empty(0, np.uint16),
                           np.empty(0, np.uint8), values.copy(), values.size)
    codes = (values.astype(np.uint32) - lo).reshape(-1, 16)
    low = np.bitwise_or.reduce(
        (codes.reshape(-1, 2, 8) & 15) << (4 * np.arange(8, dtype=np.uint32)), axis=-1)
    # CUDA fetches pairs of high-plane uint16 values in one uint32. Pad even
    # the final odd vector so a legal fetch never reads past the allocation.
    vectors = codes.shape[0]
    high = np.zeros((vectors + 1) // 2 * 2, dtype=np.uint16)
    high[:vectors] = np.bitwise_or.reduce(
        (codes >> 4) << np.arange(16, dtype=np.uint32), axis=-1).astype(np.uint16)
    palette = ((np.arange(32, dtype=np.uint16) + lo) & 255).astype(np.uint8)
    return EncodedPart(low, high, palette, np.empty(0, np.uint8), values.size)


def decode(part: EncodedPart) -> NDArray[np.uint8]:
    """Validate record bounds, then independently reconstruct the original bytes."""
    if type(part.size) is not int or part.size < 16 or part.size % 16:
        raise ValueError("Invalid scale record length")
    if any(not isinstance(a, np.ndarray) or a.dtype != dtype for a, dtype in (
        (part.low, np.uint32), (part.high, np.uint16),
        (part.palette, np.uint8), (part.raw, np.uint8),
    )):
        raise ValueError("Invalid scale record dtypes")
    if part.raw.size:
        if (part.raw.shape != (part.size,) or part.low.size
                or part.high.size or part.palette.size):
            raise ValueError("Inconsistent raw scale record")
        return part.raw.copy()
    vectors = part.size // 16
    if (part.low.shape != (vectors, 2) or part.high.shape != ((vectors + 1) // 2 * 2,)
            or part.palette.shape != (32,) or part.raw.shape != (0,)):
        raise ValueError("Inconsistent packed scale record")
    base = int(part.palette[0])
    expected = ((np.arange(32, dtype=np.uint16) + base) & 255).astype(np.uint8)
    if not np.array_equal(part.palette, expected):
        raise ValueError("Five-bit offset decoder requires an affine palette")
    codes = ((part.low[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(-1, 16)
    codes |= (((part.high[:vectors, None].astype(np.uint32)
                >> np.arange(16, dtype=np.uint32)) & 1) << 4)
    values = codes + base
    if np.any(values > 255):
        raise ValueError("Scale byte offset would carry into an adjacent byte")
    return values.astype(np.uint8).reshape(-1)
