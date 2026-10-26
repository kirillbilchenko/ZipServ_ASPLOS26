# Copyright MxWave contributors.
# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
# Adapted for ZipServ; see NOTICE and mxfp4-provenance.json.
"""Lossless, random-addressable U8 scale palettes; no model-specific values."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

U8 = NDArray[np.uint8]
U32 = NDArray[np.uint32]

@dataclass(frozen=True)
class SparseScales:
    """Two-bit selectors or a tagged direct index into an exact 16-byte exception vector."""

    count: int
    palette: tuple[int, int, int]
    selectors: U32
    exceptions: U8

    @property
    def descriptor(self) -> int:
        """Encode a source-selected palette plus verified affine/alignment fast-path flags."""
        first, second, third = self.palette
        affine = second == first + 1 and third == first + 2
        return (first | second << 8 | third << 16 | int(affine) << 24
                | int(affine and first % 4 == 0) << 25)

    @property
    def retained_bytes(self) -> int:
        """Include both GPU buffers and the 24-byte descriptor."""
        return self.selectors.nbytes + self.exceptions.nbytes + 24

    def allocated_bytes(self, alignment: int = 512) -> int:
        """Estimate per-buffer rounding, including an otherwise tiny descriptor."""
        if alignment < 1:
            raise ValueError("Require positive allocation alignment")
        return sum((size + alignment - 1) // alignment * alignment for size in
                   (self.selectors.nbytes, self.exceptions.nbytes, 24))


def pack_sparse(raw: U8) -> SparseScales:
    """Keep frequent values on a two-bit path and preserve every exceptional vector verbatim."""
    if (raw.dtype != np.uint8 or raw.ndim != 1 or not raw.flags.c_contiguous
            or not raw.size or raw.size % 16 or raw.size >= 2**32):
        raise ValueError("Require contiguous U8 whole 16-byte vectors within U32 bounds")
    # NumPy promotes bincount's input to platform integers. Bound that temporary
    # instead of making an eight-byte copy of every scale in a large layer.
    histogram = np.zeros(256, dtype=np.int64)
    for first in range(0, raw.size, 1024**2):
        histogram += np.bincount(raw[first:first + 1024**2], minlength=256)
    frequent = np.lexsort((np.arange(256), -histogram))[:3]
    palette = tuple(int(v) for v in sorted(frequent))
    lookup = np.full(256, 3, dtype=np.uint8)
    for i, value in enumerate(palette):
        lookup[value] = i
    order = np.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
    shifts = np.arange(16, dtype=np.uint32) * 2
    selectors = np.empty(raw.size // 16, dtype=np.uint32)
    escaped_values = []
    exceptions_seen = 0
    for first in range(0, raw.size, 1024**2):
        values = raw[first:first + 1024**2].reshape(-1, 16)
        codes = lookup[values]
        words = np.bitwise_or.reduce(codes[:, order].astype(np.uint32) << shifts, axis=1)
        escaped = np.any(codes == 3, axis=1)
        # Normal selectors never contain code 3. Reserve code 3 in the final
        # position as a whole-vector escape, with a direct 30-bit payload index.
        escaped_count = int(escaped.sum())
        if exceptions_seen + escaped_count >= 2**30:
            raise ValueError("Exception vector count exceeds its 30-bit tag")
        words[escaped] = 0xc0000000 | (np.arange(escaped_count, dtype=np.uint32) + exceptions_seen)
        selectors[first // 16:(first + values.size) // 16] = words
        escaped_values.append(values[escaped].copy())
        exceptions_seen += escaped_count
    exceptions = np.concatenate(escaped_values)
    return SparseScales(int(raw.size), palette, selectors, exceptions)


def unpack_sparse(packed: SparseScales) -> U8:
    """Independently restore all bytes, rejecting invalid, repeated or unreferenced escapes."""
    count = packed.count
    palette = packed.palette
    if (count < 16 or count % 16 or len(palette) != 3 or len(set(palette)) != 3
            or any(v < 0 or v > 255 for v in palette)
            or packed.selectors.dtype != np.uint32 or packed.selectors.shape != (count // 16,)
            or packed.exceptions.dtype != np.uint8 or packed.exceptions.ndim != 2
            or packed.exceptions.shape[1] != 16):
        raise ValueError("Invalid sparse scale buffers")
    output = np.empty(count, dtype=np.uint8)
    order = np.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
    values = np.array([*palette, 0], dtype=np.uint8)
    shifts = np.arange(16, dtype=np.uint32) * 2
    exceptions_seen = 0
    for first in range(0, count // 16, 65536):
        words = packed.selectors[first:first + 65536]
        codes = ((words[:, None] >> shifts) & 3).astype(np.uint8)[:, order]
        decoded = values[codes]
        escaped = words >> 30 == 3
        if np.any(codes[~escaped] == 3):
            raise ValueError("Noncanonical escape selector")
        indices = words[escaped] & 0x3fffffff
        expected = np.arange(len(indices), dtype=np.uint32) + exceptions_seen
        if (not np.array_equal(indices, expected)
                or (len(indices) and indices[-1] >= len(packed.exceptions))):
            raise ValueError("Invalid or repeated exception index")
        decoded[escaped] = packed.exceptions[indices]
        exceptions_seen += len(indices)
        output[first * 16:(first + len(words)) * 16] = decoded.reshape(-1)
    if exceptions_seen != len(packed.exceptions):
        raise ValueError("Unreferenced sparse exception vectors")
    return output
