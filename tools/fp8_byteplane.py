"""ZFB1: exact FP8 compression with seven byte-aligned words per eight words.

Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
Each 32-byte group preserves 28 bytes' low six bits and sign in place. Their
bit-6 positions hold the remaining four bytes' seven useful bits. The shared
ZFA1-style tile container supplies common bit 6 and sparse exact corrections.
This changes decoding work, not compression ratio or quantization precision.
"""
from __future__ import annotations

import numpy as np
from fp8_async import _encode, _decode


def _pack_tile(tile: np.ndarray) -> bytes:
    groups = tile.reshape(-1, 8, 4)
    packed = groups[:, :7, :].copy() & 0xbf
    tail = groups[:, 7, :]
    for word, bit in enumerate((0, 1, 2, 3, 4, 5, 7)):
        packed[:, word, :] |= ((tail >> bit) & 1) << 6
    return packed.tobytes()


def _unpack_tile(payload: bytes, common: int) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 7, 4)
    result = np.empty((len(packed), 8, 4), dtype=np.uint8)
    result[:, :7, :] = (packed & 0xbf) | (common << 6)
    result[:, 7, :] = common << 6
    for word, bit in enumerate((0, 1, 2, 3, 4, 5, 7)):
        result[:, 7, :] |= ((packed[:, word, :] >> 6) & 1) << bit
    return result.reshape(-1)


def encode(raw: bytes, *, force_raw: bool = False, alignment: int = 16) -> tuple[bytes, dict]:
    return _encode(raw, force_raw=force_raw, alignment=alignment, magic=b"ZFB1",
                   codec="fp8-async-byteplane", pack_tile=_pack_tile)


def decode(blob: bytes) -> bytes:
    return _decode(blob, magic=b"ZFB1", unpack_tile=_unpack_tile)
