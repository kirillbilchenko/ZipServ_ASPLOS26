"""Experimental lossless E4M3 byte format with independently addressed GPU tiles.

Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
This is a component experiment, not a checkpoint format or serving plugin.
All sign/mantissa bits and exceptional exponents are retained, including NaNs.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

HEADER = struct.Struct("<4sBBHQQ8s32x")


def align(value: int, boundary: int) -> int:
    """Round a byte count up to a power-of-two boundary."""
    return (value + boundary - 1) // boundary * boundary


def pack(values: np.ndarray, bits: int) -> bytes:
    """Pack little-endian fields, with zero padding in a final partial group."""
    width = 8 if bits == 3 else 8 // bits
    padded = np.pad(values.astype(np.uint32), (0, (-len(values)) % width))
    words = (padded.reshape(-1, width) << (np.arange(width) * bits)).sum(1)
    count = 3 if bits == 3 else 1
    return np.stack([(words >> (8 * i)).astype(np.uint8) for i in range(count)],
                    axis=1).tobytes()[:(len(values) * bits + 7) // 8]


def unpack(data: bytes, count: int, bits: int) -> np.ndarray:
    """Read exactly count fields and reject nonzero unused bits."""
    if len(data) != (count * bits + 7) // 8:
        raise ValueError("Invalid packed field length")
    if count * bits % 8 and data[-1] >> (count * bits % 8):
        raise ValueError("Nonzero field padding")
    source = np.frombuffer(data + b"\0\0", dtype=np.uint8).astype(np.uint32)
    positions = np.arange(count, dtype=np.int64) * bits
    words = source[positions // 8] | (source[positions // 8 + 1] << 8)
    return ((words >> (positions % 8)) & ((1 << bits) - 1)).astype(np.uint8)


def encode(raw: bytes, bits: int = 3, tile: int = 2048) -> tuple[bytes, dict]:
    """Encode complete tiles with one prefix per 128 bytes and raw-tile fallback."""
    if bits not in (2, 3) or tile not in (1024, 2048) or not raw or len(raw) % tile:
        raise ValueError("Require 2/3 palette bits and complete 1024/2048-byte tiles")
    values = np.frombuffer(raw, dtype=np.uint8)
    exponents = (values >> 3) & 15
    counts = np.bincount(exponents, minlength=16)
    escape = (1 << bits) - 1
    palette = sorted(range(16), key=lambda i: (-int(counts[i]), i))[:escape]
    lookup = np.full(16, escape, dtype=np.uint8)
    lookup[palette] = np.arange(escape, dtype=np.uint8)
    tiles = len(raw) // tile
    data_offset = 64 + align(tiles * 8, 64)
    output, offsets = bytearray(data_offset), []
    output[:64] = HEADER.pack(b"ZFP1", bits, 0, tile, len(raw), data_offset,
                              bytes(palette).ljust(8, b"\0"))
    exceptions = fallbacks = 0
    for first in range(0, len(raw), tile):
        block = values[first:first + tile]
        exp = exponents[first:first + tile]
        selectors = lookup[exp]
        escaped = selectors == escape
        warp_counts = escaped.reshape(-1, 128).sum(1)
        prefixes = np.r_[0, np.cumsum(warp_counts)[:-1]].astype("<u2").tobytes()
        residual = (block & 7) | ((block >> 7) << 3)
        candidate = prefixes + pack(residual, 4) + pack(selectors, bits) + pack(exp[escaped], 4)
        fallback = align(len(candidate), 16) >= tile
        offsets.append(len(output) | int(fallback))
        payload = block.tobytes() if fallback else candidate
        output.extend(payload)
        output.extend(bytes(align(len(output), 16) - len(output)))
        exceptions += int(escaped.sum()) if not fallback else 0
        fallbacks += int(fallback)
    output[64:64 + tiles * 8] = np.asarray(offsets, dtype="<u8").tobytes()
    encoded = bytes(output)
    return encoded, {
        "codec": f"fp8-palette{bits}-warp128", "raw_bytes": len(raw),
        "container_bytes": len(encoded), "header_bytes": 64,
        "index_bytes": data_offset - 64, "payload_bytes": len(encoded) - data_offset,
        "exception_symbols": exceptions, "raw_fallback_tiles": fallbacks,
        "tiles": tiles, "tile_bytes": tile, "palette": palette,
        "saving_pct": 100 * (1 - len(encoded) / len(raw)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "container_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def decode(encoded: bytes) -> bytes:
    """Validate the entire container before a GPU is permitted to consume it."""
    if len(encoded) < 64:
        raise ValueError("Truncated FP8 header")
    magic, bits, reserved, tile, size, data_offset, table = HEADER.unpack_from(encoded)
    if (magic != b"ZFP1" or bits not in (2, 3) or reserved or tile not in (1024, 2048)
            or not size or size % tile or encoded[32:64] != bytes(32)):
        raise ValueError("Invalid FP8 header")
    tiles = size // tile
    if data_offset != 64 + align(tiles * 8, 64) or data_offset >= len(encoded):
        raise ValueError("Invalid FP8 index")
    escape = (1 << bits) - 1
    palette = np.frombuffer(table[:escape], dtype=np.uint8)
    if len(set(palette)) != escape or max(palette) > 15 or any(table[escape:]):
        raise ValueError("Invalid FP8 palette")
    if any(encoded[64 + tiles * 8:data_offset]):
        raise ValueError("Nonzero FP8 index padding")
    entries = np.frombuffer(encoded, dtype="<u8", count=tiles, offset=64)
    offsets = [int(e) & ~1 for e in entries] + [len(encoded)]
    if offsets[0] != data_offset or any(a >= b or a % 16 or b % 16
                                      for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Invalid FP8 tile boundaries")
    result = bytearray()
    for i in range(tiles):
        payload = encoded[offsets[i]:offsets[i + 1]]
        if int(entries[i]) & 1:
            if len(payload) != tile:
                raise ValueError("Invalid FP8 raw tile")
            result.extend(payload)
            continue
        prefix_size = tile // 128 * 2
        fields_end = prefix_size + tile // 2 + tile * bits // 8
        if len(payload) < fields_end:
            raise ValueError("Truncated FP8 tile")
        residual = unpack(payload[prefix_size:prefix_size + tile // 2], tile, 4)
        selectors = unpack(payload[prefix_size + tile // 2:fields_end], tile, bits)
        escaped = selectors == escape
        count = int(escaped.sum())
        expected = np.r_[0, np.cumsum(escaped.reshape(-1, 128).sum(1))[:-1]]
        prefixes = np.frombuffer(payload, dtype="<u2", count=tile // 128)
        if not np.array_equal(prefixes, expected):
            raise ValueError("Invalid FP8 exception prefixes")
        end = fields_end + (count + 1) // 2
        if len(payload) != align(end, 16) or any(payload[end:]):
            raise ValueError("Invalid FP8 payload length/padding")
        exponents = np.zeros(tile, dtype=np.uint8)
        exponents[~escaped] = palette[selectors[~escaped]]
        exponents[escaped] = unpack(payload[fields_end:end], count, 4)
        result.extend(((residual & 7) | ((residual >> 3) << 7) | (exponents << 3)).tobytes())
    return bytes(result)
