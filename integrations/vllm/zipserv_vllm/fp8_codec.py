# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Packaged ZFB1 byteplane codec; same validated format as tools/fp8_byteplane.py."""
from __future__ import annotations

import hashlib
import struct
import numpy as np

HEADER = struct.Struct("<4sBBHQQ8s32x")


def align(value: int, boundary: int) -> int:
    return (value + boundary - 1) // boundary * boundary


def _encode(raw: bytes, *, force_raw: bool, alignment: int, magic: bytes,
            codec: str, pack_tile) -> tuple[bytes, dict]:
    """Shared tile container; the format-specific callback packs 1024 to 896 bytes."""
    if not raw or len(raw) % 1024:
        raise ValueError("Async formats require complete 1024-byte tiles")
    if alignment not in (16, 32):
        raise ValueError("ZFA1 tile alignment must be 16 or 32 bytes")
    values = np.frombuffer(raw, dtype=np.uint8)
    bit6 = ((values >> 6) & 1).reshape(-1, 16)
    common = max((0, 1), key=lambda bit: int(np.all(bit6 == bit, axis=1).sum()))
    flipped = bit6 ^ common
    masks16 = np.packbits(flipped, axis=1, bitorder="little").copy().view("<u2").reshape(-1)
    tile_count = len(raw) // 1024
    data_offset = 64 + align(tile_count * 8, 64)
    output = bytearray(data_offset)
    # Extension byte 24: 0 retains the original 16-byte layout; 1 pads to 32.
    # The descriptor still counts omitted 16-byte copies for either alignment.
    output[:64] = HEADER.pack(magic, common, 16, 1024, len(raw), data_offset,
                             bytes([int(alignment == 32)]) + bytes(7))
    entries, raw_tiles, exceptions = [], 0, 0
    for ti in range(tile_count):
        tile = values[ti * 1024:(ti + 1) * 1024]
        flips = masks16[ti * 64:(ti + 1) * 64]
        mask = sum(1 << i for i in np.flatnonzero(flips).tolist())
        payload = bytearray(mask.to_bytes(8, "little") + bytes(8))
        payload.extend(pack_tile(tile))
        payload.extend(flips[flips != 0].astype("<u2").tobytes())
        size = align(len(payload), alignment)
        use_raw = force_raw or size >= 1024
        if use_raw:
            entries.append(len(output) | 1)
            output.extend(tile.tobytes())
            raw_tiles += 1
        else:
            # Low nibble: bit0=raw; bits1..3=count of omitted 16-byte slots.
            entries.append(len(output) | ((64 - size // 16) << 1))
            output.extend(payload)
            output.extend(bytes(size - len(payload)))
            exceptions += int((flips != 0).sum())
    output[64:64 + tile_count * 8] = np.asarray(entries, dtype="<u8").tobytes()
    blob = bytes(output)
    return blob, {"codec": codec, "common_bit6": common,
        "raw_bytes": len(raw), "container_bytes": len(blob), "tile_bytes": 1024,
        "alignment_bytes": alignment,
        "tiles": tile_count, "raw_fallback_tiles": raw_tiles,
        "exception_groups": exceptions, "header_bytes": 64, "index_bytes": data_offset - 64,
        "payload_bytes": len(blob) - data_offset, "force_raw": force_raw,
        "saving_pct": 100 * (1 - len(blob) / len(raw)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "container_sha256": hashlib.sha256(blob).hexdigest()}


def _decode(blob: bytes, *, magic: bytes, unpack_tile) -> bytes:
    if len(blob) < 64:
        raise ValueError("Truncated ZFA1 header")
    actual_magic, common, group, tile, size, data_offset, reserved = HEADER.unpack_from(blob)
    if (actual_magic != magic or common not in (0, 1) or group != 16 or tile != 1024
            or not size or size % 1024 or reserved[0] not in (0, 1)
            or any(reserved[1:]) or any(blob[32:64])):
        raise ValueError("Invalid ZFA1 header")
    alignment = 32 if reserved[0] else 16
    tiles = size // 1024
    if data_offset != 64 + align(tiles * 8, 64) or data_offset >= len(blob):
        raise ValueError("Invalid ZFA1 index")
    if any(blob[64 + tiles * 8:data_offset]):
        raise ValueError("Nonzero ZFA1 index padding")
    entries = np.frombuffer(blob, dtype="<u8", offset=64, count=tiles)
    offsets = [int(value) & ~15 for value in entries] + [len(blob)]
    if offsets[0] != data_offset or any(a >= b or a % alignment or b % alignment
                                      for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Invalid ZFA1 tile boundaries")
    result = bytearray()
    for i, entry in enumerate(entries):
        entry = int(entry)
        payload = blob[offsets[i]:offsets[i + 1]]
        if entry & 1:
            if entry & 14 or len(payload) != 1024:
                raise ValueError("Invalid ZFA1 raw tile")
            result.extend(payload)
            continue
        omitted = (entry >> 1) & 7
        if not omitted or len(payload) != 1024 - omitted * 16 or any(payload[8:16]):
            raise ValueError("Invalid ZFA1 compressed tile")
        mask = int.from_bytes(payload[:8], "little")
        consumed = 912 + mask.bit_count() * 2
        if align(consumed, alignment) != len(payload) or any(payload[consumed:]):
            raise ValueError("Invalid ZFA1 exception length or padding")
        restored = unpack_tile(payload[16:912], common)
        cursor = 912
        for group_index in range(64):
            if (mask >> group_index) & 1:
                flip = int.from_bytes(payload[cursor:cursor + 2], "little")
                if not flip:
                    raise ValueError("Noncanonical empty exception group")
                bits = np.unpackbits(np.frombuffer(payload[cursor:cursor + 2], dtype=np.uint8),
                                     bitorder="little")
                restored[group_index * 16:(group_index + 1) * 16] ^= bits << 6
                cursor += 2
        result.extend(restored.tobytes())
    return bytes(result)


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
