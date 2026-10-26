"""Experimental exact seven-bit exponent-window blocks with whole-block fallback.

Copyright 2026 Kyrylo Bilchenko. SPDX-License-Identifier: Apache-2.0
Each 16/32-value block either fits one tensor-wide eight-exponent window or is
stored as raw FP8. No per-value exception stream or palette is required.
"""
from __future__ import annotations

import hashlib
import numpy as np
from fp8_palette import HEADER, align, pack, unpack


def encode(raw: bytes, group: int = 16, tile: int = 2048) -> tuple[bytes, dict]:
    """Choose a window by complete-block coverage and serialize all actual bytes."""
    if not raw or group not in (16, 32) or tile not in (1024, 2048) or len(raw) % tile:
        raise ValueError("Require complete 1024/2048-byte tiles and 16/32-value groups")
    values = np.frombuffer(raw, dtype=np.uint8)
    exponents = (values >> 3) & 15
    low = exponents.reshape(-1, group).min(1)
    high = exponents.reshape(-1, group).max(1)
    base = max(range(9), key=lambda b: int(((low >= b) & (high < b + 8)).sum()))
    good = ((low >= base) & (high < base + 8)).reshape(-1, tile // group)
    tiles = len(raw) // tile
    data_offset = 64 + align(tiles * 8, 64)
    output = bytearray(data_offset)
    output[:64] = HEADER.pack(b"ZFR1", base, group, tile, len(raw), data_offset, bytes(8))
    offsets, fallback, compressed = [], 0, 0
    for ti, first in enumerate(range(0, len(raw), tile)):
        mask_size = align(tile // group, 64) // 8
        mask = sum(1 << i for i, fits in enumerate(good[ti]) if not fits)
        payload = bytearray(mask.to_bytes(mask_size, "little"))
        block_values = values[first:first + tile]
        residuals = pack((block_values & 7) | ((block_values >> 7) << 3), 4)
        fields = pack((((block_values >> 3) & 15) - base) & 7, 3)
        for gi in range(tile // group):
            if good[ti, gi]:
                payload.extend(residuals[gi * group // 2:(gi + 1) * group // 2])
                payload.extend(fields[gi * group * 3 // 8:(gi + 1) * group * 3 // 8])
            else:
                payload.extend(raw[first + gi * group:first + (gi + 1) * group])
        use_raw = align(len(payload), 16) >= tile
        offsets.append(len(output) | int(use_raw))
        retained = raw[first:first + tile] if use_raw else payload
        output.extend(retained)
        output.extend(bytes(align(len(output), 16) - len(output)))
        fallback += int(use_raw)
        compressed += int(good[ti].sum()) if not use_raw else 0
    output[64:64 + tiles * 8] = np.asarray(offsets, dtype="<u8").tobytes()
    encoded = bytes(output)
    return encoded, {"codec": f"fp8-range-block{group}", "base_exponent": base,
        "group_bytes": group, "raw_bytes": len(raw), "container_bytes": len(encoded),
        "header_bytes": 64, "index_bytes": data_offset - 64,
        "payload_bytes": len(encoded) - data_offset, "raw_fallback_tiles": fallback,
        "compressed_blocks": compressed, "tiles": tiles, "tile_bytes": tile,
        "saving_pct": 100 * (1 - len(encoded) / len(raw)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "container_sha256": hashlib.sha256(encoded).hexdigest()}


def decode(encoded: bytes) -> bytes:
    """Independently reconstruct and validate every block before GPU consumption."""
    if len(encoded) < 64:
        raise ValueError("Truncated range header")
    magic, base, group, tile, size, data_offset, reserved = HEADER.unpack_from(encoded)
    if (magic != b"ZFR1" or base > 8 or group not in (16, 32) or tile not in (1024, 2048)
            or not size or size % tile or any(reserved) or any(encoded[32:64])):
        raise ValueError("Invalid range header")
    tiles = size // tile
    if data_offset != 64 + align(tiles * 8, 64) or data_offset >= len(encoded):
        raise ValueError("Invalid range index")
    if any(encoded[64 + tiles * 8:data_offset]):
        raise ValueError("Invalid index padding")
    entries = np.frombuffer(encoded, dtype="<u8", count=tiles, offset=64)
    offsets = [int(v) & ~1 for v in entries] + [len(encoded)]
    if offsets[0] != data_offset or any(a >= b or a % 16 or b % 16
                                      for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Invalid range tile boundaries")
    result = bytearray()
    for ti in range(tiles):
        payload = encoded[offsets[ti]:offsets[ti + 1]]
        if int(entries[ti]) & 1:
            if len(payload) != tile:
                raise ValueError("Invalid raw tile")
            result.extend(payload)
            continue
        groups = tile // group
        position = align(groups, 64) // 8
        if len(payload) < position:
            raise ValueError("Truncated range mask")
        mask = int.from_bytes(payload[:position], "little")
        if mask >> groups:
            raise ValueError("Nonzero unused range-mask bits")
        restored = np.empty((groups, group), dtype=np.uint8)
        compressed_groups, lows, exps = [], bytearray(), bytearray()
        for gi in range(groups):
            count = group if (mask >> gi) & 1 else group * 7 // 8
            block = payload[position:position + count]
            if len(block) != count:
                raise ValueError("Truncated range block")
            if (mask >> gi) & 1:
                restored[gi] = np.frombuffer(block, dtype=np.uint8)
            else:
                compressed_groups.append(gi)
                lows.extend(block[:group // 2])
                exps.extend(block[group // 2:])
            position += count
        if len(payload) != align(position, 16) or any(payload[position:]):
            raise ValueError("Invalid range payload padding")
        count = len(compressed_groups) * group
        low = unpack(bytes(lows), count, 4)
        exp = unpack(bytes(exps), count, 3) + base
        restored[compressed_groups] = ((low & 7) | ((low >> 3) << 7) | (exp << 3)).reshape(-1, group)
        result.extend(restored.tobytes())
    return bytes(result)
