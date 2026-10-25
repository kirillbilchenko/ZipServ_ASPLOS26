"""CPU reference formats for screening exact low-precision weight compression.

Copyright 2026 Kyrylo Bilchenko. Licensed under Apache-2.0.
The rANS equations/implementation are adapted from MxWave's
screen_mxfp4_entropy.py; see compression-provenance.json. Modified to support
general alphabets, canonical Huffman, palettes, vector dictionaries and a
fully accounted, independently decoded block container. These are experiments,
not a production checkpoint format or a GPU implementation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import heapq
import math
import struct

KINDS = ("fp4", "fp8-e4m3", "fp8-e5m2", "scale-u8", "bytes")
CODECS = ("palette2", "palette3", "palette4", "huffman", "rans", "vector16", "vector32",
          "vector8", "vector8-direct", "vector16-direct", "vector4", "vector4-direct",
          "vector4-warp", "vector8-warp")
HEADER = struct.Struct("<4sBBBBIIQ")
TOTAL = 4096
LOWER = 1 << 23


def align(size: int, boundary: int) -> int:
    """Round a nonnegative byte count up to an alignment boundary."""
    return (size + boundary - 1) // boundary * boundary


def pack_bits(values: bytes, width: int) -> bytes:
    """Pack unsigned symbols in little-endian bit order, zero padding the tail."""
    if not 1 <= width <= 8:
        raise ValueError("Invalid symbol width")
    output = bytearray()
    buffer = occupied = 0
    for value in values:
        if not 0 <= value < 1 << width:
            raise ValueError("Symbol exceeds bit width")
        buffer |= value << occupied
        occupied += width
        while occupied >= 8:
            output.append(buffer & 255)
            buffer >>= 8
            occupied -= 8
    if occupied:
        output.append(buffer)
    return bytes(output)


def unpack_bits(payload: bytes, count: int, width: int) -> bytes:
    """Unpack symbols, rejecting incorrect lengths and nonzero tail padding."""
    if count < 0 or not 1 <= width <= 8 or len(payload) != (count * width + 7) // 8:
        raise ValueError("Invalid packed symbol length")
    output = bytearray()
    buffer = occupied = position = 0
    for _ in range(count):
        while occupied < width:
            buffer |= payload[position] << occupied
            occupied += 8
            position += 1
        output.append(buffer & ((1 << width) - 1))
        buffer >>= width
        occupied -= width
    if buffer:
        raise ValueError("Nonzero packed tail")
    return bytes(output)


def split_fields(raw: bytes, kind: str) -> tuple[bytes, bytes, int, int]:
    """Separate exponent symbols and untouched sign/mantissa bits without float casts."""
    if kind == "fp4":
        return bytes(v for b in raw for v in (b & 15, b >> 4)), b"", 4, 0
    if kind in ("scale-u8", "bytes"):
        return raw, b"", 8, 0
    mantissa = {"fp8-e4m3": 3, "fp8-e5m2": 2}[kind]
    exponent = 7 - mantissa
    symbols = bytes((b >> mantissa) & ((1 << exponent) - 1) for b in raw)
    residual = bytes((b & ((1 << mantissa) - 1)) | ((b >> 7) << mantissa) for b in raw)
    return symbols, pack_bits(residual, mantissa + 1), exponent, mantissa + 1


def join_fields(symbols: bytes, residual: bytes, kind: str) -> bytes:
    """Recover all original bits, including signed zeros and NaN encodings."""
    if kind == "fp4":
        if len(symbols) % 2 or any(v > 15 for v in symbols):
            raise ValueError("Invalid FP4 symbols")
        return bytes(symbols[i] | (symbols[i + 1] << 4) for i in range(0, len(symbols), 2))
    if kind in ("scale-u8", "bytes"):
        return symbols
    mantissa = {"fp8-e4m3": 3, "fp8-e5m2": 2}[kind]
    low = unpack_bits(residual, len(symbols), mantissa + 1)
    if any(s >= 1 << (7 - mantissa) for s in symbols):
        raise ValueError("Invalid FP8 exponent")
    return bytes((s << mantissa) | (v & ((1 << mantissa) - 1)) | ((v >> mantissa) << 7)
                 for s, v in zip(symbols, low))


def entropy(values: bytes) -> float:
    """Return zero-order bits/symbol, not a bound on correlation-aware codecs."""
    return -sum(n / len(values) * math.log2(n / len(values))
                for n in Counter(values).values()) if values else 0.0


def frequencies(counts: list[int]) -> list[int]:
    """Normalize a nonempty byte alphabet to a 12-bit rANS probability table."""
    if not any(counts) or any(c < 0 for c in counts) or len(counts) > 256:
        raise ValueError("Require a nonempty nonnegative byte alphabet")
    size = sum(counts)
    result = [max(1, c * TOTAL // size) if c else 0 for c in counts]
    order = sorted(range(len(counts)), key=lambda i: (counts[i] * TOTAL % size, counts[i]),
                   reverse=True)
    remaining = TOTAL - sum(result)
    while remaining > 0:
        for i in order:
            if counts[i] and remaining:
                result[i] += 1
                remaining -= 1
    while remaining < 0:
        i = max(range(len(result)), key=result.__getitem__)
        result[i] -= 1
        remaining += 1
    return result


def rans_encode(symbols: bytes, freq: list[int]) -> bytes:
    """Encode one byte-renormalized rANS stream with an explicit four-byte state."""
    cumulative, cursor = [], 0
    for f in freq:
        cumulative.append(cursor)
        cursor += f
    if cursor != TOTAL:
        raise ValueError("Invalid rANS table")
    state, emitted = LOWER, bytearray()
    for symbol in reversed(symbols):
        f = freq[symbol]
        if not f:
            raise ValueError("Symbol has no probability")
        while state >= ((LOWER >> 12) << 8) * f:
            emitted.append(state & 255)
            state >>= 8
        state = ((state // f) << 12) + state % f + cumulative[symbol]
    return struct.pack("<I", state) + bytes(reversed(emitted))


def rans_decode(payload: bytes, size: int, freq: list[int]) -> bytes:
    """Invert a stream and validate its exact terminal state and byte length."""
    table, cumulative, cursor = [], [], 0
    for symbol, f in enumerate(freq):
        cumulative.append(cursor)
        table.extend([symbol] * f)
        cursor += f
    if len(payload) < 4 or cursor != TOTAL:
        raise ValueError("Invalid rANS stream/table")
    state, position = struct.unpack_from("<I", payload)[0], 4
    output = bytearray()
    for _ in range(size):
        slot = state & (TOTAL - 1)
        symbol = table[slot]
        output.append(symbol)
        state = freq[symbol] * (state >> 12) + slot - cumulative[symbol]
        while state < LOWER:
            if position == len(payload):
                raise ValueError("Truncated rANS stream")
            state = (state << 8) | payload[position]
            position += 1
    if state != LOWER or position != len(payload):
        raise ValueError("rANS terminal state or length mismatch")
    return bytes(output)


def huffman_lengths(counts: list[int]) -> bytes:
    """Build deterministic Huffman lengths; single-symbol streams use one bit."""
    heap = [(n, i, [i]) for i, n in enumerate(counts) if n]
    heapq.heapify(heap)
    lengths = [0] * len(counts)
    if len(heap) == 1:
        lengths[heap[0][1]] = 1
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        symbols = a[2] + b[2]
        for symbol in symbols:
            lengths[symbol] += 1
        heapq.heappush(heap, (a[0] + b[0], min(symbols), symbols))
    return bytes(lengths)


def canonical_codes(lengths: bytes) -> dict[int, tuple[int, int]]:
    """Construct canonical codes, rejecting empty or oversubscribed tables."""
    codes, value, previous = {}, 0, 0
    for length, symbol in sorted((n, i) for i, n in enumerate(lengths) if n):
        value <<= length - previous
        if value >= 1 << length:
            raise ValueError("Oversubscribed Huffman table")
        codes[symbol] = (value, length)
        value += 1
        previous = length
    if not codes:
        raise ValueError("Empty Huffman table")
    return codes


def huffman_encode(symbols: bytes, lengths: bytes) -> bytes:
    """Encode canonical Huffman codes in most-significant-bit order."""
    codes = canonical_codes(lengths)
    buffer = occupied = 0
    output = bytearray()
    for symbol in symbols:
        code, bits = codes[symbol]
        buffer = (buffer << bits) | code
        occupied += bits
        while occupied >= 8:
            occupied -= 8
            output.append((buffer >> occupied) & 255)
            buffer &= (1 << occupied) - 1
    if occupied:
        output.append(buffer << (8 - occupied))
    return bytes(output)


def huffman_decode(payload: bytes, count: int, lengths: bytes) -> bytes:
    """Decode known symbol counts and require a minimal, zero-padded bitstream."""
    inverse = {(bits, code): symbol for symbol, (code, bits) in canonical_codes(lengths).items()}
    output, position = bytearray(), 0
    for _ in range(count):
        code = 0
        for length in range(1, max(lengths) + 1):
            if position >= len(payload) * 8:
                raise ValueError("Truncated Huffman stream")
            code = (code << 1) | ((payload[position // 8] >> (7 - position % 8)) & 1)
            position += 1
            if (length, code) in inverse:
                output.append(inverse[length, code])
                break
        else:
            raise ValueError("Invalid Huffman code")
    remaining = len(payload) * 8 - position
    if remaining >= 8 or (remaining and payload[-1] & ((1 << remaining) - 1)):
        raise ValueError("Invalid Huffman tail")
    return bytes(output)


@dataclass(frozen=True)
class SymbolCodec:
    """A sample-wide codebook with independent block/stream encoding."""

    name: str
    table: bytes
    symbol_bits: int
    streams: int = 1

    def encode(self, symbols: bytes) -> bytes:
        """Encode symbols, retaining all palette exceptions exactly."""
        if self.name.startswith("palette"):
            bits = int(self.name[-1])
            escape = (1 << bits) - 1
            lookup = {symbol: i for i, symbol in reversed(list(enumerate(self.table)))}
            selectors = bytes(lookup.get(s, escape) for s in symbols)
            exceptions = bytes(s for s, tag in zip(symbols, selectors) if tag == escape)
            return pack_bits(selectors, bits) + pack_bits(exceptions, self.symbol_bits)
        parts = []
        for lane in range(self.streams):
            values = symbols[lane::self.streams]
            if self.name == "rans":
                freq = list(struct.unpack("<" + "H" * (len(self.table) // 2), self.table))
                parts.append(rans_encode(values, freq))
            else:
                parts.append(huffman_encode(values, self.table))
        prefix = struct.pack("<" + "H" * self.streams, *(len(p) for p in parts)) \
            if self.streams > 1 else b""
        return prefix + b"".join(parts)

    def decode(self, payload: bytes, count: int) -> bytes:
        """Decode independently of the input and validate stream boundaries."""
        if self.name.startswith("palette"):
            bits = int(self.name[-1])
            escape = (1 << bits) - 1
            prefix = (count * bits + 7) // 8
            selectors = unpack_bits(payload[:prefix], count, bits)
            exceptions = iter(unpack_bits(payload[prefix:], selectors.count(escape), self.symbol_bits))
            return bytes(next(exceptions) if tag == escape else self.table[tag] for tag in selectors)
        position = 2 * self.streams if self.streams > 1 else 0
        if len(payload) < position:
            raise ValueError("Truncated lane index")
        lengths = struct.unpack_from("<" + "H" * self.streams, payload) \
            if self.streams > 1 else [len(payload)]
        output = bytearray(count)
        for lane, length in enumerate(lengths):
            part = payload[position:position + length]
            size = len(range(lane, count, self.streams))
            if self.name == "rans":
                freq = list(struct.unpack("<" + "H" * (len(self.table) // 2), self.table))
                values = rans_decode(part, size, freq)
            else:
                values = huffman_decode(part, size, self.table)
            output[lane::self.streams] = values
            position += length
        if position != len(payload):
            raise ValueError("Invalid lane lengths")
        return bytes(output)


def vector_width(name: str) -> int:
    """Extract the vector size from a validated codec name."""
    return int(name.removesuffix("-direct").removesuffix("-warp")[6:])


def vector_encode(raw: bytes, table: bytes, width: int, direct: bool = False,
                  warp: bool = False) -> bytes:
    """Encode exact vectors; optional U16 records address exceptions without prefix scans."""
    if direct and warp:
        raise ValueError("Choose direct records or warp exception offsets")
    lookup = {table[i:i + width]: i // width for i in range(0, len(table), width)}
    count = len(raw) // width
    indexes, exceptions, prefixes = [], bytearray(), []
    for i in range(count):
        if warp and i % 32 == 0:
            prefixes.append(len(exceptions) // width)
        vector = raw[i * width:(i + 1) * width]
        index = lookup.get(vector, 255)
        if index == 255:
            indexes.append(0x8000 | (len(exceptions) // width) if direct else 255)
            exceptions.extend(vector)
        else:
            indexes.append(index)
    records = struct.pack("<" + "H" * count, *indexes) if direct else bytes(indexes)
    prefix = struct.pack("<" + "H" * len(prefixes), *prefixes) if warp else b""
    return prefix + records + exceptions + raw[count * width:]


def vector_decode(payload: bytes, size: int, table: bytes, width: int, direct: bool = False,
                  warp: bool = False) -> bytes:
    """Gather exact dictionary vectors and restore inline exceptions and tail bytes."""
    count, tail = divmod(size, width)
    if direct and warp:
        raise ValueError("Choose direct records or warp exception offsets")
    warp_bytes = 2 * ((count + 31) // 32) if warp else 0
    prefix = warp_bytes + count * (2 if direct else 1)
    if len(payload) < prefix:
        raise ValueError("Invalid vector stream length")
    indexes = struct.unpack_from("<" + "H" * count, payload) if direct else payload[warp_bytes:prefix]
    if warp:
        prefixes = struct.unpack_from("<" + "H" * (warp_bytes // 2), payload)
        seen = 0
        for group, offset in enumerate(prefixes):
            if offset != seen:
                raise ValueError("Invalid warp exception offset")
            seen += indexes[group * 32:(group + 1) * 32].count(255)
    escaped = sum(bool(index & 0x8000) for index in indexes) if direct else indexes.count(255)
    expected = prefix + escaped * width + tail
    if len(payload) != expected:
        raise ValueError("Invalid vector stream length")
    output, position = bytearray(), prefix
    for index in indexes:
        is_exception = bool(index & 0x8000) if direct else index == 255
        if is_exception:
            if direct and (index & 0x7fff) != (position - prefix) // width:
                raise ValueError("Invalid direct exception address")
            output.extend(payload[position:position + width])
            position += width
        elif index * width >= len(table):
            raise ValueError("Invalid dictionary index")
        else:
            output.extend(table[index * width:(index + 1) * width])
    return bytes(output) + payload[position:]


def encode_container(raw: bytes, kind: str, codec: str, tile_bytes: int = 2048,
                     streams: int = 1) -> tuple[bytes, dict]:
    """Build actual bytes including tables, indexes, raw fallbacks and alignment."""
    if (not raw or kind not in KINDS or codec not in CODECS or streams not in (1, 32)
            or not 64 <= tile_bytes <= 16384 or tile_bytes % 32):
        raise ValueError("Invalid screen format/tile/stream parameters")
    if streams != 1 and codec not in ("rans", "huffman"):
        raise ValueError("Independent streams apply only to entropy codecs")
    symbols, _, symbol_bits, _ = split_fields(raw, kind)
    counts = [] if codec.startswith("vector") else [symbols.count(i) for i in range(1 << symbol_bits)]
    coverage = None
    if codec.startswith("vector"):
        width = vector_width(codec)
        vectors = Counter(raw[i:i + width] for i in range(0, len(raw) - width + 1, width))
        common = sorted((v for v in vectors if vectors[v] > 1), key=lambda v: (-vectors[v], v))[:255]
        table = b"".join(common)
        coverage = sum(vectors[v] for v in common) / sum(vectors.values()) if vectors else 0.0
    elif codec == "rans":
        table = struct.pack("<" + "H" * len(counts), *frequencies(counts))
    elif codec == "huffman":
        table = huffman_lengths(counts)
    else:
        entries = (1 << int(codec[-1])) - 1
        common = sorted(range(len(counts)), key=lambda i: (-counts[i], i))[:entries]
        table = bytes(common + [0] * (entries - len(common)))
    coder = SymbolCodec(codec, table, symbol_bits, streams)
    payload, offsets, lengths, tags = bytearray(), [0], [], bytearray()
    for first in range(0, len(raw), tile_bytes):
        block = raw[first:first + tile_bytes]
        if codec.startswith("vector"):
            packed = vector_encode(block, table, vector_width(codec), codec.endswith("-direct"),
                                   codec.endswith("-warp"))
        else:
            values, residual, _, _ = split_fields(block, kind)
            packed = residual + coder.encode(values)
        use_encoded = align(len(packed), 16) < align(len(block), 16)
        retained = packed if use_encoded else block
        tags.append(int(use_encoded))
        lengths.append(len(retained))
        payload.extend(retained)
        payload.extend(bytes(align(len(payload), 16) - len(payload)))
        offsets.append(len(payload))
    header = HEADER.pack(b"ZSC1", KINDS.index(kind), CODECS.index(codec), streams, 0,
                         tile_bytes, len(table), len(raw))
    header = header.ljust(64, b"\0")
    table_region = table.ljust(align(len(table), 64), b"\0")
    metadata = (struct.pack("<" + "Q" * len(offsets), *offsets)
                + struct.pack("<" + "I" * len(lengths), *lengths) + tags)
    metadata = metadata.ljust(align(len(metadata), 64), b"\0")
    container = bytes(header + table_region + metadata + payload)
    return container, {
        "codec": codec, "streams": streams, "tile_bytes": tile_bytes,
        "header_bytes": len(header), "codebook_bytes": len(table_region),
        "index_and_tag_bytes": len(metadata), "aligned_payload_bytes": len(payload),
        "raw_fallback_tiles": tags.count(0), "tiles": len(tags),
        "container_bytes": len(container), "saving_pct": 100 * (1 - len(container) / len(raw)),
        "dictionary_vector_coverage": coverage,
        "direct_exception_addresses": codec.endswith("-direct") if codec.startswith("vector") else None,
        "warp_exception_offsets": codec.endswith("-warp") if codec.startswith("vector") else None,
        "container_sha256": hashlib.sha256(container).hexdigest(),
    }


def decode_container(container: bytes) -> bytes:
    """Reconstruct original bytes solely from the reference container."""
    if len(container) < 64:
        raise ValueError("Truncated container")
    magic, kind_id, codec_id, streams, reserved, tile, table_size, size = HEADER.unpack_from(container)
    if (magic != b"ZSC1" or kind_id >= len(KINDS) or codec_id >= len(CODECS) or reserved
            or not size or streams not in (1, 32) or not 64 <= tile <= 16384 or tile % 32):
        raise ValueError("Invalid container header")
    kind, codec = KINDS[kind_id], CODECS[codec_id]
    _, _, symbol_bits, residual_bits = split_fields(b"", kind)
    if codec.startswith("vector"):
        width = vector_width(codec)
        valid_table = table_size % width == 0 and table_size // width <= 255 and streams == 1
    elif codec.startswith("palette"):
        valid_table = table_size == (1 << int(codec[-1])) - 1 and streams == 1
    else:
        valid_table = table_size == (1 << symbol_bits) * (2 if codec == "rans" else 1)
    if not valid_table:
        raise ValueError("Invalid codebook length")
    count = (size + tile - 1) // tile
    meta_start = 64 + align(table_size, 64)
    data_start = meta_start + align((count + 1) * 8 + count * 5, 64)
    if data_start > len(container) or any(container[HEADER.size:64]):
        raise ValueError("Truncated metadata or nonzero header padding")
    table = container[64:64 + table_size]
    if any(container[64 + table_size:meta_start]):
        raise ValueError("Nonzero codebook padding")
    offsets = struct.unpack_from("<" + "Q" * (count + 1), container, meta_start)
    lengths_start = meta_start + 8 * (count + 1)
    lengths = struct.unpack_from("<" + "I" * count, container, lengths_start)
    tags_start = lengths_start + 4 * count
    tags = container[tags_start:tags_start + count]
    if (offsets[0] != 0 or offsets[-1] != len(container) - data_start
            or any(container[tags_start + count:data_start])):
        raise ValueError("Invalid container boundaries")
    coder, output = SymbolCodec(codec, table, symbol_bits, streams), bytearray()
    for i, (length, tag) in enumerate(zip(lengths, tags)):
        block_size = min(tile, size - i * tile)
        start, end = offsets[i], offsets[i + 1]
        if (tag not in (0, 1) or end - start != align(length, 16)
                or end > offsets[-1] or any(container[data_start + start + length:data_start + end])):
            raise ValueError("Invalid tile metadata/padding")
        block = container[data_start + start:data_start + start + length]
        if tag == 0:
            if length != block_size:
                raise ValueError("Invalid raw block length")
            output.extend(block)
        elif codec.startswith("vector"):
            output.extend(vector_decode(block, block_size, table, vector_width(codec),
                                        codec.endswith("-direct"), codec.endswith("-warp")))
        else:
            symbols = block_size * (2 if kind == "fp4" else 1)
            prefix = (block_size * residual_bits + 7) // 8
            values = coder.decode(block[prefix:], symbols)
            output.extend(join_fields(values, block[:prefix], kind))
    return bytes(output)
