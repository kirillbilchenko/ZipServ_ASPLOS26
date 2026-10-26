# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Screen captured KV pages with an independent SplitZip-scheme CPU reference.

This is NOT the authors' wire format or a performance implementation. It uses
the same fixed exponent codes, exact sign/mantissa, and seven-byte escape
records as their public BF16 codec. FP8 borrows Appendix C's fixed-code idea
but retains the public BF16 artifact's seven-byte escapes: the authors have
not released their FP8 encoder, so this is NOT a reproduction of Table 9.
Each page additionally pays a 32-byte header and its complete codebook. The
raw fallback pays its header too. The actual upstream GPU codec is qualified
separately. Compression ratios here do not establish serving-memory savings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

FORMATS = {"bf16": (8, 7, 2), "fp8_e4m3": (4, 3, 1), "fp8_e5m2": (5, 2, 1)}
HEADER = struct.Struct("<4sBBBBQIIQ")
ESCAPE = np.dtype([("chunk", "<u4"), ("offset", "<u2"), ("exponent", "u1")])


def words(raw: bytes, kind: str) -> np.ndarray:
    width = FORMATS[kind][2]
    if len(raw) % width:
        raise ValueError("Partial floating-point word")
    return np.frombuffer(raw, dtype="<u2" if width == 2 else "u1").astype(np.uint16)


def exponents(raw: bytes, kind: str) -> np.ndarray:
    exp_bits, mant_bits, _ = FORMATS[kind]
    return (words(raw, kind) >> mant_bits) & ((1 << exp_bits) - 1)


def pack_fields(values: np.ndarray, width: int) -> bytes:
    bits = ((values.astype(np.uint16)[:, None] >> np.arange(width)) & 1)
    return np.packbits(bits.astype(np.uint8).reshape(-1), bitorder="little").tobytes()


def unpack_fields(raw: bytes, count: int, width: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:count * width]
    return (bits.reshape(count, width).astype(np.uint16) << np.arange(width)).sum(1).astype(np.uint16)


def codebook(histogram: np.ndarray, bits: int) -> np.ndarray:
    # Stable tie-breaking keeps every report reproducible.
    return np.argsort(-histogram, kind="stable")[:1 << bits].astype(np.uint8)


def encode(raw: bytes, kind: str, table: np.ndarray, bits: int,
           *, raw_fallback: bool = True, chunk_size: int = 1024) -> tuple[bytes, dict]:
    if kind not in FORMATS or bits not in (3, 4):
        raise ValueError("Unsupported format/code width")
    exp_bits, mant_bits, width = FORMATS[kind]
    if (len(table) != 1 << bits or len(set(map(int, table))) != len(table)
            or max(map(int, table)) >= 1 << exp_bits or not 1 <= chunk_size <= 65536):
        raise ValueError("Invalid codebook or chunk size")
    source = words(raw, kind)
    exp = (source >> mant_bits) & ((1 << exp_bits) - 1)
    lut = np.full(1 << exp_bits, -1, dtype=np.int16)
    lut[table] = np.arange(len(table))
    codes = lut[exp]
    positions = np.flatnonzero(codes < 0)
    if len(source) > 2**32 or len(positions) > 2**32 - 1:
        raise ValueError("Reference is limited to 2^32 words per frame")
    escapes = np.empty(len(positions), dtype=ESCAPE)
    escapes["chunk"] = positions // chunk_size
    escapes["offset"] = positions % chunk_size
    escapes["exponent"] = exp[positions]
    sm = ((source >> exp_bits) & (1 << mant_bits)) | (source & ((1 << mant_bits) - 1))
    dense = pack_fields(np.maximum(codes, 0), bits) + pack_fields(sm, mant_bits + 1)
    body = table.tobytes() + dense + escapes.tobytes()
    coded_size = HEADER.size + len(body)
    use_raw = raw_fallback and coded_size >= HEADER.size + len(raw)
    header = HEADER.pack(b"ZKVS", 1, list(FORMATS).index(kind), bits, int(use_raw),
                         len(source), 0 if use_raw else len(positions), chunk_size, 0)
    encoded = header + (raw if use_raw else body)
    return encoded, {"raw_bytes": len(raw), "coded_bytes": coded_size,
                     "payload_bytes": len(dense) + escapes.nbytes,
                     "stored_bytes": len(encoded), "escapes": len(positions),
                     "words": len(source), "raw_fallback": use_raw}


def decode(encoded: bytes) -> bytes:
    if len(encoded) < HEADER.size:
        raise ValueError("Truncated frame")
    magic, version, kind_id, bits, raw_flag, count, n_esc, chunk_size, reserved = HEADER.unpack_from(encoded)
    if (magic != b"ZKVS" or version != 1 or kind_id >= len(FORMATS)
            or bits not in (3, 4) or raw_flag not in (0, 1) or reserved
            or count > 2**32 or not 1 <= chunk_size <= 65536):
        raise ValueError("Invalid frame header")
    kind = list(FORMATS)[kind_id]
    exp_bits, mant_bits, width = FORMATS[kind]
    data = encoded[HEADER.size:]
    if raw_flag:
        if n_esc or len(data) != count * width:
            raise ValueError("Invalid raw frame")
        return data
    table_bytes = 1 << bits
    code_bytes, sm_bytes = (count * bits + 7) // 8, (count * (mant_bits + 1) + 7) // 8
    if len(data) != table_bytes + code_bytes + sm_bytes + n_esc * ESCAPE.itemsize:
        raise ValueError("Inconsistent frame lengths")
    table = np.frombuffer(data[:table_bytes], dtype=np.uint8)
    if len(set(map(int, table))) != table_bytes or np.any(table >= (1 << exp_bits)):
        raise ValueError("Invalid exponent table")
    pos = table_bytes
    codes = unpack_fields(data[pos:pos + code_bytes], count, bits)
    pos += code_bytes
    sm = unpack_fields(data[pos:pos + sm_bytes], count, mant_bits + 1)
    pos += sm_bytes
    exp = table[codes].astype(np.uint16)
    escapes = np.frombuffer(data[pos:], dtype=ESCAPE)
    indices = escapes["chunk"].astype(np.uint64) * chunk_size + escapes["offset"]
    if (np.any(indices >= count) or np.any(escapes["offset"] >= chunk_size)
            or np.any(escapes["exponent"] >= (1 << exp_bits))
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("Invalid escape indices or exponents")
    exp[indices.astype(np.int64)] = escapes["exponent"]
    restored = ((sm & (1 << mant_bits)) << exp_bits) | (exp << mant_bits) | (sm & ((1 << mant_bits) - 1))
    return restored.astype("<u2" if width == 2 else "u1").tobytes()


def entropy(hist: np.ndarray) -> float:
    total = hist.sum()
    p = hist[hist > 0] / total if total else np.array([])
    return float(-(p * np.log2(p)).sum())


def full_pages(arrays, record):
    slots = arrays["slots"]
    for block in np.unique(slots // record["block_size"]):
        mask = slots // record["block_size"] == block
        if mask.sum() == record["block_size"]:
            yield mask


def screen(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    kind = "bf16" if manifest["kv_dtype"] == "auto" else manifest["kv_dtype"]
    exp_bits, mant_bits, width = FORMATS[kind]
    bits = 3 if kind == "fp8_e4m3" else 4
    histograms = defaultdict(lambda: np.zeros(1 << exp_bits, dtype=np.int64))
    requests = [json.loads((root / f"{run['request']['id']}.json").read_text()) for run in manifest["runs"]]
    if {r["request"]["phase"] for r in requests} != {"calibration", "evaluation"}:
        raise ValueError("Need disjoint calibration and evaluation captures")
    calibration_hashes = {r["request"]["text_sha256"] for r in requests if r["request"]["phase"] == "calibration"}
    if any(r["request"]["text_sha256"] in calibration_hashes for r in requests if r["request"]["phase"] == "evaluation"):
        raise ValueError("Calibration/evaluation input overlap")
    for request in requests:
        if request["request"]["phase"] != "calibration":
            continue
        for record in request["records"]:
            with np.load(root / record["file"], allow_pickle=False) as arrays:
                for mask in full_pages(arrays, record):
                    for component in ("key", "value"):
                        raw = arrays[component][mask].tobytes()
                        h = np.bincount(exponents(raw, kind), minlength=1 << exp_bits)
                        histograms["global"] += h
                        histograms[record["layer"] + ":" + component] += h
    tables = {key: codebook(hist, bits) for key, hist in histograms.items()}
    rows = []
    for request in requests:
        if request["request"]["phase"] != "evaluation":
            continue
        for record in request["records"]:
            path = root / record["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("Capture checksum mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                masks = list(full_pages(arrays, record))
                for component in ("key", "value"):
                    key = record["layer"] + ":" + component
                    totals = {policy: defaultdict(int) for policy in ("global", "layer_kv")}
                    head_hists = np.zeros((record["num_kv_heads"], 1 << exp_bits), dtype=np.int64)
                    for mask in masks:
                        page = arrays[component][mask]
                        raw = page.tobytes()
                        for head in range(record["num_kv_heads"]):
                            head_hists[head] += np.bincount(exponents(page[:, head].tobytes(), kind), minlength=1 << exp_bits)
                        for policy, table in (("global", tables["global"]), ("layer_kv", tables[key])):
                            packed, stats = encode(raw, kind, table, bits)
                            if decode(packed) != raw:
                                raise AssertionError("KV bitwise round trip failed")
                            for metric, value in stats.items():
                                totals[policy][metric] += value
                    rows.append({"request": request["request"]["id"], "layer": record["layer"],
                                 "component": component, "full_pages": len(masks),
                                 "tail_tokens_excluded": len(arrays["slots"]) - len(masks) * record["block_size"],
                                 "exponent_entropy": entropy(head_hists.sum(0)),
                                 "per_head_exponent_entropy": [entropy(h) for h in head_hists],
                                 "policies": {p: dict(t) for p, t in totals.items()}})
    sums = {}
    for policy in ("global", "layer_kv"):
        total = {metric: sum(row["policies"][policy].get(metric, 0) for row in rows)
                 for metric in ("raw_bytes", "coded_bytes", "payload_bytes", "stored_bytes", "escapes", "words", "raw_fallback")}
        if not total["raw_bytes"]:
            raise ValueError("No complete evaluation pages")
        total["saved_pct"] = 100 * (1 - total["stored_bytes"] / total["raw_bytes"])
        total["coded_saved_pct_without_fallback"] = 100 * (1 - total["coded_bytes"] / total["raw_bytes"])
        total["escape_pct"] = 100 * total["escapes"] / total["words"]
        sums[policy] = total
    return {"schema": 1, "capture": manifest, "kind": kind, "bits": bits,
            "baseline": "Independent SplitZip-scheme size/roundtrip reference; not GPU timing or upstream wire format",
            "framing": {"header_bytes_per_page": HEADER.size, "codebook_bytes_per_coded_page": 1 << bits,
                        "bytes_per_escape": ESCAPE.itemsize, "chunk_size": 1024,
                        "raw_fallback_has_header": True, "partial_pages_excluded": True},
            "calibration_tables": {k: v.tolist() for k, v in tables.items()},
            "totals": sums, "rows": rows, "bit_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = screen(args.capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
