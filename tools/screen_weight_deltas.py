# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Whole-checkpoint lossless storage screen, with an exact base charged once.

Compare standalone raw/zstd/byte-shuffled-zstd blocks with exact XOR deltas and
unchanged-block references. Both strategies get the same standalone fallbacks.
This is an independent BitX-style reference, not the ZipLLM implementation.
Restoration verifies the complete original safetensors files byte-for-byte.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import struct
import sys
import time

import numpy as np

FRAME = struct.Struct("<4sBBBBQQQ")
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8,
               "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
               "F8_E4M3": 1, "F8_E4M3FN": 1, "F8_E5M2": 1}


def sha_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


class Zstd:
    def __init__(self, level=3):
        self.level = level
        library = ctypes.util.find_library("zstd")
        if not library:
            raise RuntimeError("Install libzstd; no model/runtime dependency is needed")
        self.lib = ctypes.CDLL(library)
        size, ptr = ctypes.c_size_t, ctypes.c_void_p
        signatures = {
            "ZSTD_compressBound": ([size], size),
            "ZSTD_compress": ([ptr, size, ptr, size, ctypes.c_int], size),
            "ZSTD_decompress": ([ptr, size, ptr, size], size),
            "ZSTD_isError": ([size], ctypes.c_uint),
            "ZSTD_getErrorName": ([size], ctypes.c_char_p),
            "ZSTD_versionString": ([], ctypes.c_char_p),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.version = self.lib.ZSTD_versionString().decode()

    def checked(self, value):
        if self.lib.ZSTD_isError(value):
            raise ValueError(self.lib.ZSTD_getErrorName(value).decode())
        return value

    def compress(self, source: bytes):
        capacity = self.lib.ZSTD_compressBound(len(source))
        target = ctypes.create_string_buffer(capacity)
        size = self.checked(self.lib.ZSTD_compress(target, capacity, ctypes.c_char_p(source), len(source), self.level))
        return target.raw[:size]

    def decompress(self, source: bytes, size: int):
        target = ctypes.create_string_buffer(size)
        written = self.checked(self.lib.ZSTD_decompress(target, size, ctypes.c_char_p(source), len(source)))
        if written != size:
            raise ValueError("Unexpected reconstructed block size")
        return target.raw[:written]


def shuffled(raw: bytes, width: int):
    return np.frombuffer(raw, np.uint8).reshape(-1, width).T.copy().tobytes()


def encode(raw: bytes, width: int, codec: Zstd, xor=False):
    if width not in (1, 2, 4, 8) or len(raw) % width or len(raw) > 64 << 20:
        raise ValueError("Invalid word or block size")
    mode, payload = 0, raw
    candidate = codec.compress(raw)
    if len(candidate) < len(payload):
        mode, payload = 1, candidate
    if width > 1:
        candidate = codec.compress(shuffled(raw, width))
        if len(candidate) < len(payload):
            mode, payload = 2, candidate
    return FRAME.pack(b"ZWD1", 1, mode, width, int(xor), len(raw), len(payload), 0) + payload


def decode(frame: bytes, codec: Zstd):
    if len(frame) < FRAME.size:
        raise ValueError("Truncated weight frame")
    magic, version, mode, width, xor, size, stored, reserved = FRAME.unpack_from(frame)
    if (magic != b"ZWD1" or version != 1 or mode not in (0, 1, 2, 3)
            or width not in (1, 2, 4, 8) or xor not in (0, 1) or reserved
            or size > 64 << 20 or size % width or len(frame) != FRAME.size + stored):
        raise ValueError("Invalid weight frame")
    payload = frame[FRAME.size:]
    if mode == 3:
        if stored or xor:
            raise ValueError("Invalid base-reference frame")
        return None, False, size
    if mode == 0:
        if stored != size:
            raise ValueError("Invalid raw frame")
        raw = payload
    else:
        raw = codec.decompress(payload, size)
        if mode == 2:
            raw = np.frombuffer(raw, np.uint8).reshape(width, -1).T.copy().tobytes()
    return raw, bool(xor), size


def checkpoint(root: Path):
    files, tensors = {}, {}
    for path in sorted(root.glob("*.safetensors")):
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors header")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= min(32 << 20, path.stat().st_size - 8):
                raise ValueError("Invalid safetensors header length")
            header = stream.read(length)
        metadata = json.loads(header)
        intervals = []
        for key, info in metadata.items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise ValueError("Duplicate tensor name across shards")
            dtype, shape, offsets = info["dtype"], info["shape"], info["data_offsets"]
            if (dtype not in DTYPE_BYTES or not isinstance(shape, list)
                    or any(type(v) is not int or v < 0 for v in shape)
                    or len(offsets) != 2 or any(type(v) is not int or v < 0 for v in offsets)):
                raise ValueError("Invalid or unsupported tensor metadata")
            width = DTYPE_BYTES[dtype]
            size = offsets[1] - offsets[0]
            if size != math.prod(shape) * width:
                raise ValueError("Inconsistent tensor byte count")
            intervals.append(tuple(offsets))
            tensors[key] = {"dtype": dtype, "shape": shape, "file": path.name,
                            "offset": 8 + length + offsets[0], "bytes": size, "word_bytes": width}
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor or end < start:
                raise ValueError("Overlapping or uncovered safetensors payload")
            cursor = end
        if cursor + 8 + length != path.stat().st_size:
            raise ValueError("Safetensors payload does not cover its file exactly")
        files[path.name] = {"header_b64": base64.b64encode(prefix + header).decode(),
                           "bytes": path.stat().st_size, "sha256": sha_file(path)}
    if not files:
        raise ValueError("No safetensors weights found")
    return {"schema": 1, "files": files, "tensors": tensors}


class Archive:
    def __init__(self, root, source, block_bytes, label):
        self.root = root
        root.mkdir()
        self.stream = (root / "blocks.bin").open("wb")
        self.manifest = {"schema": 1, "label": label, "block_bytes": block_bytes,
                         "files": source["files"], "tensors": {}}
        self.kinds = Counter()

    def add_tensor(self, key, spec):
        self.manifest["tensors"][key] = {**spec, "blocks": []}

    def write(self, key, start, original, frame, kind):
        info = {"start": start, "raw_bytes": len(original), "offset": self.stream.tell(),
                "stored_bytes": len(frame), "kind": kind, "sha256": hashlib.sha256(original).hexdigest()}
        self.stream.write(frame)
        self.manifest["tensors"][key]["blocks"].append(info)
        self.kinds[kind] += 1

    def finish(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest, indent=2) + "\n")
        return {"stored_bytes": path.stat().st_size + (self.root / "blocks.bin").stat().st_size,
                "manifest_bytes": path.stat().st_size, "block_kinds": dict(self.kinds)}


def read_original(root, spec, start, length):
    with (root / spec["file"]).open("rb") as stream:
        stream.seek(spec["offset"] + start)
        raw = stream.read(length)
    if len(raw) != length:
        raise ValueError("Truncated original tensor")
    return raw


def pack_base(root, source, archive, codec, block_bytes):
    begin = time.perf_counter()
    hist = np.zeros(256, dtype=np.int64)
    for key, spec in source["tensors"].items():
        archive.add_tensor(key, spec)
        for start in range(0, spec["bytes"], block_bytes):
            raw = read_original(root, spec, start, min(block_bytes, spec["bytes"] - start))
            if spec["dtype"] == "BF16":
                hist += np.bincount((np.frombuffer(raw, "<u2") >> 7) & 255, minlength=256)
            archive.write(key, start, raw, encode(raw, spec["word_bytes"], codec), "independent")
    report = archive.finish()
    report.update(pack_wall_seconds=time.perf_counter() - begin, bf16_exponent_histogram=hist.tolist())
    return report


def pack_pair(base_root, base_source, target_root, target_source, standalone, delta, codec, block_bytes):
    begin = time.perf_counter()
    hist = np.zeros(256, dtype=np.int64)
    independent_encode, delta_encode, eligible_bytes = 0.0, 0.0, 0
    for key, spec in target_source["tensors"].items():
        standalone.add_tensor(key, spec)
        delta.add_tensor(key, spec)
        base = base_source["tensors"].get(key)
        aligned = base is not None and all(base[field] == spec[field] for field in ("dtype", "shape", "bytes"))
        for start in range(0, spec["bytes"], block_bytes):
            raw = read_original(target_root, spec, start, min(block_bytes, spec["bytes"] - start))
            if spec["dtype"] == "BF16":
                hist += np.bincount((np.frombuffer(raw, "<u2") >> 7) & 255, minlength=256)
            tick = time.perf_counter()
            independent = encode(raw, spec["word_bytes"], codec)
            independent_encode += time.perf_counter() - tick
            standalone.write(key, start, raw, independent, "independent")
            chosen, kind = independent, "independent"
            if aligned:
                eligible_bytes += len(raw)
                original = read_original(base_root, base, start, len(raw))
                tick = time.perf_counter()
                if original == raw:
                    chosen = FRAME.pack(b"ZWD1", 1, 3, spec["word_bytes"], 0, len(raw), 0, 0)
                    kind = "reference"
                else:
                    mask = np.bitwise_xor(np.frombuffer(raw, np.uint8), np.frombuffer(original, np.uint8)).tobytes()
                    encoded = encode(mask, spec["word_bytes"], codec, xor=True)
                    if len(encoded) < len(independent):
                        chosen, kind = encoded, "delta"
                delta_encode += time.perf_counter() - tick
            delta.write(key, start, raw, chosen, kind)
    independent_result, delta_result = standalone.finish(), delta.finish()
    return {"standalone": independent_result, "delta": delta_result, "eligible_bytes": eligible_bytes,
            "paired_pack_wall_seconds": time.perf_counter() - begin,
            "standalone_encode_seconds": independent_encode, "delta_extra_encode_seconds": delta_encode,
            "bf16_exponent_histogram": hist.tolist()}


def restore(archive_root: Path, output: Path, codec: Zstd, base_root: Path | None = None):
    manifest = json.loads((archive_root / "manifest.json").read_text())
    base = json.loads((base_root / "manifest.json").read_text()) if base_root else None
    if manifest.get("base_manifest_sha256") and (not base_root or
            sha_file(base_root / "manifest.json") != manifest["base_manifest_sha256"]):
        raise ValueError("The exact pinned base archive is required")
    if any(Path(name).name != name for name in manifest["files"]):
        raise ValueError("Unsafe checkpoint filename")
    output.mkdir()
    streams = {}
    encoded = (archive_root / "blocks.bin").open("rb")
    base_stream = (base_root / "blocks.bin").open("rb") if base else None
    timings = Counter()
    begin = time.perf_counter()
    try:
        for name, info in manifest["files"].items():
            stream = (output / name).open("wb+")
            stream.write(base64.b64decode(info["header_b64"]))
            stream.truncate(info["bytes"])
            streams[name] = stream
        for key, spec in manifest["tensors"].items():
            for i, block in enumerate(spec["blocks"]):
                tick = time.perf_counter()
                encoded.seek(block["offset"])
                raw, xor, size = decode(encoded.read(block["stored_bytes"]), codec)
                timings["target_read_decode_seconds"] += time.perf_counter() - tick
                if raw is None or xor:
                    if not base or key not in base["tensors"]:
                        raise ValueError("Required exact base tensor is missing")
                    base_spec = base["tensors"][key]
                    base_block = base_spec["blocks"][i]
                    if (base_spec["dtype"] != spec["dtype"] or base_spec["shape"] != spec["shape"]
                            or base_block["start"] != block["start"] or base_block["raw_bytes"] != size):
                        raise ValueError("Base tensor alignment mismatch")
                    tick = time.perf_counter()
                    base_stream.seek(base_block["offset"])
                    original, base_xor, base_size = decode(base_stream.read(base_block["stored_bytes"]), codec)
                    if original is None or base_xor or base_size != size:
                        raise ValueError("Nested or inconsistent base frame")
                    timings["base_read_decode_seconds"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    raw = (np.bitwise_xor(np.frombuffer(raw, np.uint8), np.frombuffer(original, np.uint8)).tobytes()
                           if xor else original)
                    timings["xor_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                if hashlib.sha256(raw).hexdigest() != block["sha256"] or len(raw) != block["raw_bytes"]:
                    raise ValueError("Reconstructed block checksum mismatch")
                timings["block_validation_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                stream = streams[spec["file"]]
                stream.seek(spec["offset"] + block["start"])
                stream.write(raw)
                timings["write_seconds"] += time.perf_counter() - tick
        tick = time.perf_counter()
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
        timings["fsync_seconds"] = time.perf_counter() - tick
    finally:
        for stream in streams.values():
            stream.close()
        encoded.close()
        if base_stream:
            base_stream.close()
    timings["wall_seconds"] = time.perf_counter() - begin
    for name, info in manifest["files"].items():
        if sha_file(output / name) != info["sha256"]:
            raise AssertionError("Full restored safetensors file differs from original")
    return {**timings, "whole_file_sha256_match": True,
            "source_bytes": sum(info["bytes"] for info in manifest["files"].values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--base-id", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--block-mib", type=int, default=8)
    parser.add_argument("--keep-restored", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.block_mib <= 64:
        parser.error("block-mib must be 1..64")
    args.output.mkdir(parents=True, exist_ok=False)
    codec = Zstd(args.level)
    base_source, target_source = checkpoint(args.base), checkpoint(args.target)
    block_bytes = args.block_mib << 20
    base_archive = Archive(args.output / "base", base_source, block_bytes, args.base_id)
    print("PACK_BASE", flush=True)
    base_result = pack_base(args.base, base_source, base_archive, codec, block_bytes)
    independent = Archive(args.output / "standalone", target_source, block_bytes, args.target_id)
    delta = Archive(args.output / "delta", target_source, block_bytes, args.target_id)
    delta.manifest["base_manifest_sha256"] = sha_file(args.output / "base/manifest.json")
    print("PACK_TARGET_AND_DELTA", flush=True)
    paired = pack_pair(args.base, base_source, args.target, target_source, independent, delta, codec, block_bytes)
    results = {"schema": 1, "base_id": args.base_id, "base_revision": args.base_revision,
               "target_id": args.target_id, "target_revision": args.target_revision,
               "zstd_version": codec.version, "zstd_level": args.level, "block_bytes": block_bytes,
               "source_bytes": {"base": sum(v["bytes"] for v in base_source["files"].values()),
                                "target": sum(v["bytes"] for v in target_source["files"].values())},
               "base": base_result, **paired,
               "scope": "Full safetensors weights and original headers; configs/tokenizers unchanged and excluded. Decode benchmarks use warm filesystem cache and fsynced outputs; decoded base is NOT cached."}
    sizes = {"base_plus_standalone": base_result["stored_bytes"] + paired["standalone"]["stored_bytes"],
             "base_plus_delta": base_result["stored_bytes"] + paired["delta"]["stored_bytes"]}
    sizes["joint_saved_pct_vs_independent"] = 100 * (1 - sizes["base_plus_delta"] / sizes["base_plus_standalone"])
    sizes["joint_saved_pct_vs_target_only"] = 100 * (1 - sizes["base_plus_delta"] / paired["standalone"]["stored_bytes"])
    results["totals"] = sizes
    print("RESTORE_AND_VERIFY", json.dumps(sizes), flush=True)
    results["restoration"] = {}
    # Pair two orders to expose filesystem-cache/order bias in this small screen.
    for i, name in enumerate(("standalone", "delta", "delta", "standalone")):
        path = args.output / f"restored-{i}-{name}"
        measured = restore(args.output / name, path, codec, args.output / "base" if name == "delta" else None)
        results["restoration"].setdefault(name, []).append(measured)
        print("RESTORED", name, json.dumps(measured), flush=True)
        if not args.keep_restored:
            shutil.rmtree(path)  # Only this invocation's verified reconstruction.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    results["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    results["script_sha256"] = sha_file(Path(__file__))
    (args.output / "readout.json").write_text(json.dumps(results, indent=2) + "\n")
    print("STORAGE_SCREEN_COMPLETE", json.dumps(sizes), flush=True)


if __name__ == "__main__":
    main()
