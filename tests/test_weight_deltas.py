# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Exact delta storage, independent fallbacks and original-file restoration."""
from pathlib import Path
import ctypes
import ctypes.util
import json
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from screen_weight_deltas import Archive, FRAME, Zstd, checkpoint, decode, encode, pack_base, pack_pair, restore, sha_file


def safetensors(path, items):
    header, body = {}, b""
    for name, dtype, width, raw in items:
        header[name] = {"dtype": dtype, "shape": [len(raw) // width], "data_offsets": [len(body), len(body) + len(raw)]}
        body += raw
    raw_header = json.dumps(header).encode() + b"   "
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + body)


@pytest.fixture
def codec():
    library = ctypes.util.find_library("zstd")
    if not library:
        pytest.skip("libzstd is not installed or discoverable")
    try:
        ctypes.CDLL(library)
    except OSError as error:
        pytest.skip(f"libzstd cannot be loaded: {error}")
    return Zstd(3)


def test_frames_preserve_all_bf16_bits_and_reject_corruption(codec):
    raw = np.arange(65536, dtype="<u2").tobytes()
    for width in (1, 2, 4, 8):
        frame = encode(raw, width, codec, xor=True)
        decoded, xor, size = decode(frame, codec)
        assert decoded == raw and xor and size == len(raw)
        with pytest.raises(ValueError):
            decode(frame[:-1], codec)
    with pytest.raises(ValueError):
        encode(b"odd", 2, codec)


def test_full_checkpoint_restore_charges_base_and_handles_unmatched_tensors(tmp_path, codec):
    rng = np.random.default_rng(13)
    raw = ((126 << 7) | rng.integers(0, 128, 32768, dtype=np.uint16)).astype("<u2").tobytes()
    changed = bytearray(raw)
    changed[1000] ^= 1
    changed = bytes(changed)
    base_path, target_path = tmp_path / "input-base", tmp_path / "input-target"
    base_path.mkdir()
    target_path.mkdir()
    safetensors(base_path / "model.safetensors", [("same", "BF16", 2, raw), ("changed", "BF16", 2, raw)])
    # Different shard layout and header whitespace must also be restored exactly.
    safetensors(target_path / "one.safetensors", [("changed", "BF16", 2, changed)])
    safetensors(target_path / "two.safetensors", [("same", "BF16", 2, raw), ("new", "U8", 1, b"new value" * 100)])
    base, target = checkpoint(base_path), checkpoint(target_path)
    base_store = Archive(tmp_path / "base", base, 16384, "base")
    base_result = pack_base(base_path, base, base_store, codec, 16384)
    independent = Archive(tmp_path / "standalone", target, 16384, "target")
    delta = Archive(tmp_path / "delta", target, 16384, "target")
    delta.manifest["base_manifest_sha256"] = sha_file(tmp_path / "base/manifest.json")
    result = pack_pair(base_path, base, target_path, target, independent, delta, codec, 16384)
    assert result["delta"]["block_kinds"].keys() >= {"reference", "delta", "independent"}
    assert result["eligible_bytes"] == len(raw) * 2
    assert result["delta"]["stored_bytes"] < result["standalone"]["stored_bytes"]
    assert base_result["stored_bytes"] > 0
    for name in ("standalone", "delta"):
        out = tmp_path / ("restore-" + name)
        report = restore(tmp_path / name, out, codec, tmp_path / "base" if name == "delta" else None)
        assert report["whole_file_sha256_match"]
        for file in target_path.glob("*.safetensors"):
            assert (out / file.name).read_bytes() == file.read_bytes()
    with pytest.raises(ValueError, match="exact pinned base"):
        restore(tmp_path / "delta", tmp_path / "missing-base", codec)


def test_checkpoint_rejects_uncovered_payload(tmp_path):
    safetensors(tmp_path / "model.safetensors", [("x", "U8", 1, b"abcd")])
    with (tmp_path / "model.safetensors").open("ab") as stream:
        stream.write(b"trailing garbage")
    with pytest.raises(ValueError, match="cover"):
        checkpoint(tmp_path)
