# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for exact scale bytes, fallback, and plugin opt-in contracts."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
from zipserv_vllm import compatibility
from zipserv_vllm.nvfp4_codec import decode, encode
from zipserv_vllm.nvfp4_plugin import checkpoint_recipes, settings
from zipserv_vllm.nvfp4_runtime import layout_rejection


def test_all_scale_bytes_and_high_plane_boundaries():
    rng = np.random.default_rng(271)
    for base in range(256):
        for size in (16, 32, 48, 512):
            raw = rng.integers(base, min(base + 32, 256), size=size, dtype=np.uint8)
            raw[0] = base
            packed = encode(raw)
            assert not packed.raw.size
            assert np.array_equal(decode(packed), raw)
            assert packed.high.nbytes % 4 == 0


def test_raw_fallback_does_not_alias_or_requantize():
    for raw in (np.arange(256, dtype=np.uint8), np.tile(np.array([0, 32], np.uint8), 16)):
        expected = raw.copy()
        part = encode(raw)
        raw[:] = 7
        assert part.raw.size
        assert np.array_equal(decode(part), expected)
    source = np.arange(64, dtype=np.uint8).reshape(8, 8)[:, :4]
    assert np.array_equal(decode(encode(source)), source.reshape(-1))


def test_corrupt_record_bounds_palette_and_carry_are_rejected():
    part = encode(np.full(16, 250, dtype=np.uint8))
    for bad in (replace(part, size=17), replace(part, high=part.high[:1]),
                replace(part, low=part.low.astype(np.uint16)),
                replace(part, palette=np.zeros(32, np.uint8)),
                replace(part, low=np.full_like(part.low, 0xffffffff))):
        with pytest.raises(ValueError):
            decode(bad)
    for raw in (np.zeros(0, np.uint8), np.zeros(17, np.uint8), np.zeros(16, np.float32)):
        with pytest.raises(ValueError):
            encode(raw)


def test_layout_not_model_name_controls_eligibility():
    assert layout_rejection((3, 4, 128), 2) is None
    assert layout_rejection((17, 8, 1024), 1) is None
    for shape, parts in (((3, 4, 192), 1), ((3, 3, 128), 1), ((1, 4, 64), 3),
                         ((0, 4, 128), 2), ((2**24, 4, 128), 2)):
        assert layout_rejection(shape, parts)


def test_opt_in_is_inert_and_explicit_off_needs_no_library():
    assert settings({}) is None
    assert settings({"ZIPSERV_NVFP4_SCALES": "off"}).library is None
    for env in ({"ZIPSERV_NVFP4_SCALES": "auto"}, {"ZIPSERV_NVFP4_SCALES": "on"},
                {"ZIPSERV_NVFP4_SCALES": "off", "ZIPSERV_MXFP4_SCALES": "on"},
                {"ZIPSERV_NVFP4_SCALES": "off", "ZIPSERV_NVFP4_MIN_SAVING_PERCENT": "nan"}):
        with pytest.raises(ValueError):
            settings(env)


def test_csf_is_never_classified_as_plain_nvfp4():
    config = {"quantization_config": {"quantized_layers": {
        "expert": {"quant_algo": "NVFP4", "weight_scale_encoding": "csf_32x2_v1", "group_size": 16},
        "future": {"quant_algo": "W4A16_NVFP4", "weight_scale_encoding": "future_codec"},
        "mtp": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
        "a4": {"quant_algo": "NVFP4", "group_size": 16},
        "head": {"quant_algo": "MXFP8"},
    }}}
    report = checkpoint_recipes(config)
    assert report["already_encoded"] == ["expert", "future"]
    assert report["plain_w4a16_group16"] == ["mtp"]
    assert report["activation_quantized_or_backend_dependent"] == ["a4"]


def test_source_profiles_reject_tampering_and_unknown_versions(tmp_path, monkeypatch):
    raw = b"reviewed loader API"
    manifest = {"adapters": {"nvfp4": ["loader.py"]}, "releases": {
        version: {"sha256": {"loader.py": hashlib.sha256(raw).hexdigest()}}
        for version in compatibility.SUPPORTED_RELEASES}}
    (tmp_path / "vllm-compatibility.json").write_text(json.dumps(manifest))
    (tmp_path / "loader.py").write_bytes(raw)
    monkeypatch.setattr(compatibility, "__file__", str(tmp_path / "compatibility.py"))
    for version in ("0.31.0", "0.32.0+local"):
        assert compatibility.check_source(tmp_path, version, "nvfp4") == version.split("+")[0]
    for version in ("0.33.0", "0.32.0rc1", "0.32.0.post1", "invalid"):
        with pytest.raises(RuntimeError, match="Unsupported"):
            compatibility.check_source(tmp_path, version, "nvfp4")
    (tmp_path / "loader.py").write_bytes(raw + b" changed")
    with pytest.raises(RuntimeError, match="loader.py"):
        compatibility.check_source(tmp_path, "0.32.0", "nvfp4")
    (tmp_path / "loader.py").unlink()
    with pytest.raises(RuntimeError, match="loader.py"):
        compatibility.check_source(tmp_path, "0.31.0", "nvfp4")
