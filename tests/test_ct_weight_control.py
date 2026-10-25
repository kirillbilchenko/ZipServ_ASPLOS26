# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Control configuration must disable weight conversions, not correctness fixes."""
import sys
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ct_weight_control as observer
from launch_ct_weight_control import private_environment
from compare_ct_weight_control import compare_launch


def template():
    return ["MXWAVE_COMPARE_ARM=B", "MXWAVE_MOE_MODE=aligned",
            "MXWAVE_GLM53_VOCAB_MODE=packed", "MXWAVE_GLM53_KDA_MODE=stock",
            "MXWAVE_PALETTE_MODEL=1", "GLM_LV_DRAFT_FP8_KV=1",
            "GLM_DENSE_FAST=1", "MXWAVE_GLM53_ATTENTION_MAPPING=1",
            "GLM_DS_DRAFT_HEAD_FP8=1", "GLM_CERT_HEAD=1", "GLM_MHC_BF16W=1",
            "VLLM_API_KEY=private", "ACCESS_TOKEN_FILE=/run/secrets/private"]


def test_optional_weights_off_preserves_correctness_and_compute():
    env = private_environment(template(), "off")
    assert all(env[key] == "0" for key in observer.DISABLED)
    assert env["MXWAVE_COMPARE_ARM"] == "A"
    assert env["MXWAVE_MOE_MODE"] == env["MXWAVE_GLM53_VOCAB_MODE"] == "stock"
    assert env["GLM_LV_DRAFT_FP8_KV"] == env["GLM_DENSE_FAST"] == "1"
    assert env["MXWAVE_GLM53_ATTENTION_MAPPING"] == "1"
    assert "VLLM_API_KEY" not in env and "ACCESS_TOKEN_FILE" not in env


def test_ct_mode_keeps_original_weight_recipe():
    env = private_environment(template(), "ct")
    assert env["MXWAVE_COMPARE_ARM"] == "B"
    assert env["MXWAVE_MOE_MODE"] == "aligned"
    assert env["GLM_CERT_HEAD"] == env["GLM_DS_DRAFT_HEAD_FP8"] == "1"


def test_guard_rejects_hidden_conversion_and_wrong_runtime(monkeypatch):
    # Other tests import the modern plugin during collection. Model the clean
    # legacy worker here, then explicitly check that the guard rejects a plugin.
    for name in tuple(sys.modules):
        if name.startswith("zipserv_vllm"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(observer.os, "environ", private_environment(template(), "off"))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__=observer.VERSION))
    assert observer.validate_environment() == "off"
    monkeypatch.setenv("GLM_CERT_HEAD", "1")
    with pytest.raises(RuntimeError, match="still enabled"):
        observer.validate_environment()
    monkeypatch.setenv("GLM_CERT_HEAD", "0")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.32.0"))
    with pytest.raises(RuntimeError, match="exact CT runtime"):
        observer.validate_environment()
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__=observer.VERSION))
    monkeypatch.setitem(sys.modules, "zipserv_vllm", SimpleNamespace())
    with pytest.raises(RuntimeError, match="modern ZipServ plugin"):
        observer.validate_environment()


def test_guard_rejects_disabling_draft_correctness_fix(monkeypatch):
    monkeypatch.setattr(observer.os, "environ", private_environment(template(), "off"))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__=observer.VERSION))
    monkeypatch.setenv("GLM_LV_DRAFT_FP8_KV", "0")
    with pytest.raises(RuntimeError, match="correctness fix"):
        observer.validate_environment()


def test_comparison_rejects_compute_and_source_changes():
    raw = {key: "same" for key in ("image", "template", "rank", "cmd", "host_config", "sources", "tools")}
    raw.update(mode="off", env=private_environment(template(), "off"),
               mounts=[], returncode=0, preflight=False)
    ct = {**raw, "mode": "ct", "env": private_environment(template(), "ct")}
    assert "GLM_CERT_HEAD" in compare_launch(raw, ct)
    changed = copy.deepcopy(ct)
    changed["env"]["GLM_DENSE_FAST"] = "0"
    with pytest.raises(ValueError, match="Non-weight flags"):
        compare_launch(raw, changed)
    changed = {**ct, "sources": "other"}
    with pytest.raises(ValueError, match="Launch mismatch: sources"):
        compare_launch(raw, changed)
