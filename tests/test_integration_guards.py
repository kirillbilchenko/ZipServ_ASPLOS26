# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests of startup selection and native descriptor ownership."""
from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError
import logging
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations/vllm"))
import zipserv_vllm
from zipserv_vllm.bf16_policy import (
    DispatchSettings, decode_budget_rejection, gemm_rejection, select_split, split_candidates, startup_rejection,
)
from zipserv_vllm.scale_ownership import scale_metadata, storage_identity


@pytest.mark.parametrize("version", ["0.30.0", "0.33.0", "0.32.0rc1", "0.32.0.dev1",
                                     "0.32.0.post1", "0.1.dev20051+g487ecf187", "invalid"])
def test_unsupported_plugin_registration_warns_and_does_not_import(monkeypatch, caplog, version):
    monkeypatch.setattr("importlib.metadata.version", lambda name: version)
    # Importing either implementation would execute registration; these sentinels
    # make this test independent of which optional modules other tests imported.
    import builtins
    original = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        fromlist = fromlist or ()
        if "hybrid" in fromlist or "vllm_plugin" in fromlist:
            raise AssertionError("Unsupported registration imported its implementation")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with caplog.at_level(logging.WARNING):
        zipserv_vllm.register()
    assert "Skipping ZipServ registration" in caplog.text


def test_missing_vllm_metadata_is_inert(monkeypatch, caplog):
    def missing(name):
        raise PackageNotFoundError(name)
    monkeypatch.setattr("importlib.metadata.version", missing)
    with caplog.at_level(logging.WARNING):
        zipserv_vllm.register()
    assert "not installed" in caplog.text


@pytest.mark.parametrize("version", ["0.31.0", "0.32.0+local"])
def test_supported_registration_imports_both_methods(monkeypatch, version):
    monkeypatch.setattr("importlib.metadata.version", lambda name: version)
    checks = []
    monkeypatch.setattr("zipserv_vllm.compatibility.check_upstream", lambda *names: checks.append(names))
    import builtins
    original = builtins.__import__
    imports = []

    def observed(name, globals=None, locals=None, fromlist=(), level=0):
        fromlist = fromlist or ()
        if "hybrid" in fromlist or "vllm_plugin" in fromlist:
            imports.append(tuple(fromlist))
            return types.SimpleNamespace(hybrid=object(), vllm_plugin=object())
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", observed)
    zipserv_vllm.register()
    assert imports == [("vllm_plugin",), ("hybrid",)]
    assert checks == [("bf16",), ("hybrid",)]


def test_startup_limits_cover_all_native_products_and_grid():
    assert startup_rejection(131072, 4096, 4, 4095) is None
    assert "int32" in startup_rejection(131072, 4096, 4, 4096)
    assert "int32" in startup_rejection(64, 65536, 1, 32768)
    assert "grid.y" in startup_rejection(4194304, 64, 1, 1)
    assert "offset bound" in startup_rejection(64, 33554368, 1, 1)
    assert "divide" in startup_rejection(128, 192, 2, 128)
    for maximum in (None, 0, -1, True):
        assert "scheduler" in startup_rejection(128, 128, 1, maximum)
    assert gemm_rejection(128, 0, 128, 1) is None


def test_split_search_covers_divisors_workspace_and_token_grid():
    rows = split_candidates(128, 7, 960, 20, 2**20)
    assert [r.split_k for r in rows] == [1, 3, 5, 15]
    assert [r.blocks for r in rows] == [2, 6, 10, 30]
    limited = split_candidates(128, 7, 960, 20, 128 * 7 * 3 * 2)
    assert [r.split_k for r in limited] == [1, 3]
    fp32 = split_candidates(128, 7, 960, 20, 128 * 7 * 3 * 2, 4)
    assert [r.split_k for r in fp32] == [1]
    assert split_candidates(128, 129, 960, 20, 2**20)[0].blocks == 4
    assert split_candidates(128, 257, 960, 20, 2**20)[0].blocks == 6
    assert split_candidates(128, 2**31, 960, 20, 2**20) == ()
    settings = DispatchSettings(mode="shape", workspace_bytes=0)
    assert select_split(1024, 1, 4096, 132, settings) == 1


def test_token_dispatch_boundaries_and_non_power_of_two_fallback():
    settings = DispatchSettings(mode="tokens")
    for tokens, split in ((1, 8), (32, 8), (33, 4), (128, 4), (129, 1), (768, 1), (769, None)):
        assert select_split(1024, tokens, 4096, 132, settings) == split
    assert select_split(128, 1, 192, 132, settings) == 3
    assert select_split(128, 129, 192, 132, settings) == 1
    for kwargs in ({"mode": "auto"}, {"partial_dtype": "fp16"},
                   {"large_batch_tokens": True}, {"workspace_bytes": -1}):
        with pytest.raises(ValueError):
            DispatchSettings(**kwargs)


def test_unknown_descriptor_aliases_and_copies_are_rejected():
    descriptor = torch.nn.Parameter(torch.arange(3, dtype=torch.int64), requires_grad=False)
    metadata = {"replacement": descriptor}
    owners = {id(descriptor): metadata}
    storages = {storage_identity(descriptor)}
    assert scale_metadata(descriptor, owners, storages) is metadata
    for candidate in (descriptor.detach(), descriptor.view(-1), descriptor[1:],
                      descriptor.view(torch.uint8), descriptor.clone()):
        with pytest.raises(RuntimeError, match="Unrecognised.*descriptor"):
            scale_metadata(candidate, owners, storages)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        assert scale_metadata(torch.ones(2, 3, 64, dtype=dtype), owners, storages) is None
    assert scale_metadata(None, owners, storages) is None


def test_installed_mxfp4_wrapper_keeps_raw_fallback_but_rejects_descriptor_alias(monkeypatch):
    from zipserv_vllm import mxfp4_backend as backend

    calls = []

    def stock(b_scales, tag=None):
        calls.append((b_scales, tag))
        return "stock-result"

    ops = types.SimpleNamespace(moe_wna16_marlin_gemm=stock)
    vllm = types.ModuleType("vllm")
    vllm._custom_ops = ops
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    scalar = types.ModuleType("vllm.scalar_type")
    scalar.scalar_types = types.SimpleNamespace(float4_e2m1f=object())
    monkeypatch.setitem(sys.modules, scalar.__name__, scalar)
    name = ("vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4")
    preparation = types.ModuleType(name)

    def prepare(method, layer):
        pass

    prepare.__module__ = name
    preparation.CompressedTensorsW4A4Mxfp4MoEMethod = type(
        "Method", (), {"process_weights_after_loading": prepare})
    monkeypatch.setitem(sys.modules, name, preparation)
    descriptor = torch.tensor([1, 2, 3], dtype=torch.int64)
    monkeypatch.setattr(backend, "_INSTALLED", False)
    monkeypatch.setattr(backend, "_SETTINGS", None)
    monkeypatch.setattr(backend, "_STORAGE", {id(descriptor): {"replacement": descriptor}})
    monkeypatch.setattr(backend, "_DESCRIPTOR_STORAGES", {storage_identity(descriptor)})
    monkeypatch.setattr(backend, "native", lambda kind: None)
    backend.install(types.SimpleNamespace(mode="on"))
    raw = torch.ones(2, 2, dtype=torch.bfloat16)
    assert ops.moe_wna16_marlin_gemm(raw, "positional") == "stock-result"
    assert ops.moe_wna16_marlin_gemm(b_scales=raw, tag="keyword") == "stock-result"
    assert [tag for _, tag in calls] == ["positional", "keyword"]
    for unknown in (descriptor.detach(), descriptor.clone(), descriptor.view(torch.uint8)):
        with pytest.raises(RuntimeError, match="Unrecognised.*descriptor"):
            ops.moe_wna16_marlin_gemm(unknown)
    assert len(calls) == 2


def test_scale_cache_checks_both_outer_and_native_descriptor_boundaries(monkeypatch):
    from contextlib import nullcontext
    from zipserv_vllm import scale_cache_plugin as plugin, scale_cache

    descriptor = torch.tensor([1, 2, 3], dtype=torch.int64)
    metadata = {"replacement": descriptor, "cache_key": "test"}
    backend = types.ModuleType("model_plugin")
    calls = []
    backend.prepare_layer = lambda *args: None
    backend.install = lambda: None
    backend._STORAGE = {id(descriptor): metadata}
    backend._GEMM_OP = lambda x, weight, scales: calls.append(scales) or "consumer"
    monkeypatch.setitem(sys.modules, "model_plugin", backend)

    def outer(input, b_qweight, b_scales):
        return backend._GEMM_OP(input, b_qweight, b_scales)

    ops = types.SimpleNamespace(moe_wna16_marlin_gemm=outer)
    vllm = types.ModuleType("vllm"); vllm._custom_ops = ops
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    for key, value in {"ZIPSERV_SCALE_DICTIONARY": "1", "ZIPSERV_SCALE_BACKEND_MODULE": "model_plugin",
                       "ZIPSERV_SCALE_REPRESENTATION": "dictionary", "ZIPSERV_SCALE_CACHE": "0",
                       "MXWAVE_MOE_MODE": "aligned", "ZIPSERV_SCALE_DICTIONARY_LIBRARY": "/unused"}.items():
        monkeypatch.setenv(key, value)
    for name, value in {"_INSTALLED": False, "_LIBRARY": None, "_BACKEND": None,
                        "_REPRESENTATION": "dictionary", "_DESCRIPTORS": {id(descriptor): metadata},
                        "_DESCRIPTOR_STORAGES": {storage_identity(descriptor)},
                        "_CACHES": {None: types.SimpleNamespace(borrow=lambda key: nullcontext(descriptor))}}.items():
        monkeypatch.setattr(plugin, name, value)
    monkeypatch.setattr(plugin, "version", lambda name: "0.32.0")
    monkeypatch.setattr(scale_cache, "load_dictionary_library", lambda path: object())
    plugin.register()
    raw = torch.ones(2, 2, dtype=torch.bfloat16)
    assert ops.moe_wna16_marlin_gemm(None, None, raw) == "consumer"
    assert ops.moe_wna16_marlin_gemm(input=None, b_qweight=None, b_scales=descriptor) == "consumer"
    for bad in (descriptor.detach(), descriptor.clone(), descriptor.view(torch.uint8)):
        with pytest.raises(RuntimeError, match="Unrecognised.*descriptor"):
            ops.moe_wna16_marlin_gemm(None, None, bad)
        with pytest.raises(RuntimeError, match="Unrecognised.*descriptor"):
            backend._GEMM_OP(None, None, bad)
    assert len(calls) == 2


@pytest.fixture
def plugin(monkeypatch):
    """Use actual plugin logic with only the vLLM class/config boundary stubbed."""
    class LinearBase(torch.nn.Module):
        pass

    class GateLinear(LinearBase):
        pass

    class UnquantizedLinearMethod:
        pass

    config = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=128),
        compilation_config=types.SimpleNamespace(mode=0, max_cudagraph_capture_size=128,
            cudagraph_mode=types.SimpleNamespace(name="NONE")))
    definitions = {
        "vllm.model_executor.layers.linear": dict(
            LinearBase=LinearBase, LinearMethodBase=object,
            UnquantizedLinearMethod=UnquantizedLinearMethod,
            dispatch_unquantized_gemm=lambda: lambda layer, x, weight, bias: torch.nn.functional.linear(x, weight, bias),
            register_weight_loader_v2_supported_method=lambda cls: cls),
        "vllm.model_executor.layers.quantization": dict(register_quantization_config=lambda name: lambda cls: cls),
        "vllm.model_executor.layers.quantization.base_config": dict(QuantizationConfig=object),
        "vllm.model_executor.layers.fused_moe.router.gate_linear": dict(GateLinear=GateLinear),
        "vllm.model_executor.parameter": dict(ModelWeightParameter=torch.nn.Parameter),
        "vllm.model_executor.utils": dict(set_weight_attrs=lambda *args: None),
        "vllm.config": dict(get_current_vllm_config=lambda: config),
    }
    for name, contents in definitions.items():
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            key = ".".join(parts[:i])
            if key not in definitions or key == name:
                if key not in sys.modules or key == name:
                    module = types.ModuleType(key)
                    module.__path__ = []
                    monkeypatch.setitem(sys.modules, key, module)
        sys.modules[name].__dict__.update(contents)
    spec = importlib.util.spec_from_file_location(
        "zipserv_vllm._guard_test_plugin", ROOT / "integrations/vllm/zipserv_vllm/vllm_plugin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, config, LinearBase, GateLinear, UnquantizedLinearMethod


def test_router_uses_stock_method_and_dense_linear_remains_eligible(plugin):
    module, config, LinearBase, GateLinear, stock = plugin
    policy = module.ZipServConfig(ignore=("ignored",))
    assert isinstance(policy.get_quant_method(GateLinear(), "router"), stock)
    assert isinstance(policy.get_quant_method(LinearBase(), "ignored"), stock)
    assert isinstance(policy.get_quant_method(LinearBase(), "ordinary"), module.ZipServLinearMethod)


def test_unsafe_startup_retains_dense_weight_before_packing(plugin, monkeypatch, caplog):
    module, config, LinearBase, _, _ = plugin
    config.scheduler_config.max_num_batched_tokens = 2**31
    layer = LinearBase()
    layer.weight = torch.nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16), requires_grad=False)
    before = layer.weight.detach().clone()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor)

    def forbidden(weight):
        raise AssertionError("Unsafe layer reached the CPU encoder")

    monkeypatch.setattr(module, "compress", forbidden)
    method = module.ZipServLinearMethod()
    with caplog.at_level(logging.WARNING):
        method.process_weights_after_loading(layer)
    assert not layer.zipserv_compressed
    assert "int32" in layer.zipserv_fallback_reason
    assert torch.equal(before, layer.weight)
    assert "keeps dense" in caplog.text
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    assert torch.equal(method.apply(layer, x), torch.nn.functional.linear(x, before))


def test_graph_padding_limit_is_checked_at_startup(plugin, monkeypatch):
    module, config, LinearBase, _, _ = plugin
    config.compilation_config.max_cudagraph_capture_size = 2**31
    layer = LinearBase()
    layer.weight = torch.nn.Parameter(torch.zeros(64, 64, dtype=torch.bfloat16))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor)
    module.ZipServLinearMethod().process_weights_after_loading(layer)
    assert not layer.zipserv_compressed
    assert layer.zipserv_max_tokens == 2**31


def test_dispatch_rejects_unqualified_graph_and_compile_modes(plugin, monkeypatch):
    module, config, LinearBase, _, _ = plugin
    layer = LinearBase()
    layer.weight = torch.nn.Parameter(torch.zeros(64, 64, dtype=torch.bfloat16))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    for compile_mode, graph_mode in ((1, "NONE"), (0, "FULL"), (0, "PIECEWISE")):
        config.compilation_config.mode = compile_mode
        config.compilation_config.cudagraph_mode.name = graph_mode
        with pytest.raises(ValueError, match="decode-only graphs"):
            module.ZipServLinearMethod(dispatch=DispatchSettings(mode="tokens")).process_weights_after_loading(layer)


def test_decode_graph_above_fused_threshold_keeps_dense_at_startup(plugin, monkeypatch):
    module, config, LinearBase, _, _ = plugin
    config.compilation_config.cudagraph_mode.name = "FULL_DECODE_ONLY"
    config.compilation_config.max_cudagraph_capture_size = 1024
    layer = LinearBase()
    layer.weight = torch.nn.Parameter(torch.zeros(64, 64, dtype=torch.bfloat16))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor)
    module.ZipServLinearMethod(dispatch=DispatchSettings(mode="tokens")).process_weights_after_loading(layer)
    assert not layer.zipserv_compressed
    assert "graph capture maximum" in layer.zipserv_fallback_reason


def test_decoded_weight_budget_keeps_large_layers_dense_before_packing(plugin, monkeypatch):
    module, _, LinearBase, _, _ = plugin
    assert "939524096" in decode_budget_rejection(57344, 8192, DispatchSettings(mode="tokens"))
    assert decode_budget_rejection(57344, 8192, DispatchSettings()) is None
    layer = LinearBase()
    layer.weight = torch.nn.Parameter(torch.zeros(64, 64, dtype=torch.bfloat16))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor)
    module.ZipServLinearMethod(dispatch=DispatchSettings(mode="tokens", decode_workspace_bytes=8191)).process_weights_after_loading(layer)
    assert not layer.zipserv_compressed
    assert "decode_workspace_bytes" in layer.zipserv_fallback_reason


def test_original_plugin_dispatches_large_batches_to_stock_and_guards_capture(plugin, monkeypatch):
    module, _, LinearBase, _, _ = plugin
    layer = LinearBase()
    layer.zipserv_compressed = True
    layer.zipserv_metadata = (64, 192, 4096, 0, 120)
    layer.zipserv_sm_count = 20
    for name in module.BUFFER_NAMES:
        layer.register_buffer("zipserv_" + name, torch.empty(0, dtype=torch.uint8))
    raw = torch.randn(64, 192, dtype=torch.bfloat16)
    layer.zipserv_stock_gemm = lambda layer, x, w, b: torch.nn.functional.linear(x, w, b)
    monkeypatch.setattr(module, "decompress", lambda packed, **kwargs: raw)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    settings = DispatchSettings(mode="tokens", partial_dtype="fp32")
    method = module.ZipServLinearMethod(dispatch=settings)
    seen = []

    def fused(x, buffers, *metadata):
        seen.append(metadata[-2:])
        return torch.nn.functional.linear(x, raw)

    monkeypatch.setattr(module, "_linear", fused)
    bias = torch.randn(64, dtype=torch.bfloat16)
    for tokens in (1, 129, 769):
        x = torch.randn(tokens, 192, dtype=torch.bfloat16)
        actual = method.apply(layer, x, bias)
        expected = (torch.nn.functional.linear(x, raw) + bias if tokens <= 768
                    else torch.nn.functional.linear(x, raw, bias))
        assert torch.equal(actual, expected)
    assert seen == [(3, True), (1, True)]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="excluded from CUDA graphs"):
        method.apply(layer, x)
