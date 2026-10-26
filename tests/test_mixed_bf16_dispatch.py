"""CPU checks for head dispatch and bounded prefill fallback in the mixed adapter."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
from zipserv_vllm import mixed_backend as backend
from zipserv_vllm.mixed_plugin import Settings, settings


def method(monkeypatch, *, head=False, threshold=128, logical_n=65):
    # A padded vocabulary shard verifies cropping and bias on both execution paths.
    weight = torch.arange(128 * 512, dtype=torch.float32).reshape(128, 512).remainder(8) / 8
    packed = SimpleNamespace(m=128, k=64 * 8)
    calls = []

    def fused(x, candidate, *, split_k):
        assert candidate is packed
        calls.append(("fused", split_k))
        return torch.nn.functional.linear(x, weight)

    def decode(candidate, *, max_bytes):
        assert candidate is packed
        calls.append(("decode", max_bytes))
        return weight

    def stock(layer, x, decoded, bias):
        calls.append(("stock", tuple(decoded.shape)))
        return torch.nn.functional.linear(x, decoded, bias)

    monkeypatch.setattr(backend, "linear", fused)
    monkeypatch.setattr(backend, "decompress", decode)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    implementation = backend.BF16Method(
        packed, stock, logical_n, is_head=head, prefill_tokens=threshold,
        decode_workspace_bytes=123456)
    return implementation, weight[:logical_n], calls


@pytest.mark.parametrize("tokens", [1, 8, 32, 33, 128])
def test_wide_head_always_uses_split_one_and_preserves_crop_bias(monkeypatch, tokens):
    implementation, weight, calls = method(monkeypatch, head=True)
    x = torch.ones(tokens, 512)
    bias = torch.arange(weight.shape[0], dtype=torch.float32)
    actual = implementation.apply(None, x, bias)
    torch.testing.assert_close(actual, torch.nn.functional.linear(x, weight, bias))
    assert calls == [("fused", 1)]


@pytest.mark.parametrize("tokens,split", [(8, 8), (32, 8), (33, 4), (128, 4)])
def test_projection_token_policy_is_unchanged(monkeypatch, tokens, split):
    implementation, _, calls = method(monkeypatch)
    implementation.apply(None, torch.ones(tokens, 512))
    assert calls == [("fused", split)]


def test_prefill_decodes_once_with_budget_and_stock_bias(monkeypatch):
    implementation, weight, calls = method(monkeypatch, head=True)
    x = torch.ones(2, 65, 512)
    bias = torch.ones(weight.shape[0])
    actual = implementation.apply(None, x, bias)
    torch.testing.assert_close(actual, torch.nn.functional.linear(x, weight, bias))
    assert calls == [("decode", 123456), ("stock", (65, 512))]
    assert implementation.calls == {"decode_then_stock": 1}


def test_prefill_never_allocates_a_decoded_weight_during_graph_capture(monkeypatch):
    implementation, _, calls = method(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="decode-only"):
        implementation.apply(None, torch.ones(129, 512))
    assert calls == []


def test_prefill_controls_have_explicit_defaults_and_validate_bounds():
    default = settings({"ZIPSERV_MIXED_BF16": "off"})
    assert default.bf16_prefill_tokens == 768
    assert default.bf16_decode_workspace_bytes == 2**30
    selected = settings({"ZIPSERV_MIXED_BF16": "off",
                         "ZIPSERV_MIXED_BF16_PREFILL_TOKENS": "128",
                         "ZIPSERV_MIXED_BF16_DECODE_WORKSPACE_BYTES": "0"})
    assert selected.bf16_prefill_tokens == 128
    assert selected.bf16_decode_workspace_bytes == 0
    for key, value in (("ZIPSERV_MIXED_BF16_PREFILL_TOKENS", "0"),
                       ("ZIPSERV_MIXED_BF16_PREFILL_TOKENS", "1.5"),
                       ("ZIPSERV_MIXED_BF16_DECODE_WORKSPACE_BYTES", "-1")):
        with pytest.raises(ValueError):
            settings({"ZIPSERV_MIXED_BF16": "off", key: value})


def test_preparation_recognizes_both_unquantized_head_methods(monkeypatch):
    # Exercise the real post-load selector. CPU tensors must stay dense, but both
    # stock head method types must appear in the audited candidate inventory.
    class LinearBase(torch.nn.Module):
        pass

    class GateLinear(LinearBase):
        pass

    class ParallelLMHead(torch.nn.Module):
        pass

    class UnquantizedLinearMethod:
        pass

    class UnquantizedEmbeddingMethod:
        pass

    def module(name, **values):
        item = ModuleType(name)
        item.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, item)

    module("vllm", envs=SimpleNamespace(VLLM_BATCH_INVARIANT=False))
    module("vllm.model_executor.kernels.linear.mxfp8.marlin", MarlinMxfp8LinearKernel=type("FP8", (), {}))
    module("vllm.model_executor.layers.fused_moe.router.gate_linear", GateLinear=GateLinear)
    module("vllm.model_executor.layers.linear", LinearBase=LinearBase,
           UnquantizedLinearMethod=UnquantizedLinearMethod, dispatch_unquantized_gemm=lambda: None)
    module("vllm.model_executor.layers.vocab_parallel_embedding", ParallelLMHead=ParallelLMHead,
           UnquantizedEmbeddingMethod=UnquantizedEmbeddingMethod)
    model = torch.nn.Module()
    for name, cls, method_cls in (
        ("nvidia_head", ParallelLMHead, UnquantizedLinearMethod),
        ("ct_head", ParallelLMHead, UnquantizedEmbeddingMethod),
        ("projection", LinearBase, UnquantizedLinearMethod),
        ("embedding", torch.nn.Module, UnquantizedEmbeddingMethod),
    ):
        layer = cls()
        layer.weight = torch.nn.Parameter(torch.ones(64, 64, dtype=torch.bfloat16), requires_grad=False)
        layer.quant_method = method_cls()
        model.add_module(name, layer)
    config = SimpleNamespace(
        offload_config=SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=0),
                                       prefetch=SimpleNamespace(offload_group_size=0)),
        compilation_config=SimpleNamespace(mode=0, cudagraph_mode=SimpleNamespace(name="NONE")),
        lora_config=None, weight_transfer_config=None,
        model_config=SimpleNamespace(enable_sleep_mode=False, dtype=torch.bfloat16))
    worker = SimpleNamespace(vllm_config=config, model_runner=SimpleNamespace(
        get_model=lambda: model, model_memory_usage=0))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    selected = Settings("on", "off", ("*",), (), 1, 128, 2**30)
    backend._prepare(worker, selected)
    rows = worker._zipserv_mixed["rows"]
    assert {row["name"] for row in rows} == {"nvidia_head", "ct_head", "projection"}
    assert all(not row["compressed"] and row["fallback"] == "unsupported source layout" for row in rows)
    config.compilation_config = SimpleNamespace(
        mode=0, cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"), max_cudagraph_capture_size=256)
    with pytest.raises(ValueError, match="every decode graph"):
        backend._prepare(worker, selected)
