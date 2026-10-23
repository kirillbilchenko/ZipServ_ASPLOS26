# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Exercise actual vLLM 0.32 loaders and ZipServ execution without loading a model."""

from __future__ import annotations

import json
import tempfile

import torch
from zipserv_vllm import register
from zipserv_vllm.runtime import BUFFER_NAMES, PackedWeight, decompress

register()
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import get_quantization_config
from zipserv_vllm.vllm_plugin import ZipServConfig


def check_layers() -> None:
    """Check stock loader dispatch, dense fallback, exact weights, graphs and Dynamo."""
    torch.manual_seed(42)
    assert get_quantization_config("zipserv") is ZipServConfig
    quant = ZipServConfig()
    common = {"bias": False, "params_dtype": torch.bfloat16, "quant_config": quant}
    layers = [
        ReplicatedLinear(128, 128, **common),
        ColumnParallelLinear(128, 128, disable_tp=True, **common),
        MergedColumnParallelLinear(128, [64, 64], disable_tp=True, **common),
        QKVParallelLinear(128, 64, 2, 1, disable_tp=True, **common),
        RowParallelLinear(128, 128, disable_tp=True, **common),
        ReplicatedLinear(128, 65, **common),  # unsupported alignment falls back
    ]
    reports = []
    for layer in layers:
        expected = torch.randn_like(layer.weight)
        if isinstance(layer, MergedColumnParallelLinear):
            for shard, part in enumerate(expected.split([64, 64])):
                layer.weight.weight_loader(layer.weight, part, shard)
        elif isinstance(layer, QKVParallelLinear):
            for shard, part in zip(
                ("q", "k", "v"), expected.split([128, 64, 64]), strict=True
            ):
                layer.weight.weight_loader(layer.weight, part, shard)
        else:
            layer.weight.weight_loader(layer.weight, expected)
        layer.quant_method.process_weights_after_loading(layer)
        if layer.zipserv_compressed:
            packed = PackedWeight(
                tuple(getattr(layer, "zipserv_" + name) for name in BUFFER_NAMES),
                *layer.zipserv_metadata,
            )
            assert torch.equal(
                decompress(packed).cpu().view(torch.uint16), expected.view(torch.uint16)
            )
            assert layer.weight is None
        else:
            assert torch.equal(layer.weight.cpu(), expected)
        dense = expected.cuda()
        for n in (0, 1, 3, 8, 9, 33, 128, 129):
            x = torch.randn(n, 128, dtype=torch.bfloat16, device="cuda")
            actual = layer.quant_method.apply(layer, x)
            reference = torch.nn.functional.linear(x, dense)
            error = (
                actual.float() - reference.float()
            ).norm() / reference.float().norm().clamp_min(1e-30)
            assert error < 0.01, (type(layer).__name__, n, float(error))
        x = torch.randn(3, 128, dtype=torch.bfloat16, device="cuda")
        compiled = torch.compile(
            lambda t, current=layer: current.quant_method.apply(current, t),
            backend="eager",
            fullgraph=True,
        )
        assert torch.equal(compiled(x), layer.quant_method.apply(layer, x))
        reports.append(
            {"layer": type(layer).__name__, "compressed": layer.zipserv_compressed}
        )
    print(
        json.dumps(
            {
                "vllm": __import__("vllm").__version__,
                "layers": reports,
                "scope": "TP1 loader/kernel/Dynamo contracts; no full-model qualification",
            }
        )
    )


def main() -> None:
    """Use real single-rank distributed groups required by vLLM parameter loaders."""
    with (
        tempfile.TemporaryDirectory(prefix="zipserv-vllm-check-") as directory,
        set_current_vllm_config(VllmConfig()),
    ):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="nccl",
            distributed_init_method="file://" + directory + "/rendezvous",
        )
        initialize_model_parallel(backend="nccl")
        try:
            check_layers()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
