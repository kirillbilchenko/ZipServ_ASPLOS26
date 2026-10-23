# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
# Adapted from MxWave; modified for this checkout. See NOTICE.
"""vLLM 0.32 custom quantization interface for losslessly compressed BF16 linears."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

from .runtime import BUFFER_NAMES, _linear, compress


@register_quantization_config("zipserv")
class ZipServConfig(QuantizationConfig):
    """Online lossless packing from a raw BF16 checkpoint, compatible with ZipServ's name."""

    def __init__(self, split_k: int = 1, ignore: tuple[str, ...] = ()) -> None:
        super().__init__()
        if isinstance(split_k, bool) or not isinstance(split_k, int) or split_k < 1:
            raise ValueError("split_k must be positive")
        self.split_k = split_k
        self.ignore = ignore

    def get_name(self) -> str:
        """Return the upstream ZipServ method identifier."""
        return "zipserv"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        """Accept BF16 exclusively; converting FP16 would no longer be lossless."""
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        """Require BF16 tensor cores and asynchronous copy instructions."""
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        """Use default online BF16 packing without rewriting the checkpoint."""
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ZipServConfig:
        """Reject checkpoints in other packed/quantized formats."""
        if config.get("quant_method", "zipserv") != "zipserv":
            raise ValueError(
                "ZipServ zipserv expects raw BF16 weights, not another quantization format"
            )
        if config.get("checkpoint_format", "bf16") != "bf16":
            raise ValueError("This port loads raw BF16 checkpoints only")
        ignore = config.get("ignore", [])
        if not isinstance(ignore, list) or not all(
            isinstance(name, str) for name in ignore
        ):
            raise ValueError("ignore must be a list of exact vLLM module prefixes")
        return cls(config.get("split_k", 1), tuple(ignore))

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        """Compress dense linears; leave embeddings, heads, attention, and MoE unchanged."""
        if isinstance(layer, LinearBase):
            if prefix in self.ignore:
                return UnquantizedLinearMethod()
            return ZipServLinearMethod(self.split_k)
        return None


@register_weight_loader_v2_supported_method
class ZipServLinearMethod(LinearMethodBase):
    """Load TP-sharded BF16 weights on CPU and replace eligible weights with CUDA byte buffers."""

    requires_device_loading = False

    def __init__(self, split_k: int = 1) -> None:
        self.split_k = split_k

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **attrs: Any,
    ) -> None:
        """Use the stock v2 shard loader with a CPU BF16 destination."""
        if params_dtype != torch.bfloat16:
            raise ValueError(
                "ZipServ requires BF16 checkpoint loading; select --dtype bfloat16"
            )
        original_loader = attrs.pop("weight_loader")

        def load_bf16(
            param: nn.Parameter, loaded: Tensor, *args: Any, **kwargs: Any
        ) -> None:
            if loaded.dtype != torch.bfloat16:
                raise ValueError(
                    "ZipServ requires BF16 source linear weights; implicit dtype casts are rejected"
                )
            original_loader(param, loaded, *args, **kwargs)

        weight = ModelWeightParameter(
            data=torch.empty(
                (sum(output_partition_sizes), input_size_per_partition),
                dtype=params_dtype,
                device="cpu",
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=load_bf16,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, attrs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Pack only aligned weights that shrink; release the original dense parameter."""
        weight = layer.weight.detach()
        m, k = weight.shape
        device = torch.device("cuda", torch.cuda.current_device())
        if (
            m <= 0
            or k <= 0
            or m % 64
            or k % 64
            or (k // 64) % self.split_k
            or m * k > 2**31 - 4097
            or (m // 64) * self.split_k > 65535
        ):
            layer.weight.data = weight.to(device)
            layer.zipserv_compressed = False
            return
        packed = compress(weight.cpu())
        if packed.storage_bytes >= weight.numel() * weight.element_size():
            layer.weight.data = weight.to(device)
            layer.zipserv_compressed = False
            return
        packed = packed.to(device)
        for name, buffer in zip(BUFFER_NAMES, packed.buffers, strict=True):
            layer.register_buffer("zipserv_" + name, buffer)
        layer.zipserv_metadata = (
            m,
            k,
            packed.max_common,
            packed.max_rare,
            packed.start_exp,
        )
        layer.zipserv_storage_bytes = packed.storage_bytes
        layer.zipserv_compressed = True
        layer.register_parameter("weight", None)

    def apply(self, layer: nn.Module, x: Tensor, bias: Tensor | None = None) -> Tensor:
        """Execute the opaque stream-aware CUDA operation so compilation preserves it."""
        if not layer.zipserv_compressed:
            return torch.nn.functional.linear(x, layer.weight, bias)
        buffers = [getattr(layer, "zipserv_" + name) for name in BUFFER_NAMES]
        output = _linear(x, buffers, *layer.zipserv_metadata, self.split_k)
        if bias is not None:
            output = output + bias
        return output
