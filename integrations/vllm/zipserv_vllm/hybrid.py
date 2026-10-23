# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
# Adapted from MxWave; modified for this checkout. See NOTICE.
"""Keep stock MXFP4 methods while compressing an untied BF16 output head."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

from .vllm_plugin import ZipServLinearMethod


@register_quantization_config("zipserv-mxfp4")
class ZipServHybridConfig(CompressedTensorsConfig):
    """Explicit opt-in adapter for an existing compressed-tensors MXFP4 checkpoint."""

    def get_name(self) -> str:
        """Return the independent hybrid method identifier."""
        return "zipserv-mxfp4"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        """Require BF16 for the lossless output-head execution path."""
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        """Require the BF16 tensor cores used by the external ZipServ kernel."""
        return 80

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        """Select this adapter only when explicitly requested for the right format."""
        if user_quant != "zipserv-mxfp4":
            return None
        if (
            hf_quant_cfg.get("quant_method") != "compressed-tensors"
            or hf_quant_cfg.get("format") != "mxfp4-pack-quantized"
        ):
            return None
        text_config = getattr(hf_config, "text_config", hf_config)
        if getattr(text_config, "tie_word_embeddings", False):
            raise ValueError(
                "The experimental ZipServ hybrid requires an untied BF16 output head"
            )
        return "zipserv-mxfp4"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ZipServHybridConfig:
        """Validate the checkpoint format and preserve its existing quantization settings."""
        if (
            config.get("quant_method") != "compressed-tensors"
            or config.get("format") != "mxfp4-pack-quantized"
        ):
            raise ValueError(
                "ZipServ hybrid requires compressed-tensors mxfp4-pack-quantized"
            )
        options = config.get("zipserv", {})
        if not isinstance(options, dict):
            raise TypeError("zipserv must be a configuration object")
        split = options.get("split_k", 1)
        if isinstance(split, bool) or not isinstance(split, int) or split < 1:
            raise ValueError("zipserv.split_k must be a positive integer")
        result = cast(ZipServHybridConfig, super().from_config(config))
        result.split_k = split
        return result

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        """Delegate MXFP4, embeddings and caches; replace only an unquantized output head."""
        method = super().get_quant_method(layer, prefix)
        if isinstance(layer, ParallelLMHead):
            if method is not None and not isinstance(
                method, UnquantizedEmbeddingMethod
            ):
                raise ValueError(
                    f"ZipServ cannot replace a quantized or transformed head: {prefix}"
                )
            return ZipServLMHeadMethod(self.split_k)
        return method


class ZipServLMHeadMethod(ZipServLinearMethod):
    """Use the stock vocabulary shard loader with the existing BF16 ZipServ runtime."""

    def tie_weights(self, layer: nn.Module, embed_tokens: nn.Module) -> nn.Module:
        """Reject weight tying, which would retain the dense embedding and add a duplicate."""
        raise ValueError(
            "The experimental ZipServ hybrid requires an untied BF16 output head"
        )
