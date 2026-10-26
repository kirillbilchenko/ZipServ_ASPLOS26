# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
# Adapted from MxWave; modified for this checkout. See NOTICE.
"""vLLM 0.32 custom quantization interface for losslessly compressed BF16 linears."""

from __future__ import annotations

import logging
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

from .bf16_policy import (
    DispatchSettings,
    decode_budget_rejection,
    select_split,
    startup_rejection,
)
from .runtime import (
    BUFFER_NAMES,
    PackedWeight,
    _linear,
    compress,
    decompress,
    require_fp32_partials,
)

logger = logging.getLogger(__name__)
DEFAULT_DISPATCH = DispatchSettings()


@register_quantization_config("zipserv")
class ZipServConfig(QuantizationConfig):
    """Online lossless packing from a raw BF16 checkpoint, compatible with ZipServ's name."""

    def __init__(self, split_k: int = 1, ignore: tuple[str, ...] = (),
                 dispatch: DispatchSettings = DEFAULT_DISPATCH) -> None:
        super().__init__()
        if isinstance(split_k, bool) or not isinstance(split_k, int) or split_k < 1:
            raise ValueError("split_k must be positive")
        self.split_k = split_k
        self.ignore = ignore
        self.dispatch = dispatch

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
        return cls(config.get("split_k", 1), tuple(ignore), DispatchSettings(
            mode=config.get("dispatch", "fixed"),
            large_batch_tokens=config.get("large_batch_tokens", 768),
            workspace_bytes=config.get("workspace_bytes", 64 * 2**20),
            partial_dtype=config.get("partial_dtype", "bf16"),
            decode_workspace_bytes=config.get("decode_workspace_bytes", 256 * 2**20),
        ))

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        """Compress dense linears; leave embeddings, heads, attention, and MoE unchanged."""
        if isinstance(layer, LinearBase):
            from vllm.model_executor.layers.fused_moe.router.gate_linear import (
                GateLinear,
            )

            if isinstance(layer, GateLinear) or prefix in self.ignore:
                logger.info("ZipServ retains stock linear dispatch for %s", prefix)
                return UnquantizedLinearMethod()
            return ZipServLinearMethod(self.split_k, self.dispatch)
        return None


@register_weight_loader_v2_supported_method
class ZipServLinearMethod(LinearMethodBase):
    """Load TP-sharded BF16 weights on CPU and replace eligible weights with CUDA byte buffers."""

    requires_device_loading = False

    def __init__(self, split_k: int = 1,
                 dispatch: DispatchSettings = DEFAULT_DISPATCH) -> None:
        self.split_k = split_k
        self.dispatch = dispatch

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
        from vllm.config import get_current_vllm_config

        config = get_current_vllm_config()
        max_tokens = getattr(config.scheduler_config, "max_num_batched_tokens", None)
        compilation = getattr(config, "compilation_config", None)
        # Graph padding can exceed the scheduler budget. Validate both at load.
        graph_max = getattr(compilation, "max_cudagraph_capture_size", None)
        if type(max_tokens) is int and type(graph_max) is int:
            max_tokens = max(max_tokens, graph_max)
        settings = self.dispatch
        if settings.mode != "fixed":
            graph_mode = getattr(getattr(compilation, "cudagraph_mode", None), "name", None)
            if (compilation is None or int(compilation.mode) != 0
                    or graph_mode not in ("NONE", "FULL_DECODE_ONLY")):
                raise ValueError("Experimental BF16 dispatch requires compilation mode 0 and eager or full-decode-only graphs")
            # Larger requests decode the weight and use stock GEMM. Split 1
            # must be valid for every fused token count; each higher split is
            # checked separately by the planner before any scratch allocation.
            fused_max = min(max_tokens, settings.large_batch_tokens) if type(max_tokens) is int else None
            reason = startup_rejection(m, k, 1, fused_max)
            reason = reason or decode_budget_rejection(m, k, settings)
            if reason is None and graph_mode == "FULL_DECODE_ONLY" and (
                type(graph_max) is not int or graph_max < 1
                or graph_max > settings.large_batch_tokens
            ):
                reason = "Decode graph capture maximum must be positive and within the fused token threshold"
            if reason is None:
                from vllm.model_executor.layers.linear import dispatch_unquantized_gemm

                layer.zipserv_stock_gemm = dispatch_unquantized_gemm()
                layer.zipserv_sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        else:
            reason = startup_rejection(m, k, self.split_k, max_tokens)
        layer.zipserv_max_tokens = max_tokens
        layer.zipserv_fallback_reason = reason
        if reason is not None:
            logger.warning(
                "ZipServ keeps dense weight %s (max_planned_tokens=%s, Split-K=%s): %s",
                (m, k), max_tokens, self.split_k, reason,
            )
            layer.weight.data = weight.to(device)
            layer.zipserv_compressed = False
            return
        if settings.partial_dtype == "fp32":
            require_fp32_partials()
        packed = compress(weight.cpu())
        if packed.storage_bytes >= weight.numel() * weight.element_size():
            layer.zipserv_fallback_reason = "Packed weight does not shrink"
            logger.info("ZipServ keeps dense weight %s: packed weight does not shrink", (m, k))
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
        if settings.mode != "fixed":
            logger.warning(
                "ZipServ experimental %s dispatch for %s: fused through %s tokens; "
                "larger batches allocate a temporary decoded BF16 weight (%s bytes) "
                "and use stock GEMM outside graph capture. Thresholds are uncalibrated.",
                settings.mode, (m, k), settings.large_batch_tokens, m * k * 2,
            )

    def apply(self, layer: nn.Module, x: Tensor, bias: Tensor | None = None) -> Tensor:
        """Execute the opaque stream-aware CUDA operation so compilation preserves it."""
        if not layer.zipserv_compressed:
            return torch.nn.functional.linear(x, layer.weight, bias)
        buffers = [getattr(layer, "zipserv_" + name) for name in BUFFER_NAMES]
        m, k, *_ = layer.zipserv_metadata
        settings = self.dispatch
        split = self.split_k if settings.mode == "fixed" else select_split(
            m, x.numel() // k, k, layer.zipserv_sm_count, settings, self.split_k)
        if split is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Experimental BF16 large-batch fallback is excluded from CUDA graphs")
            packed = PackedWeight(tuple(buffers), *layer.zipserv_metadata)
            return layer.zipserv_stock_gemm(
                layer, x, decompress(packed, max_bytes=settings.decode_workspace_bytes), bias)
        output = _linear(x, buffers, *layer.zipserv_metadata, split,
                         settings.partial_dtype == "fp32")
        if bias is not None:
            output = output + bias
        return output
