# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Opt-in eager vLLM/MxWave adapter for exact dictionaries and bounded scale caching.

The separately supplied MxWave model_plugin/moe_runtime and aligned-nibble GEMM
remain the consumer. No hooks are installed unless ZIPSERV_SCALE_DICTIONARY=1.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import os
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

_INSTALLED = False
_CACHES = {}
_SOURCES = {}
_LIBRARY = None
_BACKEND = None
_REPRESENTATION = "dictionary"
_RAW_SOURCES = {}
_RAW_LAYERS = []
_DESCRIPTORS = {}
_DESCRIPTOR_STORAGES = set()


def _options():
    enabled = os.environ.get("ZIPSERV_SCALE_CACHE", "0")
    policy = os.environ.get("ZIPSERV_SCALE_CACHE_POLICY", "retain")
    mib = int(os.environ.get("ZIPSERV_SCALE_CACHE_MIB", "256"))
    if enabled not in ("0", "1") or policy not in ("retain", "lru") or not 0 <= mib <= 65536:
        raise ValueError("Require cache=0/1, retain/lru policy, and 0..65536 MiB per worker")
    return enabled == "1", mib * 2**20, policy


def register() -> None:
    """Install the explicitly enabled scale adapter in every vLLM worker."""
    global _INSTALLED, _LIBRARY, _BACKEND, _REPRESENTATION
    if os.environ.get("ZIPSERV_SCALE_DICTIONARY", "0") != "1" or _INSTALLED:
        return
    from packaging.version import Version
    if Version(version("vllm")).release != (0, 32, 0):
        raise RuntimeError("Scale caching currently targets vLLM 0.32.0")
    _REPRESENTATION = os.environ.get("ZIPSERV_SCALE_REPRESENTATION", "dictionary")
    if _REPRESENTATION not in ("dictionary", "raw"):
        raise ValueError("Scale representation must be dictionary or raw")
    if _REPRESENTATION == "dictionary" and os.environ.get("MXWAVE_MOE_MODE") != "aligned":
        raise ValueError("Scale caching requires MXWAVE_MOE_MODE=aligned and its qualified consumer")
    enabled, capacity, policy = _options()
    if _REPRESENTATION == "raw" and enabled:
        raise ValueError("The stock raw-scale control requires scale caching off")
    import torch
    backend_name = os.environ.get("ZIPSERV_SCALE_BACKEND_MODULE", "mxwave_glm53._runtime.model_plugin")
    if backend_name not in ("model_plugin", "mxwave_glm53._runtime.model_plugin"):
        raise ValueError("Select an explicitly supported aligned-nibble consumer module")
    model_plugin = importlib.import_module(backend_name)
    if backend_name == "mxwave_glm53._runtime.model_plugin":
        # Use the packaged 0.32 API contract, without enabling its separate sparse-scale profile.
        from mxwave_glm53.plugin import check_upstream
        from vllm.models.glm5next.common.model import Glm5NextForConditionalGeneration
        check_upstream()
        aliases = {"in_proj_qkvbfg_a": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"],
                   "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
                   "wk_weights_proj": ["wk", "weights_proj"]}
        mapping = dict(Glm5NextForConditionalGeneration.packed_modules_mapping)
        for name, parts in aliases.items():
            if name in mapping and mapping[name] != parts:
                raise RuntimeError("The GLM checkpoint's fused projection mapping changed")
            mapping[name] = parts
        Glm5NextForConditionalGeneration.packed_modules_mapping = mapping
    _BACKEND = model_plugin
    if _REPRESENTATION == "raw":
        # Observe the stock post-loader without replacing any scale tensor or
        # GEMM. This is a separate startup mode, not a fully populated cache.
        module = importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe."
            "compressed_tensors_moe_w4a4_mxfp4")
        method_class = module.CompressedTensorsW4A4Mxfp4MoEMethod
        original = method_class.process_weights_after_loading

        @functools.wraps(original)
        def observe_raw(method, layer):
            from vllm.config import get_current_vllm_config
            config = get_current_vllm_config()
            if not config.model_config.enforce_eager or int(config.compilation_config.mode) != 0:
                raise RuntimeError("The raw-scale comparison requires the same eager execution profile")
            original(method, layer)
            if method.use_cutlass_mxfp4:
                raise ValueError("The raw comparison requires the same Marlin MoE backend")
            label = getattr(layer, "prefix", getattr(method.moe, "layer_name", "unknown"))
            for name in ("w13_weight_scale", "w2_weight_scale"):
                scales = getattr(layer, name)
                if scales.dtype != torch.float8_e8m0fnu or scales.ndim != 3 or not scales.is_contiguous():
                    raise ValueError("Require the same prepared contiguous Marlin E8M0 scales")
                key = f"prepare-{len(_RAW_LAYERS)}:{label}/{name}"
                _RAW_SOURCES[key] = {"tensor": scales, "device": scales.device.index,
                                     "shape": list(scales.shape), "raw_bytes": scales.numel(),
                                     "raw_scale_sha256": hashlib.sha256(scales.view(torch.uint8).cpu().numpy()).hexdigest()}
            _RAW_LAYERS.append(str(label))
        method_class.process_weights_after_loading = observe_raw
        _INSTALLED = True
        return
    from .low_precision_codec import encode_container
    from .scale_cache import DecodedScaleCache, EncodedScale, load_dictionary_library
    from .scale_ownership import scale_metadata, storage_identity

    _LIBRARY = load_dictionary_library(Path(os.environ["ZIPSERV_SCALE_DICTIONARY_LIBRARY"]))
    model_plugin.install()
    original_prepare = model_plugin.prepare_layer
    original_gemm = model_plugin._GEMM_OP

    @functools.wraps(original_prepare)
    def prepare(method, layer, mode):
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        if not config.model_config.enforce_eager or int(config.compilation_config.mode) != 0:
            raise RuntimeError("Scale caching requires --enforce-eager and compilation mode 0")
        if mode != "aligned" or model_plugin.native(1).encoding != "aligned_nibbles":
            raise ValueError("Scale cache requires the unchanged aligned-nibble native consumer")
        if method.use_cutlass_mxfp4:
            raise ValueError("Scale caching requires the Marlin MoE backend")
        label = getattr(layer, "prefix", getattr(method.moe, "layer_name", "unknown"))
        entry = {"layer": str(label), "mode": "dictionary", "scales": []}
        for name in ("w13_weight_scale", "w2_weight_scale"):
            scales = getattr(layer, name)
            if scales.dtype != torch.float8_e8m0fnu or scales.ndim != 3 or not scales.is_contiguous():
                raise ValueError("Require prepared contiguous Marlin E8M0 expert scales")
            shape = tuple(scales.shape)
            raw_sha = hashlib.sha256(scales.view(torch.uint8).cpu().numpy()).hexdigest()
            record = {"name": name, "shape": list(shape), "raw_bytes": scales.numel(),
                      "retained_bytes": scales.numel(), "compressed": False}
            entry["scales"].append(record)
            try:
                packed = model_plugin.pack_scales(scales, Path(os.environ["MXWAVE_MOE_RECORD_CODEC"]),
                                                  encoding="aligned_nibbles")
            except ValueError as error:
                # Leave this tensor and its original stock quant descriptors intact.
                record["fallback"] = str(error)
                continue
            decoded = model_plugin.native(1).decode(packed)
            if not torch.equal(decoded, scales.view(torch.uint8)):
                raise RuntimeError("Native scale preparation changed source bits")
            del decoded
            descriptor = torch.nn.Parameter(packed.descriptor, requires_grad=False)
            device = scales.device
            cache = _CACHES.get(device.index)
            if cache is None:
                cache = _CACHES[device.index] = DecodedScaleCache(capacity, enabled, policy, str(device))
            # RoutedExperts in 0.32 may have no public layer label. Scope by this
            # worker's preparation ordinal as well, so identical/missing labels
            # cannot alias distinct model tensors.
            key = f"prepare-{len(model_plugin._LAYERS)}:{entry['layer']}/{name}"
            raw = packed.records.view(torch.uint8).cpu().numpy().tobytes()
            blob, accounting = encode_container(raw, "bytes", "vector4-warp", 1024)
            source = EncodedScale(_LIBRARY, blob, int(descriptor[1].item()), device, descriptor)
            cache.register(key, source, verify=True)
            references = model_plugin._replace_config_scales(method, scales, descriptor)
            if references < 1:
                raise RuntimeError("No prepared quant descriptor owned the original scale tensor")
            setattr(layer, name, descriptor)
            metadata = {"shape": shape, "kind": 1, "cache_key": key, "replacement": descriptor}
            model_plugin._STORAGE[id(descriptor)] = metadata
            _DESCRIPTORS[id(descriptor)] = metadata
            _DESCRIPTOR_STORAGES.add(storage_identity(descriptor))
            _SOURCES[key] = {"device": device.index, "shape": list(shape),
                             "raw_scale_sha256": raw_sha, "native_sha256": source.raw_sha256,
                             "container_sha256": source.container_sha256, "accounting": accounting}
            record.update(native_baseline_bytes=packed.retained_bytes, compressed=True,
                          retained_bytes=len(blob) + 16, dictionary=True, native_dictionary_gpu_exact=True,
                          quant_descriptor_references=references)
            del packed, raw, blob, scales
        model_plugin._LAYERS.append(entry)
        layer.mxwave_scale_qualification = entry
        print(f"ZipServ dictionary scales prepared: {label}", flush=True)

    def gemm(*args, **kwargs):
        scales = kwargs.get("scales", args[2] if len(args) > 2 else None)
        metadata = scale_metadata(scales, _DESCRIPTORS, _DESCRIPTOR_STORAGES)
        if metadata is None:
            return original_gemm(*args, **kwargs)
        cache = _CACHES[scales.device.index]
        with cache.borrow(metadata["cache_key"]) as descriptor:
            if descriptor is not scales:
                raise RuntimeError("The prepared consumer descriptor changed ownership")
            return original_gemm(*args, **kwargs)

    model_plugin.prepare_layer = prepare
    model_plugin._GEMM_OP = gemm
    # The supplied backend may itself use id() to choose native vs stock GEMM.
    # Reject aliases before that outer fallback, as well as at the native call.
    import inspect

    from vllm import _custom_ops as ops
    original_outer = ops.moe_wna16_marlin_gemm
    scale_position = list(inspect.signature(original_outer).parameters).index("b_scales")

    @functools.wraps(original_outer)
    def guarded_outer(*args, **kwargs):
        scales = kwargs.get("b_scales", args[scale_position] if len(args) > scale_position else None)
        scale_metadata(scales, _DESCRIPTORS, _DESCRIPTOR_STORAGES)
        return original_outer(*args, **kwargs)

    ops.moe_wna16_marlin_gemm = guarded_outer
    _INSTALLED = True


def report() -> dict:
    """Expose actual cache use and all retained payload owners on the current worker."""
    import torch
    return {"installed": _INSTALLED, "granularity": "whole_scale_tensor", "execution": "eager",
            "representation": _REPRESENTATION,
            "cache": {str(device): cache.report() for device, cache in _CACHES.items()},
            "sources": _SOURCES, "cached_source_groups": len(_SOURCES),
            "raw_sources": {k: {name: value for name, value in s.items() if name != "tensor"}
                            for k, s in _RAW_SOURCES.items()},
            "raw_scale_bytes": sum(s["raw_bytes"] for s in _RAW_SOURCES.values()),
            "layers_seen": len(_RAW_LAYERS) if _REPRESENTATION == "raw" else len(_BACKEND._LAYERS) if _BACKEND else 0,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "backend_source_sha256": hashlib.sha256(Path(_BACKEND.__file__).read_bytes()).hexdigest()
            if _BACKEND else None}


class ScaleCacheWorker:
    """Explicit worker RPCs for isolated, idle on/off measurements and source checks."""

    def zipserv_scale_cache_report(self) -> dict:
        """Return per-worker cache counters and memory accounting."""
        return report()

    def zipserv_scale_cache_configure(self, enabled: bool, mib: int = 256,
                                      policy: str = "retain") -> dict:
        """Toggle caching only at an idle worker boundary; invalidate previous residents."""
        if type(mib) is not int or not 0 <= mib <= 65536:
            raise ValueError("Cache budget must be an integer in 0..65536 MiB")
        if not _CACHES:
            raise RuntimeError("No real model scale groups were registered")
        for cache in _CACHES.values():
            cache.configure(enabled, mib * 2**20, policy)
        return report()

    def zipserv_scale_cache_reset_stats(self) -> dict:
        """Reset counters without changing warm resident data."""
        for cache in _CACHES.values():
            cache.reset_stats()
        return report()

    def zipserv_scale_cache_verify(self) -> dict:
        """Reconstruct every actual source group through the cache and native nibble decoder."""
        if _REPRESENTATION == "raw":
            import torch
            if not _RAW_SOURCES:
                raise RuntimeError("No stock raw model scales were registered")
            for key, source in _RAW_SOURCES.items():
                value = source["tensor"].view(torch.uint8).cpu().numpy()
                if hashlib.sha256(value).hexdigest() != source["raw_scale_sha256"]:
                    raise RuntimeError(f"Stock raw source bits changed: {key}")
            return {"status": "passed", "groups_exact": len(_RAW_SOURCES), "keys": list(_RAW_SOURCES)}
        if not _CACHES or not _SOURCES:
            raise RuntimeError("No prepared cache sources to verify")
        checks = []
        for key, source in _SOURCES.items():
            cache = _CACHES[source["device"]]
            with cache.borrow(key) as descriptor:
                value = SimpleNamespace(shape=tuple(source["shape"]), descriptor=descriptor,
                                        encoding="aligned_nibbles")
                decoded = _BACKEND.native(1).decode(value).cpu().numpy()
                if hashlib.sha256(decoded).hexdigest() != source["raw_scale_sha256"]:
                    raise RuntimeError(f"Cached model source bits changed: {key}")
            checks.append(key)
        return {"status": "passed", "groups_exact": len(checks), "keys": checks}
