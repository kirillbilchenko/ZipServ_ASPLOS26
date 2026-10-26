# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, experiment-only BF16 addition to the qualified legacy CT serving recipe.

This is deliberately separate from the supported vLLM 0.32 plugin. It wraps the
existing CT loader, preserves its codecs, and targets only 11 ordinary MLA output
projections. Select off/on before startup; captured graphs cannot be hot-switched.
"""
from __future__ import annotations

import functools
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import weakref

from same_runtime_plugin import SameRuntimeWorker

CONFIG_SHA = '194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198'
INDEX_SHA = '6d1364a27eef78675390422b1b5230601817fb7460ce816951b2cd515934c85d'
VLLM_VERSION = '0.1.dev20051+g487ecf187'


def dispatch_settings():
    from zipserv_vllm.bf16_policy import DispatchSettings
    return DispatchSettings(
        mode=os.environ.get('ZIPSERV_CT_BF16_DISPATCH', 'tokens'),
        partial_dtype=os.environ.get('ZIPSERV_CT_BF16_PARTIAL_DTYPE', 'bf16'),
        large_batch_tokens=int(os.environ.get('ZIPSERV_CT_BF16_LARGE_BATCH_TOKENS', '768')),
        workspace_bytes=int(os.environ.get('ZIPSERV_CT_BF16_WORKSPACE_BYTES', str(64 * 2**20))),
        decode_workspace_bytes=int(os.environ.get('ZIPSERV_CT_BF16_DECODE_WORKSPACE_BYTES', str(256 * 2**20))),
    )


def _source():
    root = Path(os.environ.get('ZIPSERV_CT_SOURCE', '/model'))
    config = (root / 'config.json').read_bytes()
    index = (root / 'model.safetensors.index.json').read_bytes()
    if (hashlib.sha256(config).hexdigest() != CONFIG_SHA
            or hashlib.sha256(index).hexdigest() != INDEX_SHA):
        raise RuntimeError('The CT source configuration or index differs from the qualified checkpoint')
    kinds = json.loads(config)['text_config']['layer_types']
    return root, [i for i, kind in enumerate(kinds) if kind == 'deepseek_sparse_attention'], json.loads(index)['weight_map']


def _hash_tensor(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def _write(report, stage):
    root = Path(os.environ.get('ZIPSERV_CT_REPORTS', '/reports'))
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'zipserv-extra-{stage}-r{report["rank"]}.json'
    temporary = target.with_suffix('.incomplete')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)


def _method_class():
    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod, dispatch_unquantized_gemm
    from zipserv_vllm.runtime import decompress, linear
    from zipserv_vllm.bf16_policy import select_split

    class CTExtraMethod(UnquantizedLinearMethod):
        """Keep graph-compatible fused decoding and use stock GEMM for large prefills."""

        def __init__(self, packed):
            self.packed = packed
            self.settings = dispatch_settings()
            self.sms = torch.cuda.get_device_properties(packed.buffers[0].device).multi_processor_count
            self.stock_gemm = dispatch_unquantized_gemm()

        def apply(self, layer, x, bias=None):
            tokens = x.numel() // x.shape[-1]
            split = select_split(self.packed.m, tokens, self.packed.k, self.sms, self.settings)
            if split is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('This trial qualifies decode-only CUDA graphs, not captured prefills')
                return self.stock_gemm(layer, x, decompress(
                    self.packed, max_bytes=self.settings.decode_workspace_bytes), bias)
            output = linear(x, self.packed, split_k=split, partial_dtype=self.settings.partial_dtype)
            return output if bias is None else output + bias

    return CTExtraMethod


def _prepare(worker):
    import torch
    from safetensors import safe_open
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod
    from zipserv_vllm.runtime import BUFFER_NAMES, PackedWeight, compress, decompress

    enabled = os.environ['ZIPSERV_CT_EXTRA_BF16'] == 'on'
    settings = dispatch_settings()
    from zipserv_vllm.bf16_policy import decode_budget_rejection
    if enabled and decode_budget_rejection(4096, 8192, settings):
        raise RuntimeError('CT BF16 trial targets exceed the configured decoded-weight budget')
    compilation = worker.vllm_config.compilation_config
    if (int(compilation.mode) != 0 or compilation.cudagraph_mode.name != 'FULL_DECODE_ONLY'
            or compilation.max_cudagraph_capture_size > settings.large_batch_tokens):
        raise RuntimeError('Require decode-only graphs within the fused token threshold')
    rank = get_tensor_model_parallel_rank()
    root, indices, weight_map = _source()
    runner = worker.model_runner
    graph_manager = runner.cudagraph_manager
    if len(indices) != 11 or (graph_manager is not None and graph_manager.graphs):
        raise RuntimeError('Require 11 MLA targets and installation before any graph capture')
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('This legacy CT experiment is qualified only on GB10')
    method_class = _method_class()
    report = {'status': 'preparing', 'enabled': enabled, 'rank': rank,
              'config_sha256': CONFIG_SHA, 'index_sha256': INDEX_SHA,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'library_sha256': hashlib.sha256(Path(os.environ['ZIPSERV_LIBRARY']).read_bytes()).hexdigest(),
              'policy': asdict(settings),
              'cuda_allocated_before': torch.cuda.memory_allocated(), 'layers': []}
    worker._zipserv_ct_extra_report = report
    weak_weights = []
    try:
        for index in indices:
            prefix = f'language_model.model.layers.{index}.self_attn.o_proj'
            layer = runner.model.get_submodule(prefix)
            if (type(layer) is not RowParallelLinear
                    or type(layer.quant_method) is not UnquantizedLinearMethod
                    or layer.weight.dtype != torch.bfloat16
                    or tuple(layer.weight.shape) != (4096, 8192)
                    or layer.tp_size != 2 or not layer.input_is_parallel or layer.bias is not None):
                raise RuntimeError(f'{prefix}: target was transformed or has an unexpected layout')
            attention = runner.model.get_submodule(prefix.rsplit('.', 1)[0])
            # Tables contain raw pointers: installation must precede their first forward.
            if any(value for key, value in attention.__dict__.items()
                   if key.startswith('_glm_l2') and isinstance(value, dict)):
                raise RuntimeError(f'{prefix}: a prefetch table may already contain dense pointers')
            cpu = layer.weight.detach().cpu().contiguous()
            source_key = f'model.language_model.layers.{index}.self_attn.o_proj.weight'
            with safe_open(root / weight_map[source_key], framework='pt', device='cpu') as handle:
                source = handle.get_slice(source_key)[:, rank * 8192:(rank + 1) * 8192].contiguous()
            if source.dtype != torch.bfloat16 or not torch.equal(cpu.view(torch.int16), source.view(torch.int16)):
                raise RuntimeError(f'{prefix}: loaded tensor differs from the original CT shard')
            del source
            row = {'prefix': prefix, 'source_key': source_key, 'source_sha256': _hash_tensor(cpu),
                   'raw_bytes': cpu.numel() * cpu.element_size(), 'packed_bytes': None,
                   'source_bits_exact': True, 'compressed': False}
            report['layers'].append(row)
            if enabled:
                packed = compress(cpu).to(layer.weight.device)
                row['packed_bytes'] = packed.storage_bytes
                if packed.storage_bytes >= row['raw_bytes']:
                    raise RuntimeError(f'{prefix}: expected target did not shrink')
                restored = decompress(packed)
                if not torch.equal(restored.view(torch.int16), layer.weight.view(torch.int16)):
                    raise RuntimeError(f'{prefix}: GPU decode changed source bits')
                del restored
                # The existing CT prefetcher enumerates parameters. Register the packed
                # byte arrays there so it sees the live representation automatically.
                parameters = []
                for name, buffer in zip(BUFFER_NAMES, packed.buffers, strict=True):
                    parameter = torch.nn.Parameter(buffer, requires_grad=False)
                    layer.register_parameter('zipserv_extra_' + name, parameter)
                    parameters.append(parameter)
                owned = PackedWeight(tuple(parameters), packed.m, packed.k,
                                     packed.max_common, packed.max_rare, packed.start_exp)
                layer.quant_method = method_class(owned)
                weak_weights.append(weakref.ref(layer.weight))
                layer.register_parameter('weight', None)
                row['compressed'] = True
                del packed, owned, parameters, parameter, buffer
            del cpu
        torch.cuda.synchronize()
        gc.collect()
        if any(ref() is not None for ref in weak_weights):
            raise RuntimeError('An original dense parameter remains referenced after compression')
        report.update(status='prepared', cuda_allocated_after=torch.cuda.memory_allocated(),
                      raw_target_bytes=sum(row['raw_bytes'] for row in report['layers']),
                      retained_target_bytes=sum(row['packed_bytes'] if row['compressed'] else row['raw_bytes']
                                                for row in report['layers']))
        # Keep vLLM's own accounting consistent with post-load adapter allocations.
        runner.model_memory_usage += report['cuda_allocated_after'] - report['cuda_allocated_before']
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        _write(report, 'loaded')


def install():
    """Install only when explicitly enabled for an off/on legacy CT experiment."""
    mode = os.environ.get('ZIPSERV_CT_EXTRA_BF16')
    if mode is None:
        return
    if mode not in ('off', 'on'):
        raise ValueError('ZIPSERV_CT_EXTRA_BF16 must be off or on')
    import vllm
    from vllm import envs
    if vllm.__version__ != VLLM_VERSION or envs.VLLM_BATCH_INVARIANT:
        raise RuntimeError('The CT trial requires the exact qualified legacy vLLM and dense dispatch')
    if os.environ.get('MXWAVE_COMPARE_ARM') != 'B':
        raise RuntimeError('Both arms must retain CT fused scale, embedding and KDA compression')
    settings = dispatch_settings()
    if mode == 'on' and settings.partial_dtype == 'fp32':
        from zipserv_vllm.runtime import require_fp32_partials
        require_fp32_partials()
    _source()
    import same_runtime_plugin
    same_runtime_plugin.install()
    from vllm.v1.worker.gpu_worker import Worker
    if getattr(Worker.load_model, '_zipserv_ct_extra', False):
        return
    original_load = Worker.load_model

    @functools.wraps(original_load)
    def loaded(worker, *args, **kwargs):
        result = original_load(worker, *args, **kwargs)
        _prepare(worker)
        return result

    loaded._zipserv_ct_extra = True
    Worker.load_model = loaded
    original_warm = Worker.compile_or_warm_up_model

    @functools.wraps(original_warm)
    def warmed(worker, *args, **kwargs):
        result = original_warm(worker, *args, **kwargs)
        _write(CTExtraWorker.zipserv_ct_extra_report(worker), 'startup')
        return result

    Worker.compile_or_warm_up_model = warmed


class CTExtraWorker(SameRuntimeWorker):
    """Retain the original CT worker reports and add source verification for this trial."""

    def zipserv_ct_extra_report(self):
        import torch
        return {**self._zipserv_ct_extra_report,
                'current_cuda_allocated_bytes': torch.cuda.memory_allocated(),
                'current_cuda_reserved_bytes': torch.cuda.memory_reserved(),
                'ct_recipe': SameRuntimeWorker.mxwave_same_runtime_report(self)}

    def zipserv_ct_extra_verify(self):
        from zipserv_vllm.runtime import decompress
        rows = []
        for row in self._zipserv_ct_extra_report['layers']:
            layer = self.model_runner.model.get_submodule(row['prefix'])
            weight = decompress(layer.quant_method.packed) if row['compressed'] else layer.weight
            matched = _hash_tensor(weight) == row['source_sha256']
            rows.append({'prefix': row['prefix'], 'source_bits_exact': matched})
            del weight
            if not matched:
                raise RuntimeError(f'{row["prefix"]}: live weights no longer match the original CT source')
        return {'status': 'passed', 'rank': self._zipserv_ct_extra_report['rank'], 'layers': rows}
