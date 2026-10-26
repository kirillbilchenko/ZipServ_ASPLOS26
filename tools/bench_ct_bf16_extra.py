# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Screen additional CT MLA output projections with real weights and CUDA graphs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def main():
    """Compare the actual vLLM dense method and the new ZipServ runtime on TP2 shards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    if args.report.exists() or not 5 <= args.repeats <= 100:
        parser.error('Require a fresh report and 5..100 repetitions')
    os.environ['ZIPSERV_LIBRARY'] = str(args.library.resolve())
    import torch
    from safetensors import safe_open
    from vllm.model_executor.layers.linear import (
        UnquantizedLinearMethod, dispatch_unquantized_gemm,
    )
    from zipserv_vllm.runtime import compress, decompress, linear

    config_bytes = (args.source/'config.json').read_bytes()
    index_bytes = (args.source/'model.safetensors.index.json').read_bytes()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    index_sha = hashlib.sha256(index_bytes).hexdigest()
    if (config_sha != '194d290ebb0f19aedc60d24ea9e236e33c86616e460f4a108a842f6fd5b8b198'
            or index_sha != '6d1364a27eef78675390422b1b5230601817fb7460ce816951b2cd515934c85d'):
        raise ValueError('Require the explicitly qualified CT checkpoint configuration and index')
    config = json.loads(config_bytes)
    quant = config.get('quantization_config', {})
    if quant.get('quant_method') != 'compressed-tensors' or quant.get('format') != 'mixed-precision':
        raise ValueError('Require the existing CT MXFP4 checkpoint')
    kinds = config['text_config']['layer_types']
    layers = [i for i, kind in enumerate(kinds) if kind == 'deepseek_sparse_attention']
    if not layers:
        raise ValueError('No explicitly declared MLA layers')
    index = json.loads(index_bytes)['weight_map']
    result = {'status': 'running', 'scope': 'real_CT_MLA_output_TP2_shards_on_one_GPU',
              'device': torch.cuda.get_device_name(), 'vllm': __import__('vllm').__version__,
              'torch': torch.__version__, 'cases': [], 'declared_mla_layers': layers,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
              'config_sha256': config_sha, 'index_sha256': index_sha,
              'limits': ['Real weights, synthetic BF16 activations; no full-model or quality qualification.',
                         'Both TP2 shards are tested sequentially on one GPU; collectives are excluded.',
                         'CUDA event timestamps are captured inside each graph.',
                         'The 128 MiB pressure write does not guarantee complete cache eviction.',
                         'Dense and packed weights coexist for comparison; payload counts are not process memory savings.']}
    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temp = args.report.with_suffix('.incomplete')
        temp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        temp.replace(args.report)
    def capture(fn):
        for _ in range(4):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        begin = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            begin.record()
            output = fn()
            end.record()
        return graph, begin, end, output
    save()
    try:
        torch.manual_seed(174)
        pressure = torch.zeros(128*2**20, dtype=torch.uint8, device='cuda')
        for layer in sorted({layers[0], layers[-1]}):
            source_key = f'model.language_model.layers.{layer}.self_attn.o_proj.weight'
            with safe_open(args.source/index[source_key], framework='pt', device='cpu') as f:
                shape = f.get_slice(source_key).get_shape()
            if len(shape) != 2 or shape[1] % 128 or shape[0] % 64:
                raise ValueError('Require aligned TP2 row-parallel BF16 weights')
            width = shape[1]//2
            for rank in (0, 1):
                with safe_open(args.source/index[source_key], framework='pt', device='cpu') as f:
                    cpu = f.get_slice(source_key)[:, rank*width:(rank+1)*width].contiguous()
                if cpu.dtype != torch.bfloat16:
                    raise ValueError('Only original BF16 source weights are eligible')
                started = time.perf_counter()
                packed_cpu = compress(cpu)
                packing_seconds = time.perf_counter()-started
                packed = packed_cpu.to('cuda:0')
                weight = cpu.cuda()
                restored = decompress(packed)
                if not torch.equal(restored.view(torch.int16), weight.view(torch.int16)):
                    raise RuntimeError('ZipServ changed CT source bits')
                del restored, packed_cpu
                dense_layer = torch.nn.Module()
                dense_layer.register_parameter('weight', torch.nn.Parameter(weight, requires_grad=False))
                dense_method = UnquantizedLinearMethod()
                dense_method.process_weights_after_loading(dense_layer)
                case = {'source_key': source_key, 'tp_rank': rank, 'shape': list(cpu.shape),
                        'source_sha256': hashlib.sha256(cpu.view(torch.uint8).numpy()).hexdigest(),
                        'raw_bytes': cpu.numel()*2, 'packed_bytes': packed.storage_bytes,
                        'packing_seconds': packing_seconds, 'gpu_bits_exact': True,
                        'measurements': []}
                result['cases'].append(case)
                del cpu
                for tokens in (1, 8, 32, 128, 512, 1024):
                    x = torch.randn(tokens, width, dtype=torch.bfloat16, device='cuda')
                    arm_names = ('dense', 'zipserv_split_1', 'zipserv_split_4',
                                 'zipserv_split_8', 'decode_then_dense')
                    arms = [capture(lambda: dense_method.apply(dense_layer, x)),
                            capture(lambda: linear(x, packed)),
                            capture(lambda: linear(x, packed, split_k=4)),
                            capture(lambda: linear(x, packed, split_k=8)),
                            capture(lambda: dispatch_unquantized_gemm()(
                                dense_layer, x, decompress(packed), None))]
                    snapshots = []
                    for graph, _, end, output in arms:
                        graph.replay(); end.synchronize(); snapshots.append(output.clone())
                    errors = {}
                    for arm_name, snapshot in zip(arm_names[1:], snapshots[1:], strict=True):
                        difference = snapshot.float()-snapshots[0].float()
                        relative = float(difference.norm()/snapshots[0].float().norm().clamp_min(1e-30))
                        errors[arm_name] = relative
                        if not torch.isfinite(difference).all() or relative > 0.01:
                            raise RuntimeError(f'{arm_name} GEMM component error exceeded 1%: {relative}')
                        if arm_name == 'decode_then_dense' and not torch.equal(
                                snapshot.view(torch.int16), snapshots[0].view(torch.int16)):
                            raise RuntimeError('Decoded stock GEMM must match the original stock GEMM exactly')
                    for pressured in (False, True):
                        samples = [[] for _ in arms]
                        paired = {name: [] for name in arm_names[1:]}
                        for repetition in range(args.repeats):
                            values = {}
                            order = list(range(len(arms)))
                            offset = repetition % len(arms)
                            order = order[offset:] + order[:offset]
                            if repetition % 2:
                                order.reverse()
                            for index_arm in order:
                                if pressured:
                                    pressure.add_(1)
                                graph, begin, end, _ = arms[index_arm]
                                graph.replay(); end.synchronize()
                                us = begin.elapsed_time(end)*1000
                                samples[index_arm].append(us); values[index_arm] = us
                            for i, name in enumerate(arm_names[1:], start=1):
                                paired[name].append(100*(values[i]/values[0]-1))
                        row = {'tokens': tokens, 'pressure_mib': 128 if pressured else 0,
                               'relative_l2_error': errors,
                               'samples_us': dict(zip(arm_names, samples, strict=True)),
                               'median_us': {name: statistics.median(sample)
                                             for name, sample in zip(arm_names, samples, strict=True)},
                               'paired_median_slowdown_pct': {name: statistics.median(sample)
                                                             for name, sample in paired.items()},
                               'paired_sample_p95_slowdown_pct': {
                                   name: sorted(sample)[int(.95*(len(sample)-1))]
                                   for name, sample in paired.items()}}
                        case['measurements'].append(row)
                    for snapshot, (_, _, _, output) in zip(snapshots, arms, strict=True):
                        if not torch.equal(snapshot.view(torch.int16), output.view(torch.int16)):
                            raise RuntimeError('Repeated identical GEMM inputs were not deterministic')
                    del arms, snapshots, snapshot, difference, x, output, graph, begin, end
                    if torch.cuda.max_memory_allocated() > 1024*2**20:
                        raise RuntimeError('GPU test exceeded the 1 GiB allocation budget')
                    save()
                print(json.dumps({'source': source_key, 'rank': rank,
                                  'saving_pct': 100*(1-case['packed_bytes']/case['raw_bytes']),
                                  'pressured_slowdown_pct': {r['tokens']: r['paired_median_slowdown_pct']
                                      for r in case['measurements'] if r['pressure_mib']}}), flush=True)
                del packed, weight, dense_layer, dense_method
        result.update(status='passed', cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
