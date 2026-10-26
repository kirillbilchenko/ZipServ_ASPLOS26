# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Paired BF16 policy/kernel experiments on real checkpoint tensors, synthetic activations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--libraries', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=21)
    parser.add_argument('--checks-only', action='store_true')
    parser.add_argument('--skip-head', action='store_true')
    args = parser.parse_args()
    if args.report.exists() or not 5 <= args.repeats <= 100:
        parser.error('Use a fresh report and 5..100 paired repetitions')
    import torch
    from safetensors import safe_open
    from zipserv_vllm import runtime
    from zipserv_vllm.bf16_policy import DispatchSettings, select_split, split_candidates
    from zipserv_vllm.vllm_plugin import ZipServLinearMethod
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    torch.set_num_threads(8)
    torch.manual_seed(10917)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    libraries = {}
    for name in ('control', 'baseline', 'popcount', 'shmem_once'):
        os.environ['ZIPSERV_LIBRARY'] = str((args.libraries / (name + '.so')).resolve())
        runtime._library.cache_clear()
        libraries[name] = runtime._library()
    # All shared libraries remain alive for graph lifetimes. Library selection
    # happens outside timing; it is a benchmark device, not runtime routing.
    active = [libraries['baseline']]
    runtime._library = lambda: active[0]
    result = {
        'status': 'running', 'scope': 'real GLM checkpoint weights; synthetic BF16 activations; one GPU',
        'device': torch.cuda.get_device_name(), 'sms': sms,
        'torch': torch.__version__, 'vllm': __import__('vllm').__version__,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'library_sha256': {name: hashlib.sha256((args.libraries/(name+'.so')).read_bytes()).hexdigest()
                           for name in libraries},
        'checks': [], 'cases': [], 'repeats': args.repeats,
        'limitations': [
            'No full-model serving, collectives, perplexity or GSM8K qualification.',
            'Checkpoint BF16 tensors may be transformed by a serving recipe; this is not additional CT coverage.',
            'Pressure is a 128 MiB write, not a guarantee of complete cache eviction.',
            'CUDA graph events are captured inside each graph. Eager timings include host launch gaps.',
            'Policy thresholds and shape scores are hypotheses; sweep results are measured separately.',
            'All arms share the same input, dense backend and native packing bytes.',
            'Dense and packed tensors coexist; payload savings are not process memory savings.',
        ],
    }

    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix('.incomplete')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.report)

    def same(a, b):
        return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

    def native_call(lib, x, packed, output, workspace, split, fp32=False):
        fn = lib.zipserv_mm_fp32 if fp32 else lib.zipserv_mm
        runtime._check(fn(torch.cuda.current_stream().cuda_stream,
            *(t.data_ptr() for t in packed.buffers), packed.max_common, packed.max_rare,
            packed.start_exp, x.data_ptr(), output.data_ptr(), packed.m, x.shape[0], packed.k,
            workspace.data_ptr(), split))

    def correctness():
        bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16).reshape(256, 256)
        all_bits = runtime.compress(bits.view(torch.bfloat16)).to(device)
        for name, lib in libraries.items():
            active[0] = lib
            assert same(runtime.decompress(all_bits).cpu(), bits.view(torch.bfloat16)), name
        result['checks'].append('all 65536 BF16 words decode exactly in all four libraries')
        # Multiple row tiles, non-power-of-two split, activation tails, and
        # guarded FP32 workspace exercise the changed epilogue/reduction ABI.
        cpu = torch.randint(-2, 3, (128, 192)).to(torch.bfloat16)
        packed = runtime.compress(cpu).to(device)
        weight = cpu.to(device)
        for name, lib in libraries.items():
            active[0] = lib
            for tokens in (0, 1, 7, 8, 9, 17, 33, 129, 257):
                x = torch.randint(-2, 3, (tokens, 192), device=device).to(torch.bfloat16)
                expected = (x.float() @ weight.float().T).to(torch.bfloat16)
                for fp32 in ((False,) if name == 'control' else (False, True)):
                    count = 3 * tokens * 128
                    dtype = torch.float32 if fp32 else torch.bfloat16
                    scratch = torch.full((count + 128,), 61.0, device=device, dtype=dtype)
                    guarded = torch.full((tokens * 128 + 128,), 61.0, device=device, dtype=torch.bfloat16)
                    output = guarded[64:-64].view(tokens, 128)
                    native_call(lib, x, packed, output, scratch[64:-64], 3, fp32)
                    torch.cuda.synchronize()
                    assert same(output, expected), (name, tokens, fp32)
                    assert bool((scratch[:64] == 61).all() and (scratch[-64:] == 61).all())
                    assert bool((guarded[:64] == 61).all() and (guarded[-64:] == 61).all())
        result['checks'].append('integer GEMM exact; output/scratch guards intact for all variants, tails and split 3')
        active[0] = libraries['baseline']
        first = runtime.compress(cpu).to(device)
        second_cpu = torch.randint(-1, 2, (64, 128)).to(torch.bfloat16)
        second = runtime.compress(second_cpu).to(device)
        x = torch.randint(-1, 2, (7, 192), device=device).to(torch.bfloat16)
        def chain():
            return runtime.linear(runtime.linear(x, first, split_k=3, partial_dtype='fp32'),
                                  second, split_k=2, partial_dtype='fp32')
        chain(); torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = chain()
        for repetition in range(6):
            x.copy_(torch.randint(-1, 2, x.shape, device=device).to(torch.bfloat16))
            graph.replay(); torch.cuda.synchronize()
            expected = ((x.float() @ weight.float().T).to(torch.bfloat16).float()
                        @ second_cpu.to(device).float().T).to(torch.bfloat16)
            assert same(output, expected)
        result['checks'].append('two-layer FP32-partial graph replays match reference with changing inputs')

    pressure = torch.zeros(128 * 2**20, dtype=torch.uint8, device=device)

    def prepare(fn, library, captured):
        active[0] = libraries[library]
        for _ in range(3):
            output = fn()
        torch.cuda.synchronize()
        if not captured:
            return {'fn': fn, 'library': library, 'output': output, 'graph': None}
        graph = torch.cuda.CUDAGraph()
        begin = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            begin.record()
            output = fn()
            end.record()
        return {'graph': graph, 'begin': begin, 'end': end, 'output': output, 'library': library}

    def once(arm):
        active[0] = libraries[arm['library']]
        if arm['graph'] is not None:
            started = time.perf_counter_ns()
            arm['graph'].replay()
            arm['end'].synchronize()
            return arm['begin'].elapsed_time(arm['end']) * 1000, (time.perf_counter_ns()-started)/1000
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter_ns()
        begin.record()
        arm['output'] = arm['fn']()
        end.record(); end.synchronize()
        return begin.elapsed_time(end) * 1000, (time.perf_counter_ns()-started)/1000

    def measure(arms, pressured, repeats):
        names = list(arms)
        values = {n: [] for n in names}; host = {n: [] for n in names}
        paired = {n: [] for n in names[1:]}
        for repetition in range(repeats):
            offset = repetition % len(names)
            order = names[offset:] + names[:offset]
            if repetition % 2:
                order.reverse()
            sample = {}
            for name in order:
                if pressured:
                    pressure.add_(1)
                us, host_us = once(arms[name])
                values[name].append(us); host[name].append(host_us); sample[name] = us
            for name in names[1:]:
                paired[name].append(100 * (sample[name] / sample[names[0]] - 1))
        return {'pressure_mib': 128 if pressured else 0, 'samples_us': values,
                'median_us': {n: statistics.median(v) for n, v in values.items()},
                'median_host_wall_us': {n: statistics.median(v) for n, v in host.items()},
                'paired_median_slowdown_percent': {n: statistics.median(v) for n, v in paired.items()},
                'paired_p10_p90_percent': {n: [sorted(v)[int(.1*(len(v)-1))], sorted(v)[int(.9*(len(v)-1))]]
                                           for n, v in paired.items()}}

    def host_attribute_check(x, packed):
        output = torch.empty((x.shape[0], packed.m), device=device, dtype=torch.bfloat16)
        empty = output.new_empty(0)
        samples = {name: [] for name in ('baseline', 'shmem_once')}
        for name in samples:
            for _ in range(3):
                native_call(libraries[name], x, packed, output, empty, 1)
        for repetition in range(args.repeats):
            for name in (('baseline', 'shmem_once') if repetition % 2 else ('shmem_once', 'baseline')):
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                for _ in range(32):
                    native_call(libraries[name], x, packed, output, empty, 1)
                samples[name].append((time.perf_counter_ns()-started)/32000)
                torch.cuda.synchronize()
        ratios = [100*(new/old-1) for old, new in zip(samples['baseline'], samples['shmem_once'], strict=True)]
        return {'scope': 'host enqueue time per direct ABI call, 32 calls per batch, scratch/output preallocated',
                'samples_us': samples, 'median_us': {n: statistics.median(v) for n, v in samples.items()},
                'paired_median_change_percent': statistics.median(ratios)}

    save()
    try:
        correctness(); save()
        if args.checks_only:
            result['status'] = 'passed'; save(); return
        index_bytes = (args.source/'model.safetensors.index.json').read_bytes()
        result['index_sha256'] = hashlib.sha256(index_bytes).hexdigest()
        index = json.loads(index_bytes)['weight_map']
        prefix = 'model.language_model.layers.3.self_attn.'
        cases = [('narrow', prefix+'kv_a_proj_with_mqa.weight', None, None),
                 ('mla_o_tp0', prefix+'o_proj.weight', None, 8192),
                 ('q_b_tp0', prefix+'q_b_proj.weight', 8192, None)]
        if not args.skip_head:
            cases.append(('head_tp0', 'lm_head.weight', 77440, None))
        for label, key, rows, columns in cases:
            with safe_open(args.source/index[key], framework='pt', device='cpu') as f:
                cpu = f.get_slice(key)[:rows, :columns].contiguous()
            assert cpu.dtype == torch.bfloat16 and cpu.ndim == 2
            active[0] = libraries['baseline']
            packed = runtime.compress(cpu).to(device)
            weight = cpu.to(device)
            assert same(runtime.decompress(packed), weight)
            dense_layer = torch.nn.Module()
            dense_layer.register_parameter('weight', torch.nn.Parameter(weight, requires_grad=False))
            stock = UnquantizedLinearMethod()
            stock.process_weights_after_loading(dense_layer)
            layer = torch.nn.Module()
            for name, buffer in zip(runtime.BUFFER_NAMES, packed.buffers, strict=True):
                layer.register_buffer('zipserv_'+name, buffer)
            layer.zipserv_metadata = (packed.m, packed.k, packed.max_common, packed.max_rare, packed.start_exp)
            layer.zipserv_compressed = True
            layer.zipserv_sm_count = sms
            layer.zipserv_stock_gemm = stock._gemm_impl
            policies = {mode: ZipServLinearMethod(dispatch=DispatchSettings(mode=mode)) for mode in ('tokens', 'shape')}
            case = {'name': label, 'source_key': key, 'shape': list(cpu.shape),
                    'slice_rows': rows, 'slice_columns': columns,
                    'source_sha256': hashlib.sha256(cpu.view(torch.uint8).numpy()).hexdigest(),
                    'raw_bytes': cpu.nbytes, 'packed_bytes': packed.storage_bytes,
                    'measurements': [], 'split_sweeps': []}
            result['cases'].append(case)
            del cpu
            for tokens in ((1, 8, 32, 128) if label == 'head_tp0' else (1, 8, 32, 128, 512, 1024, 2048, 8192)):
                x = torch.randn(tokens, packed.k, device=device, dtype=torch.bfloat16)
                captured = tokens <= 768
                split = select_split(packed.m, tokens, packed.k, sms, DispatchSettings(mode='shape'))
                factories = {
                    'dense': (lambda: stock.apply(dense_layer, x), 'baseline'),
                    'control_fixed_1': (lambda: runtime.linear(x, packed), 'control'),
                    'baseline_fixed_1': (lambda: runtime.linear(x, packed), 'baseline'),
                    'token_policy': (lambda: policies['tokens'].apply(layer, x), 'baseline'),
                    'shape_policy': (lambda: policies['shape'].apply(layer, x), 'baseline'),
                }
                if split is not None:
                    factories.update({
                        'fp32_same_split': (lambda: runtime.linear(x, packed, split_k=split, partial_dtype='fp32'), 'baseline'),
                        'popcount_same_split': (lambda: runtime.linear(x, packed, split_k=split), 'popcount'),
                        'shmem_once_same_split': (lambda: runtime.linear(x, packed, split_k=split), 'shmem_once'),
                    })
                arms = {name: prepare(fn, library, captured) for name, (fn, library) in factories.items()}
                reference = (x.float() @ weight.float().T)
                errors = {}
                for name, arm in arms.items():
                    once(arm)
                    error = arm['output'].float() - reference
                    errors[name] = {'relative_l2': float(error.norm()/reference.norm().clamp_min(1e-30)),
                                    'max_absolute': float(error.abs().max())}
                    if not torch.isfinite(error).all() or errors[name]['relative_l2'] > .02:
                        raise RuntimeError(f'{label}/{tokens}/{name}: GEMM error exceeds 2% L2')
                assert same(arms['baseline_fixed_1']['output'], arms['control_fixed_1']['output'])
                if split is None:
                    assert same(arms['token_policy']['output'], arms['dense']['output'])
                    assert same(arms['shape_policy']['output'], arms['dense']['output'])
                else:
                    assert same(arms['shape_policy']['output'], arms['popcount_same_split']['output'])
                    assert same(arms['shape_policy']['output'], arms['shmem_once_same_split']['output'])
                row = {'tokens': tokens, 'timing': 'CUDA_graph' if captured else 'eager',
                       'selected_shape_split': split,
                       'selected_token_split': select_split(packed.m, tokens, packed.k, sms, DispatchSettings(mode='tokens')),
                       'fp32_workspace_bytes': (packed.m*tokens*split*4 if split is not None and split > 1 else 0),
                       'errors_vs_fp32': errors,
                       'timings': [measure(arms, pressure_on, args.repeats) for pressure_on in (False, True)]}
                if tokens == 1:
                    row['host_attribute_check'] = host_attribute_check(x, packed)
                case['measurements'].append(row); save()
                # Sweep all legal splits under the same 64 MiB workspace budget.
                # Rank independently for warm and pressure conditions.
                if tokens in (1, 32, 128, 512):
                    for candidate in split_candidates(packed.m, tokens, packed.k, sms, 64*2**20):
                        arm = prepare(lambda s=candidate.split_k: runtime.linear(x, packed, split_k=s), 'baseline', True)
                        comparison = {'dense': arms['dense'], 'split': arm}
                        sweep = {'tokens': tokens, **asdict(candidate),
                                 'timings': [measure(comparison, pressure_on, args.repeats) for pressure_on in (False, True)]}
                        case['split_sweeps'].append(sweep)
                        del comparison, arm
                save()
                print(json.dumps({'case': label, 'tokens': tokens, 'split': split, 'status': 'passed'}), flush=True)
                del arms, factories, reference, error, x
                torch.cuda.empty_cache()
            del layer, dense_layer, stock, policies, packed, weight
            torch.cuda.empty_cache()
        result['status'] = 'passed'
        result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        save()
    except BaseException as exc:
        result['status'] = 'failed'; result['failure'] = repr(exc); save(); raise


if __name__ == '__main__':
    main()
