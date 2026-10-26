# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Frozen 300-question comparison of existing CT against CT plus current ZipServ BF16."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import eval_ct_gsm8k as common
from eval_exact_ct_gsm8k import IMAGE, PARENT_SHA, SCORER_SHA
from bench_scale_cache_model import rpc
from bench_ct_bf16_model import validate_workers, serving_metrics

POLICY = {'mode': 'shape', 'large_batch_tokens': 768, 'workspace_bytes': 64 * 2**20,
          'partial_dtype': 'fp32', 'decode_workspace_bytes': 256 * 2**20}
PHYSICAL_GRAPHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 28, 32]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def freeze(parent_bytes):
    if sha(parent_bytes) != PARENT_SHA:
        raise ValueError('Require the hash-pinned full GSM8K parent protocol')
    parent = json.loads(parent_bytes)
    indices = sorted(random.Random(42).sample(range(1319), 300))
    protocol = {**parent, 'version': 3, 'runtime_profile': 'CT_plus_current_ZipServ_BF16_300_v1',
                'questions_per_arm': 300, 'cases': [parent['cases'][i] for i in indices],
                'selection': {'method': 'uniform sample without replacement, sorted source indices',
                              'seed': 42, 'indices': indices},
                'parent_protocol_sha256': PARENT_SHA, 'scoring_source_sha256': SCORER_SHA,
                'speculative_decoding': True, 'bf16_policy': POLICY,
                'capture_sizes': [1, 2, 4, 6, 8, 12, 16, 18, 24, 32],
                'physical_graph_token_counts': PHYSICAL_GRAPHS}
    protocol['limits'] = parent['limits'] + [
        'A preselected 300/1319 test subset; not the complete GSM8K result.',
        'Both arms retain CT fused scales, vocabulary/KDA compression, and DFlash.',
        'Only the 11 eligible BF16 MLA output projections change storage/dispatch.',
        'One run per arm cannot resolve small performance or quality differences.',
        'GSM8K throughput includes variable generated lengths; use the separate fixed-output timing test too.',
    ]
    return protocol


def validate_launch(launch, mode):
    if (launch['image'] != IMAGE or launch['rank'] != 0 or launch['mode'] != mode
            or launch.get('preflight') or launch.get('returncode') != 0):
        raise ValueError('Require a successful pinned CT rank-zero serving launch')
    command, env = launch['cmd'], launch['env']
    def option(key):
        return command[command.index(key)+1]
    graph = json.loads(option('--compilation-config'))
    if (option('--max-model-len') != '500000' or option('--kv-cache-memory-bytes') != str(6 * 2**30)
            or option('--max-num-batched-tokens') != '6919' or option('--max-num-seqs') != '32'
            or option('--kv-cache-dtype') != 'fp8_e4m3' or '--enable-prefix-caching' not in command
            or option('--worker-extension-cls') != 'ct_bf16_extra_plugin.CTExtraWorker'
            or json.loads(option('--speculative-config'))['method'] != 'dflash'
            or graph['mode'] != 0 or graph['cudagraph_mode'] != 'FULL_DECODE_ONLY'
            or graph['cudagraph_capture_sizes'] != [1, 2, 4, 6, 8, 12, 16, 18, 24, 32]
            or env.get('MXWAVE_COMPARE_ARM') != 'B' or env.get('MXWAVE_MOE_MODE') != 'aligned'
            or env.get('GLM_LV_DRAFT_FP8_KV') != '1' or env.get('ZIPSERV_CT_EXTRA_BF16') != mode):
        raise ValueError('The production-like CT runtime/weight recipe changed')


def normalized_launch(launch):
    # Docker inspect can reorder its mount list between calls. Compare unique
    # destination mappings, while retaining every source and access-mode check.
    if len({m['destination'] for m in launch['mounts']}) != len(launch['mounts']):
        raise ValueError('Duplicate mount destinations are ambiguous')
    mounts = sorted(
        ({**m, 'source': '<private>' if m['destination'] in ('/cache', '/reports') else m['source']}
         for m in launch['mounts']), key=lambda m: m['destination'])
    return {key: launch[key] for key in ('image', 'cmd', 'sources', 'tools', 'library_sha256', 'host_config')} | {
        'env': {k: v for k, v in launch['env'].items() if k != 'ZIPSERV_CT_EXTRA_BF16'}, 'mounts': mounts}


def audit(base, mode, launch):
    workers = rpc(base, 'zipserv_ct_extra_report', [], 2)
    validate_workers(workers, mode == 'on')
    for worker in workers:
        recipe = worker['ct_recipe']
        if (worker['policy'] != POLICY or worker['library_sha256'] != launch['library_sha256']
                or recipe['max_num_batched_tokens'] != 6919 or recipe['max_num_seqs'] != 32
                or recipe['draft_layers'] != 5 or not recipe['shared_head']
                or recipe['physical_graph_token_counts'] != PHYSICAL_GRAPHS
                or recipe['experts']['layers_seen'] != 42
                or recipe['vocabulary']['resident_layers'] != 1 or len(recipe['kda']['layers']) != 68):
            raise ValueError('Live worker configuration differs from the frozen CT/BF16 contract')
    verification = rpc(base, 'zipserv_ct_extra_verify', [], 2)
    if any(row['status'] != 'passed' or len(row['layers']) != 11 for row in verification):
        raise ValueError('Live BF16 source bytes failed verification')
    return {'workers': workers, 'source_verification': verification,
            'cuda_allocated_bytes_total': sum(w['current_cuda_allocated_bytes'] for w in workers),
            'cuda_reserved_bytes_total': sum(w['current_cuda_reserved_bytes'] for w in workers),
            'target_raw_bytes': sum(w['raw_target_bytes'] for w in workers),
            'target_retained_bytes': sum(w['retained_target_bytes'] for w in workers)}


def run(args):
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol != freeze(args.parent_protocol.read_bytes()):
        raise ValueError('The selected questions or request settings changed')
    sources = {name: sha(Path(__file__).with_name(name).read_bytes()) for name in
               ('eval_ct_bf16_gsm8k.py', 'eval_ct_gsm8k.py', 'eval_exact_ct_gsm8k.py',
                'bench_scale_cache_model.py', 'bench_ct_bf16_model.py')}
    if sources['eval_ct_gsm8k.py'] != SCORER_SHA:
        raise ValueError('The frozen scorer changed')
    launch = json.loads(args.server_receipt.read_text())
    validate_launch(launch, args.mode)
    args.output.mkdir()
    (args.output/'protocol.json').write_bytes(protocol_bytes)
    (args.output/'server-launch.json').write_bytes(args.server_receipt.read_bytes())
    metadata = {'arm': args.mode, 'model': 'glm-5.3-flash', 'status': 'running',
                'protocol_sha256': sha(protocol_bytes), 'client_sources': sources,
                'client_sha256': sha(json.dumps(sources, sort_keys=True).encode()),
                'started_utc': datetime.now(timezone.utc).isoformat()}
    results = []; started = None; generation_seconds = None
    def save():
        elapsed = generation_seconds if generation_seconds is not None else time.monotonic()-started if started else 0
        tokens = sum(row.get('response', {}).get('usage', {}).get('completion_tokens', 0) for row in results)
        latencies = sorted(row['elapsed_seconds'] for row in results)
        report = {**metadata, **common.summarize(results, 300), 'generation_seconds': elapsed,
                  'completion_tokens': tokens, 'aggregate_completion_tokens_per_second': tokens/elapsed if elapsed else None,
                  'median_request_seconds': statistics.median(latencies) if latencies else None,
                  'p95_request_seconds': latencies[int(.95*(len(latencies)-1))] if latencies else None}
        tmp = args.output/'summary.incomplete'; tmp.write_text(json.dumps(report, indent=2)+'\n')
        tmp.replace(args.output/'summary.json')
    save()
    try:
        before = audit(args.base, args.mode, launch)
        (args.output/'workers-before.json').write_text(json.dumps(before, indent=2)+'\n')
        metrics = {'before': serving_metrics(args.base)}
        started = time.monotonic()
        with (args.output/'responses.jsonl').open('x') as log, ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(common.execute, case, protocol, args.base, 'glm-5.3-flash')
                       for case in protocol['cases']]
            for future in as_completed(futures):
                row = future.result(); results.append(row)
                log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush(); os.fsync(log.fileno()); save()
                if len(results) % 25 == 0:print(json.dumps(common.summarize(results, 300)), flush=True)
        generation_seconds = time.monotonic()-started
        metrics['after'] = serving_metrics(args.base)
        (args.output/'serving-metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        after = audit(args.base, args.mode, launch)
        (args.output/'workers-after.json').write_text(json.dumps(after, indent=2)+'\n')
        metadata.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        metadata.update(status='failed', error=repr(exc)); raise
    finally:
        save()


def compare(control, candidate):
    runs = []
    for root, mode in ((control, 'off'), (candidate, 'on')):
        protocol = json.loads((root/'protocol.json').read_text())
        summary = json.loads((root/'summary.json').read_text())
        rows = [json.loads(line) for line in (root/'responses.jsonl').read_text().splitlines()]
        launch = json.loads((root/'server-launch.json').read_text()); validate_launch(launch, mode)
        if (summary['status'] != 'complete' or len(rows) != 300
                or {r['source_index'] for r in rows} != set(protocol['selection']['indices'])):
            raise ValueError('Require both complete, unique 300-question runs')
        runs.append((summary, {r['source_index']: r for r in rows}, launch, protocol,
                     json.loads((root/'workers-before.json').read_text())))
    (a, ar, al, ap, am), (b, br, bl, bp, bm) = runs
    if (ap != bp or a['protocol_sha256'] != b['protocol_sha256'] or a['client_sha256'] != b['client_sha256']
            or normalized_launch(al) != normalized_launch(bl)):
        raise ValueError('Unmatched questions, clients, source builds or serving configuration')
    gains = [i for i in ar if not ar[i]['answer_exact_match'] and br[i]['answer_exact_match']]
    losses = [i for i in ar if ar[i]['answer_exact_match'] and not br[i]['answer_exact_match']]
    discordant = len(gains)+len(losses)
    p = min(1., 2 * sum(math.comb(discordant, i) for i in range(min(len(gains), len(losses))+1))/2**discordant) if discordant else 1.
    return {'control': a, 'candidate': b, 'matched_runtime_contract': True,
            'comparison_source_sha256': sha(Path(__file__).read_bytes()),
            'mount_comparison': 'unique destination mappings; Docker list ordering ignored',
            'accuracy_delta_percentage_points': 100*(b['accuracy']-a['accuracy']),
            'newly_correct_indices': sorted(gains), 'newly_incorrect_indices': sorted(losses),
            'paired_mcnemar_exact_p': p,
            'target_payload_saved_bytes': am['target_retained_bytes']-bm['target_retained_bytes'],
            'cuda_allocated_saving_bytes_before_eval': am['cuda_allocated_bytes_total']-bm['cuda_allocated_bytes_total'],
            'gsm8k_aggregate_throughput_change_percent': 100*(b['aggregate_completion_tokens_per_second']/a['aggregate_completion_tokens_per_second']-1),
            'limits': ap['limits'] + ['CUDA allocated memory excludes driver/NCCL allocations and allocator reservations.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('freeze'); p.add_argument('--parent-protocol', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run')
    for name in ('parent-protocol', 'protocol', 'server-receipt', 'output'):p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--mode', choices=('off','on'), required=True); p.add_argument('--base', default='http://127.0.0.1:18795')
    p = sub.add_parser('compare'); p.add_argument('control', type=Path); p.add_argument('candidate', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        with args.output.open('x') as f:json.dump(freeze(args.parent_protocol.read_bytes()), f, indent=2); f.write('\n')
    elif args.action == 'run':run(args)
    else:print(json.dumps(compare(args.control, args.candidate), indent=2))


if __name__ == '__main__':
    main()
