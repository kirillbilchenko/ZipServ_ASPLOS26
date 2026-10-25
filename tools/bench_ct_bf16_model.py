# Copyright 2026 Kyrylo Bilchenko
# SPDX-License-Identifier: Apache-2.0
"""Measure a private CT serving arm while preserving its existing adapter stack."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.request

from bench_scale_cache_model import request, rpc

PROMPTS = [
    'Explain why the sky appears blue in two short sentences.',
    'Write a Python function that returns the first n Fibonacci numbers and explain its complexity.',
]
LONG_PROMPT = ('A warehouse records the number of boxes received and shipped each day. '
               'To calculate the closing inventory, add received boxes to the opening inventory '
               'and subtract shipped boxes. Negative inventory indicates a data problem.\n') * 64 + (
                   '\nSummarize the inventory rule and give one numerical example.')


def completion(base, prompt, salt, barrier=None):
    body = {'model': 'glm-5.3-flash', 'prompt': prompt, 'temperature': 0, 'seed': 174,
            'max_tokens': 64, 'min_tokens': 64, 'ignore_eos': True,
            'stream': True, 'stream_options': {'include_usage': True},
            'return_token_ids': True, 'add_special_tokens': False, 'cache_salt': salt}
    query = urllib.request.Request(base + '/v1/completions', data=json.dumps(body).encode(),
                                   headers={'Content-Type': 'application/json'})
    if barrier:
        barrier.wait(timeout=30)
    started = time.perf_counter()
    ids, arrivals, chunks, usage, done = [], [], [], None, False
    with urllib.request.urlopen(query, timeout=600) as response:
        for line in response:
            if line.strip() == b'data: [DONE]':
                done = True
                continue
            if not line.startswith(b'data: '):
                continue
            event = json.loads(line[6:])
            if 'error' in event:
                raise RuntimeError(str(event['error']))
            for choice in event.get('choices', []):
                chunks.append(choice.get('text') or '')
                emitted = choice.get('token_ids') or []
                if emitted:
                    arrivals.append({'seconds': time.perf_counter() - started, 'tokens': len(emitted)})
                    ids.extend(emitted)
            usage = event.get('usage') or usage
    elapsed = time.perf_counter() - started
    if not done or len(ids) != 64 or not arrivals or not usage or usage.get('completion_tokens') != 64:
        raise RuntimeError('Incomplete streaming output; refusing to infer successful timing')
    duration = arrivals[-1]['seconds'] - arrivals[0]['seconds']
    remaining = 64 - arrivals[0]['tokens']
    return {'token_ids': ids, 'text': ''.join(chunks), 'elapsed_seconds': elapsed,
            'ttft_seconds': arrivals[0]['seconds'], 'arrivals': arrivals, 'usage': usage,
            'decode_tokens_per_second': remaining / duration if remaining and duration else None,
            'cache_salt': salt, 'prompt_tokens': len(prompt)}


def validate_workers(workers, enabled):
    if len(workers) != 2 or {worker['rank'] for worker in workers} != {0, 1}:
        raise RuntimeError('Both CT tensor-parallel workers must acknowledge the trial')
    for worker in workers:
        recipe = worker['ct_recipe']
        if (worker['enabled'] != enabled or worker['status'] != 'prepared'
                or len(worker['layers']) != 11
                or any(layer['compressed'] != enabled for layer in worker['layers'])
                or recipe['arm'] != 'B' or not recipe['speculation']
                or not recipe['physical_graph_token_counts']
                or recipe['max_model_len'] != 500000
                or recipe['kv_cache_memory_bytes'] != 6 * 1024**3):
            raise RuntimeError('The exact CT plus optional BF16 recipe did not load')


def serving_metrics(base):
    """Retain speculation and prefix-cache counters without including them in timing."""
    with urllib.request.urlopen(base + '/metrics', timeout=30) as response:
        lines = response.read().decode().splitlines()
    return [line for line in lines if not line.startswith('#') and
            any(name in line for name in ('spec_decode', 'prefix_cache'))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:18795')
    parser.add_argument('--mode', required=True, choices=('off', 'on'))
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--repeats', type=int, default=6)
    args = parser.parse_args()
    if args.report.exists() or not 3 <= args.repeats <= 20:
        parser.error('Require a fresh report and 3..20 repetitions')
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if args.mode == 'on' and (not reference or reference.get('mode') != 'off'):
        parser.error('The candidate requires a measured CT baseline report')
    result = {'status': 'running', 'scope': 'full_CT_recipe_with_optional_extra_BF16',
              'mode': args.mode, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'prompt_texts': [*PROMPTS, LONG_PROMPT],
              'requests': [], 'cohorts': [],
              'limits': ['A short, controlled serving test; not broad model-quality qualification.',
                         'Cold requests use distinct cache salts while prefix caching remains enabled.',
                         'Separate server loads; one off/on pair cannot exclude drift or establish small gains.',
                         'Live CUDA allocations exclude allocator reservation, driver and NCCL memory.',
                         'Streaming decode rates exclude the first speculative token burst.']}
    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix('.incomplete')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.report)
    save()
    try:
        result['initial_workers'] = rpc(args.base, 'zipserv_ct_extra_report', [], 2)
        validate_workers(result['initial_workers'], args.mode == 'on')
        result['source_checks_before'] = rpc(args.base, 'zipserv_ct_extra_verify', [], 2)
        if any(worker.get('status') != 'passed' for worker in result['source_checks_before']):
            raise RuntimeError('GPU source verification failed')
        rendered = [request(args.base, '/tokenize', {'model': 'glm-5.3-flash',
                    'messages': [{'role': 'user', 'content': prompt}], 'add_generation_prompt': True})
                    for prompt in (*PROMPTS, LONG_PROMPT)]
        prompts = [row['tokens'] for row in rendered]
        if any(not prompt or len(prompt) > 6919 or not all(type(token) is int for token in prompt)
               for prompt in prompts):
            raise RuntimeError('Invalid or oversized chat-template prompt')
        result['rendered_prompts'] = rendered
        if reference and prompts != [row['tokens'] for row in reference['rendered_prompts']]:
            raise RuntimeError('The baseline and candidate did not render identical prompts')
        result['serving_metrics_before'] = serving_metrics(args.base)
        result['warmup_requests'] = [completion(args.base, prompt, f'ct-bf16-extra-warm-{i}')
                                     for i, prompt in enumerate(prompts)]
        result['warmup_cohorts'] = []
        for warmup in range(2):
            barrier = threading.Barrier(4)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(completion, args.base, prompts[index % 2],
                                       f'ct-bf16-extra-warm-c4-{warmup}-{index}', barrier)
                           for index in range(4)]
                result['warmup_cohorts'].append([future.result() for future in futures])
        save()
        for repeat in range(args.repeats):
            for workload, prompt_index in (('short_cold', 0), ('short_warm', 1), ('long_cold', 2)):
                salt = (f'ct-bf16-extra-warm-{prompt_index}' if workload == 'short_warm'
                        else f'ct-bf16-extra-{workload}-{repeat}')
                row = completion(args.base, prompts[prompt_index], salt)
                row.update(workload=workload, repeat=repeat, prompt_index=prompt_index)
                result['requests'].append(row)
                save()
        for repeat in range(3):
            barrier = threading.Barrier(4)
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(completion, args.base, prompts[index % 2],
                                       f'ct-bf16-extra-c4-{repeat}-{index}', barrier) for index in range(4)]
                cohort = [future.result() for future in futures]
            result['cohorts'].append({'repeat': repeat, 'elapsed_seconds': time.perf_counter() - started,
                                      'requests': cohort})
            save()
        result['source_checks_after'] = rpc(args.base, 'zipserv_ct_extra_verify', [], 2)
        result['serving_metrics_after'] = serving_metrics(args.base)
        result['final_workers'] = rpc(args.base, 'zipserv_ct_extra_report', [], 2)
        validate_workers(result['final_workers'], args.mode == 'on')
        if any(worker.get('status') != 'passed' for worker in result['source_checks_after']):
            raise RuntimeError('Post-benchmark source verification failed')
        metrics = ('elapsed_seconds', 'ttft_seconds', 'decode_tokens_per_second')
        result['summary'] = {
            workload: {metric: statistics.median(row[metric] for row in result['requests']
                                                if row['workload'] == workload) for metric in metrics}
            for workload in ('short_cold', 'short_warm', 'long_cold')}
        result['summary']['c4'] = {'aggregate_tokens_per_second': statistics.median(
            256 / cohort['elapsed_seconds'] for cohort in result['cohorts'])}
        repeatability = {workload: len({tuple(row['token_ids']) for row in result['requests']
                                       if row['workload'] == workload})
                         for workload in ('short_cold', 'short_warm', 'long_cold')}
        result['distinct_outputs_by_workload'] = repeatability
        concurrent_repeatability = {str(index): len({tuple(cohort['requests'][index]['token_ids'])
                                                    for cohort in result['cohorts']})
                                    for index in range(4)}
        result['distinct_outputs_by_concurrent_request'] = concurrent_repeatability
        matched = []
        concurrent_matched = []
        if reference:
            for raw, packed in zip(reference['requests'], result['requests'], strict=True):
                if (raw['workload'], raw['repeat'], raw['cache_salt']) != (
                        packed['workload'], packed['repeat'], packed['cache_salt']):
                    raise RuntimeError('Requests are not matched across arms')
                matched.append({'workload': raw['workload'], 'repeat': raw['repeat'],
                                'tokens_equal': raw['token_ids'] == packed['token_ids'],
                                'latency_change_pct': 100 * (packed['elapsed_seconds'] / raw['elapsed_seconds'] - 1),
                                'decode_rate_change_pct': 100 * (packed['decode_tokens_per_second'] / raw['decode_tokens_per_second'] - 1)})
            result['comparison'] = matched
            for raw_cohort, packed_cohort in zip(reference['cohorts'], result['cohorts'], strict=True):
                for index, (raw, packed) in enumerate(zip(raw_cohort['requests'], packed_cohort['requests'], strict=True)):
                    if raw['cache_salt'] != packed['cache_salt']:
                        raise RuntimeError('Concurrent requests are not matched across arms')
                    concurrent_matched.append({'repeat': packed_cohort['repeat'], 'request_index': index,
                                               'tokens_equal': raw['token_ids'] == packed['token_ids']})
            result['concurrent_comparison'] = concurrent_matched
        result['output_check_passed'] = (
            all(n == 1 for n in (*repeatability.values(), *concurrent_repeatability.values()))
            and all(row['tokens_equal'] for row in (*matched, *concurrent_matched))
            and (reference is None or reference.get('output_check_passed', False)))
        result['status'] = 'measured' if result['output_check_passed'] else 'measured_with_output_variation'
        print(json.dumps({'status': result['status'], 'mode': args.mode, 'summary': result['summary']}), flush=True)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
