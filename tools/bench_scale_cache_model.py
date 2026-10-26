"""Measure real-model scale caching through an isolated eager vLLM server."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request

PROMPTS = ["Explain why the sky appears blue in two short sentences.",
           "Continue this sequence of square numbers and explain its rule: 1, 4, 9, 16, 25."]


def request(base, route, body, timeout=300):
    """Send a JSON request to the explicitly selected private benchmark engine."""
    req = urllib.request.Request(base.rstrip('/') + route, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def rpc(base, method, args, workers):
    """Require every expected TP worker to acknowledge the same cache operation."""
    data = request(base, '/collective_rpc', {'method': method, 'args': args,
                                           'kwargs': {}, 'timeout': 300}, timeout=360)
    results = data.get('results', [])
    if len(results) != workers:
        raise RuntimeError('Missing worker acknowledgement')
    return results


def completion(base, model, prompt, tokens):
    """Measure streamed arrivals and retain actual token IDs for matched-output checks."""
    body = {'model': model, 'prompt': prompt, 'temperature': 0, 'seed': 174,
            'max_tokens': tokens, 'min_tokens': tokens, 'ignore_eos': True,
            'stream': True, 'stream_options': {'include_usage': True}, 'return_token_ids': True}
    req = urllib.request.Request(base.rstrip('/') + '/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    ids, arrivals, chunks, usage, done = [], [], [], None, False
    with urllib.request.urlopen(req, timeout=300) as response:
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
    if (not done or len(ids) != tokens or not arrivals or usage is None
            or usage.get('completion_tokens') != tokens):
        raise RuntimeError('Missing completion, token IDs, or usage; do not infer timing from partial data')
    decode_seconds = arrivals[-1]['seconds'] - arrivals[0]['seconds']
    remaining = tokens - arrivals[0]['tokens']
    return {'token_ids': ids, 'text': ''.join(chunks), 'elapsed_seconds': elapsed, 'ttft_seconds': arrivals[0]['seconds'],
            'decode_tokens_per_second': remaining / decode_seconds if decode_seconds and remaining else None,
            'mean_intertoken_seconds': decode_seconds / remaining if remaining else None,
            'arrivals': arrivals, 'usage': usage}


def summarize(result):
    """Compare matched requests and expose the actual registered-scale memory trade-off."""
    phases = result['phases']
    output = {'memory_scope': 'Registered compressed scale groups only; excludes FP4 codes, raw fallback scales, KV and other model state.',
              'baseline': 'Same compressed model with decoded scale caching disabled.',
              'workers_by_phase': []}
    for phase in phases:
        workers = []
        for rank, worker in enumerate(phase['final_workers']):
            if worker.get('representation') == 'raw':
                workers.append({'rank': rank, 'representation': 'raw',
                                **{k: worker[k] for k in ('raw_scale_bytes', 'cuda_allocated_bytes', 'cuda_reserved_bytes')}})
            for device, state in worker['cache'].items():
                baseline = state['uncached_native_scale_bytes']
                workers.append({'rank': rank, 'device': device,
                                **{key: state[key] for key in ('hit_rate_pct', 'hits', 'misses',
                                   'compressed_bytes', 'decoded_cache_bytes', 'decode_workspace_bytes',
                                   'retained_payload_bytes', 'uncached_native_scale_bytes',
                                   'cuda_allocated_bytes', 'cuda_reserved_bytes')},
                                'registered_scale_saving_bytes': baseline - state['retained_payload_bytes'],
                                'registered_scale_saving_pct': 100 * (1 - state['retained_payload_bytes'] / baseline)})
        output['workers_by_phase'].append({'phase': phase['number'], 'enabled': phase['enabled'], 'workers': workers})
    if result.get('representation') == 'raw':
        output['baseline'] = 'Stock vLLM GEMM and E8M0 scales; original CT FP4 quantization retained.'
        output['memory_scope'] = 'All prepared raw MoE scales; no scale dictionaries, decode scratch or decoded cache.'
    if len(phases) == 4 and [p['enabled'] for p in phases] == [False, True, True, False]:
        paired = []
        for off, on in ((phases[0], phases[1]), (phases[3], phases[2])):
            for a, b in zip(off['requests'], on['requests'], strict=True):
                if (a['prompt_index'], a['repeat']) != (b['prompt_index'], b['repeat']):
                    raise RuntimeError('Comparison requests are not matched')
                paired.append({'tokens_equal': a['token_ids'] == b['token_ids'],
                               'elapsed_reduction_pct': 100 * (1 - b['elapsed_seconds'] / a['elapsed_seconds']),
                               'ttft_reduction_pct': 100 * (1 - b['ttft_seconds'] / a['ttft_seconds']),
                               'decode_throughput_gain_pct': 100 * (b['decode_tokens_per_second'] / a['decode_tokens_per_second'] - 1)})
        output['matched_requests'] = len(paired)
        metrics = ('elapsed_reduction_pct', 'ttft_reduction_pct', 'decode_throughput_gain_pct')
        output['paired_median'] = {key: statistics.median(p[key] for p in paired) for key in metrics}
        output['paired_range'] = {key: [min(p[key] for p in paired), max(p[key] for p in paired)] for key in metrics}
        exact = [p for p in paired if p['tokens_equal']]
        output['pairs_with_equal_tokens'] = len(exact)
        output['equal_token_pairs_median'] = {key: statistics.median(p[key] for p in exact) for key in metrics} if exact else None
        output['timing_note'] = 'Positive values mean improvement with caching. All pairs use the same prompts and output length; token equality is reported separately. Sample ranges are not confidence bounds; compare against a native-scale baseline separately.'
    return output


def main():
    """Run fixed-workload off/on/on/off trials, recording cold and warmed cache state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:18794')
    parser.add_argument('--model', default='zipserv-scale-cache-test')
    parser.add_argument('--cache', choices=('compare', 'on', 'off'), default='compare')
    parser.add_argument('--representation', choices=('dictionary', 'raw'), default='dictionary')
    parser.add_argument('--cache-mib', type=int, default=256)
    parser.add_argument('--cache-policy', choices=('retain', 'lru'), default='retain')
    parser.add_argument('--prompt-format', choices=('chat', 'text'), default='chat')
    parser.add_argument('--output-check', choices=('strict', 'record'), default='strict',
                        help='record keeps diagnostic timings after output variation; it never qualifies output equality')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--tokens', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.representation == 'raw' and args.cache != 'off':
        parser.error('Use --cache off for the raw-scale control')
    if (args.report.exists() or not 0 <= args.cache_mib <= 65536 or args.workers < 1
            or not 2 <= args.tokens <= 1024 or not 1 <= args.repeats <= 100):
        parser.error('Use a fresh report, valid cache budget, and bounded positive workload')
    result = {'status': 'running', 'scope': 'full_model_eager', 'model': args.model,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'cache_budget_mib_per_worker': args.cache_mib, 'policy': args.cache_policy,
              'worker_count': args.workers, 'phases': [], 'prompts': PROMPTS,
              'prompt_format': args.prompt_format,
              'representation': args.representation,
              'output_check': args.output_check,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits': ['Whole scale-tensor cache, not a routed-expert or attention KV cache.',
                         'Same loaded model and compressed backing in every arm; eager execution only.',
                         'Loopback streaming arrivals include API scheduling; this is not isolated GEMM timing.',
                         'Fixed short prompts and sequential requests do not qualify other workloads.',
                         'Output-token equality on these prompts is not a broad model-quality evaluation.']}
    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix('.incomplete')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.report)
    save()
    try:
        result['initial_workers'] = rpc(args.base, 'zipserv_scale_cache_report', [], args.workers)
        if any(not r['installed'] or r.get('representation', 'dictionary') != args.representation
               or not (r['cached_source_groups'] if args.representation == 'dictionary' else r['raw_sources'])
               for r in result['initial_workers']):
            raise RuntimeError('The real model did not register cache sources')
        result['source_checks'] = rpc(args.base, 'zipserv_scale_cache_verify', [], args.workers)
        if any(r.get('status') != 'passed' for r in result['source_checks']):
            raise RuntimeError('Real-model GPU source checks failed')
        save()
        prompts = PROMPTS
        if args.prompt_format == 'chat':
            rendered = [request(args.base, '/tokenize', {'model': args.model,
                        'messages': [{'role': 'user', 'content': p}],
                        'add_generation_prompt': True}) for p in PROMPTS]
            prompts = [r['tokens'] for r in rendered]
            if any(not p or not all(type(t) is int and t >= 0 for t in p) for p in prompts):
                raise RuntimeError('The server did not render valid chat-template token IDs')
            result['rendered_prompts'] = rendered
        # Warm shared JIT/attention/GEMM paths before comparison, including the exact prompts.
        if args.representation == 'dictionary':
            rpc(args.base, 'zipserv_scale_cache_configure', [False, args.cache_mib, args.cache_policy], args.workers)
        result['warmup_requests'] = [completion(args.base, args.model, p, args.tokens) for p in prompts]
        references = [r['token_ids'] for r in result['warmup_requests']]
        save()
        states = [False, True, True, False] if args.cache == 'compare' else [args.cache == 'on']
        for number, enabled in enumerate(states):
            phase = {'number': number, 'enabled': enabled, 'requests': []}
            result['phases'].append(phase)
            phase['cold_state'] = (rpc(args.base, 'zipserv_scale_cache_configure',
                                      [enabled, args.cache_mib, args.cache_policy], args.workers)
                                   if args.representation == 'dictionary' else
                                   rpc(args.base, 'zipserv_scale_cache_report', [], args.workers))
            phase['cold_request'] = completion(args.base, args.model, prompts[0], args.tokens)
            phase['cold_request']['tokens_equal_reference'] = phase['cold_request']['token_ids'] == references[0]
            if not phase['cold_request']['tokens_equal_reference'] and args.output_check == 'strict':
                raise RuntimeError('Generated tokens differ from the cache-off reference; check baseline repeatability before attributing this to caching')
            phase['warm_state'] = rpc(args.base, 'zipserv_scale_cache_reset_stats', [], args.workers)
            for repeat in range(args.repeats):
                for index, prompt in enumerate(prompts):
                    row = completion(args.base, args.model, prompt, args.tokens)
                    row.update(prompt_index=index, repeat=repeat,
                               tokens_equal_reference=row['token_ids'] == references[index])
                    phase['requests'].append(row)
                    save()
                    if not row['tokens_equal_reference'] and args.output_check == 'strict':
                        raise RuntimeError('Generated tokens differ from the cache-off reference; check baseline repeatability before attributing this to caching')
            phase['final_workers'] = rpc(args.base, 'zipserv_scale_cache_report', [], args.workers)
            for worker in phase['final_workers']:
                for state in worker['cache'].values():
                    if state['enabled'] != enabled or state['lookups'] == 0:
                        raise RuntimeError('Cache flag did not control actual model execution')
                    if not enabled and (state['hits'] or state['decoded_cache_bytes']):
                        raise RuntimeError('Disabled cache retained resident decoded entries')
                    if state['decoded_cache_bytes'] > args.cache_mib * 2**20:
                        raise RuntimeError('Cache exceeded the requested memory budget')
            phase['median_elapsed_seconds'] = statistics.median(r['elapsed_seconds'] for r in phase['requests'])
            phase['median_ttft_seconds'] = statistics.median(r['ttft_seconds'] for r in phase['requests'])
            phase['median_decode_tokens_per_second'] = statistics.median(
                r['decode_tokens_per_second'] for r in phase['requests'])
            phase['source_checks_after'] = rpc(args.base, 'zipserv_scale_cache_verify', [], args.workers)
            if any(r.get('status') != 'passed' for r in phase['source_checks_after']):
                raise RuntimeError('Real-model source reconstruction failed after a measured phase')
            save()
            print(json.dumps({k: phase[k] for k in ('number', 'enabled', 'median_elapsed_seconds',
                                                    'median_ttft_seconds', 'median_decode_tokens_per_second')}), flush=True)
        result['summary'] = summarize(result)
        result['output_checks'] = [{'phase': p['number'], 'enabled': p['enabled'],
                                   'different_requests': sum(not r['tokens_equal_reference'] for r in p['requests']),
                                   'cold_tokens_equal': p['cold_request']['tokens_equal_reference']}
                                  for p in result['phases']]
        result['output_check_passed'] = all(r['different_requests'] == 0 and r['cold_tokens_equal']
                                            for r in result['output_checks'])
        result['status'] = 'passed' if result['output_check_passed'] else 'measured_with_output_variation'
        print(json.dumps({'status': result['status'], 'summary': result['summary'],
                          'output_checks': result['output_checks']}), flush=True)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
