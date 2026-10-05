"""Trusted fresh-process MLX worker. It never executes candidate Python/shell."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import resource
import time
import traceback

import numpy as np

from ..runtime import candidate, prepare
from ..system import admission, system_info, real_path
from .statistics import summary
from .workloads import WORKLOADS, WARMUPS, prompt
from .families import BASELINE_MODELS, EVOLVE_MODELS

PARTIAL_REPORT = {}
PLANS = {}


def validate_worker_scope(args):
    if args.model not in BASELINE_MODELS:
        raise ValueError('model outside the pinned baseline allowlist')
    if args.model not in EVOLVE_MODELS and not (
            args.baseline and args.mode == args.baseline_mode == 'control' and not args.memory_only):
        raise ValueError('new family is unchanged-MLX baseline-only; optimizer plans are not qualified')


def error_gate(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid/nonfinite correctness tensors')
    scaled = float(np.max(np.abs(a - b) / (0.002 + 0.005 * np.abs(b))))
    norm = float(np.linalg.norm(b.ravel()))
    delta = float(np.linalg.norm((a - b).ravel()))
    relative = delta / norm if norm else (0.0 if not delta else float('inf'))
    if scaled > 1 or relative > 0.005:
        raise ValueError(f'numerical correctness failed: scaled={scaled}, relative={relative}')
    return {'max_scaled': scaled, 'relative_l2': relative, 'atol': 0.002, 'rtol': 0.005}


def trace(model, token_ids):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    x = mx.array([token_ids], dtype=mx.int32)
    tokens, logits = [], []
    for _ in range(4):
        y = model(x, cache=cache)[:, -1, :].astype(mx.float32)
        mx.eval(y)
        tokens.append(int(mx.argmax(y[0]).item()))
        logits.append(np.array(y))
        x = mx.array([[tokens[-1]]], dtype=mx.int32)
    states = [[np.array(a.astype(mx.float32)) for a in c.state] for c in cache]
    return tokens, logits, states, [c.offset for c in cache]


def correctness(model, ids, mode):
    from mlx.utils import tree_flatten
    before = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
    baseline = trace(model, ids)
    with candidate(model, mode, prepared=PLANS.get(mode)) as stats:
        during = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
        measured = trace(model, ids)
    if before != during or baseline[0] != measured[0] or baseline[3] != measured[3]:
        raise ValueError('parameter bytes, greedy tokens or KV offsets diverged')
    if mode != 'control' and stats['candidate_calls'] <= 0:
        raise ValueError('candidate used only fallback')
    errors = [error_gate(a, b) for a, b in zip(measured[1], baseline[1])]
    if len(measured[2]) != len(baseline[2]):
        raise ValueError('KV structure diverged')
    for a_layer, b_layer in zip(measured[2], baseline[2]):
        if len(a_layer) != len(b_layer):
            raise ValueError('KV structure diverged')
        errors.extend(error_gate(a, b) for a, b in zip(a_layer, b_layer))
    return {'result': 'PASS', 'parameter_bytes': before, 'exact_greedy_parity': True,
            'kv_offset_parity': True, 'logit_kv_errors': errors, 'candidate_graph_calls': dict(stats)}


def measured_request(model, tokenizer, workload, mode):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    ids = prompt(tokenizer, workload)
    mx.synchronize()
    mx.reset_peak_memory()
    with candidate(model, mode, prepared=PLANS.get(mode)) as stats:
        started = time.perf_counter()
        arrivals, tokens = [], []
        for token, _ in generate_step(mx.array(ids, dtype=mx.int32), model,
                                      max_tokens=workload.output_tokens, prefill_step_size=512):
            tokens.append(int(token))
            arrivals.append(time.perf_counter())
        mx.synchronize()
        ended = time.perf_counter()
    if len(tokens) != workload.output_tokens or (mode != 'control' and not stats['candidate_calls']):
        raise ValueError('fixed-work generation or candidate route did not complete')
    return {'tokens': tokens, 'arrivals_ms': [(t - started) * 1000 for t in arrivals],
            'ttft_ms': (arrivals[0] - started) * 1000, 'wall_ms': (ended - started) * 1000,
            'decode_tps': (len(tokens) - 1) / (arrivals[-1] - arrivals[0]),
            'goodput_tps': len(tokens) / (ended - started),
            'prompt_over_ttft_tps': len(ids) / (arrivals[0] - started),
            'peak_mlx_bytes': mx.get_peak_memory(), 'candidate_graph_calls': dict(stats),
            'timing_boundary': 'MLX-LM generate_step through final synchronization; no token-to-text/network',
            'prefill_boundary': 'prompt/TTFT is a proxy, not pure prefill compute'}


def profile(model, tokenizer):
    """Representative-layer wall probes, not an additive whole-model attribution."""
    import mlx.core as mx
    layer = model.model.layers[0]
    original_mlp = getattr(layer.mlp, 'original', layer.mlp)
    info = []
    for rows in (1, 64):
        ids = prompt(tokenizer, WORKLOADS[0])[:rows]
        x = model.model.embed_tokens(mx.array([ids], dtype=mx.int32))
        x = layer.input_layernorm(x)
        mx.eval(x)
        operations = {'mlp': layer.mlp, 'input_norm': layer.input_layernorm,
                      'gate_projection': original_mlp.gate_proj, 'up_projection': original_mlp.up_proj}
        for name, operation in operations.items():
            def run():
                y = operation(x)
                mx.eval(y)
                mx.synchronize()
                return y
            for _ in range(3):
                run()
            samples = []
            for _ in range(10):
                begin = time.perf_counter()
                run()
                samples.append((time.perf_counter() - begin) * 1000)
            info.append({'operation': name, 'shape': list(x.shape), 'dtype': str(x.dtype),
                         'backend': 'mlx_metal', 'wall_ms': samples, 'summary': summary(samples)})
    slowest = max(info, key=lambda row: row['summary']['median'])['operation']
    return {'scope': 'isolated first-layer representative probes; not pure GPU time or whole-request attribution',
            'operations': info, 'largest_measured_operator': slowest,
            'next_search_family': 'native_mlx_mlp_compiler_plans' if slowest != 'input_norm' else None,
            'selection_rule': 'MLP is the bounded composite containing the measured gate/up operators; no universal bottleneck claim'}


def pure_phase_probe(model, tokenizer):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    results = []
    for workload in WORKLOADS:
        samples = []
        for index in range(3 + 10):
            cache = make_prompt_cache(model)
            ids = mx.array([prompt(tokenizer, workload)], dtype=mx.int32)
            mx.eval(ids)
            mx.synchronize()
            begin = time.perf_counter()
            logits = model(ids, cache=cache)[:, -1, :]
            mx.eval(logits, [c.state for c in cache])
            mx.synchronize()
            elapsed = (time.perf_counter() - begin) * 1000
            if index >= 3:
                samples.append(elapsed)
        results.append({'workload': workload.name, 'prefill_wall_ms': samples,
                        'summary': summary(samples),
                        'boundary': 'fresh-cache model call plus logits/KV evaluate and wait; not generate_step TTFT'})
    return results


def run(args):
    global PARTIAL_REPORT
    validate_worker_scope(args)  # Before admission, framework imports or model I/O.
    admission_before = admission()
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from benchmarks.recovered_mlx_baseline import artifact, ROOT as MODEL_ROOT
    from mlx_lm import load
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    hardware = system_info()
    if hardware['software']['mlx'] != '0.32.2' or hardware['software']['mlx-lm'] != '0.31.3':
        raise ValueError('this experimental policy requires its pinned MLX versions')
    local_model = real_path(MODEL_ROOT / args.model)
    provenance = real_path(local_model / '.cache/huggingface/download')
    for item in local_model.iterdir():
        if item.suffix in ('.json', '.safetensors', '.jinja', '.txt'):
            real_path(item)
            real_path(provenance / (item.name + '.metadata'))
    path, model_identity = artifact(args.model)
    config = json.loads((path / 'config.json').read_text())
    expected_type = {'Llama-3.2-1B-Instruct-4bit': 'llama',
                     'gemma-4-e2b-it-4bit': 'gemma4',
                     'DeepSeek-R1-Distill-Qwen-1.5B-4bit': 'qwen2'}.get(args.model)
    if expected_type and config.get('model_type') != expected_type:
        raise ValueError('pinned baseline architecture mismatch')
    admission()
    begin = time.perf_counter()
    model, tokenizer = load(str(path), lazy=False, tokenizer_config={'trust_remote_code': False})
    mx.eval(model.parameters())
    report = {'schema_version': 1, 'success': False, 'hardware': hardware,
              'artifact': model_identity, 'model_load_ms': (time.perf_counter() - begin) * 1000,
              'architecture': config.get('model_type'),
              'quantization': config.get('quantization'),
              'batch_size': 1, 'workloads': [vars(w) for w in WORKLOADS],
              'software': hardware['software'], 'warmups_per_route': WARMUPS,
              'reference_backend': 'unchanged_mlx_lm', 'compute_backend': 'mlx_metal',
              'load_boundary': 'load local pinned snapshot, evaluate parameters; artifact hashing excluded',
              'mode': args.mode, 'baseline_mode': args.baseline_mode, 'pairs': [], 'records': [],
              'correctness': 'PENDING', 'backend_verified': False,
              'admission_before': admission_before, 'ssd_offload': 'disabled', 'ane_dispatches': 0,
              'energy_joules': None, 'kernel_compilation_ms': None,
              'peak_process_memory_boundary': 'fresh worker lifetime high-water, not per-request peak',
              'kv_memory_bytes': None, 'temporary_buffer_bytes': None,
              'production_promotion_eligible': False}
    report['correctness_scope'] = 'baseline repeatability/logits/cache self-consistency; independent candidate parity only for non-control routes'
    report['family_optimization_qualified'] = args.model in EVOLVE_MODELS
    PARTIAL_REPORT = report
    PLANS.clear()
    for mode in {args.mode, args.baseline_mode, 'control'}:
        PLANS[mode] = prepare(model, mode)
    report['phase'] = 'COMPILE_AND_CORRECTNESS'
    ids = prompt(tokenizer, WORKLOADS[0])
    compile_begin = time.perf_counter()
    report['correctness_evidence'] = correctness(model, ids, args.mode)
    report['compile_and_correctness_ms'] = (time.perf_counter() - compile_begin) * 1000
    report['compile_result'] = 'FIRST_EVALUATION_PASS'  # Not pure compiler time.
    report['correctness'] = 'PASS'
    if args.baseline_mode != 'control':
        report['parent_correctness_evidence'] = correctness(model, ids, args.baseline_mode)
    gc.collect()
    mx.clear_cache()
    report['phase'] = 'BENCHMARK'
    if args.memory_only:
        for workload in WORKLOADS:
            admission()
            for _ in range(WARMUPS):
                measured_request(model, tokenizer, workload, args.mode)
            report['records'].append({'workload': workload.name,
                                      **measured_request(model, tokenizer, workload, args.mode)})
        report['backend_verified'] = args.mode == 'control' or all(r['candidate_graph_calls']['candidate_calls'] > 0 for r in report['records'])
        report['memory_probe_boundary'] = 'matched single-route fresh worker load/correctness/3 warmups/fixed requests'
    elif args.baseline:
        with candidate(model, args.baseline_mode, prepared=PLANS[args.baseline_mode]):
            report['profile'] = (profile(model, tokenizer) if args.model in EVOLVE_MODELS else
                                 {'scope': 'no family-specific operator attribution adapter qualified',
                                  'next_search_family': None})
            report['pure_phase_probes'] = pure_phase_probe(model, tokenizer)
        for workload in WORKLOADS:
            admission()
            for index in range(WARMUPS + 10):
                row = measured_request(model, tokenizer, workload, args.baseline_mode)
                if index >= WARMUPS:
                    report['records'].append({'workload': workload.name, **row})
        report['summary'] = {w.name: {key: summary(r[key] for r in report['records'] if r['workload'] == w.name)
                                     for key in ('ttft_ms', 'decode_tps', 'wall_ms', 'peak_mlx_bytes')}
                             for w in WORKLOADS}
        report['backend_verified'] = True
        report['sample_text'] = tokenizer.decode(report['records'][0]['tokens'])
    else:
        rng = random.Random(730 + args.replication)
        for workload in WORKLOADS:
            admission()
            # Warm each candidate/control path identically, outside measurements.
            for _ in range(WARMUPS):
                measured_request(model, tokenizer, workload, args.baseline_mode)
                measured_request(model, tokenizer, workload, args.mode)
                if args.baseline_mode != 'control':
                    measured_request(model, tokenizer, workload, 'control')
            for index in range(args.pairs):
                admission()
                order = ['baseline', 'candidate']
                if args.baseline_mode != 'control':
                    order.append('mlx_reference')
                rng.shuffle(order)
                rows = {}
                for route in order:
                    route_mode = {'baseline': args.baseline_mode, 'candidate': args.mode,
                                  'mlx_reference': 'control'}[route]
                    rows[route] = measured_request(model, tokenizer, workload, route_mode)
                if args.baseline_mode == 'control':
                    rows['mlx_reference'] = dict(rows['baseline'])
                report['pairs'].append({'workload': workload.name, 'pair': index,
                                        'replication': args.replication, 'order': order, **rows})
        report['backend_verified'] = all(r['candidate']['candidate_graph_calls']['candidate_calls'] > 0 for r in report['pairs'])
    report['admission_after'] = admission()
    report['peak_process_memory_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['success'] = True
    report['phase'] = 'COMPLETE'
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=sorted(BASELINE_MODELS), required=True)
    parser.add_argument('--mode', choices=('control', 'compile_decode', 'compile_prefill', 'compile_both'), default='control')
    parser.add_argument('--baseline-mode', choices=('control', 'compile_decode', 'compile_prefill', 'compile_both'), default='control')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--memory-only', action='store_true')
    parser.add_argument('--pairs', type=int, choices=(5, 10, 15), default=5)
    parser.add_argument('--replication', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as error:
        report = {**PARTIAL_REPORT, 'schema_version': 1, 'success': False, 'failure': type(error).__name__,
                  'reason': str(error), 'traceback': traceback.format_exc(),
                  'production_promotion_eligible': False}
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
