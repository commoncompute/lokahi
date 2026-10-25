"""Per-chip search of the native engine's Metal dispatch thresholds (Tuning).

Coordinate descent over the fields `lokahi --tuning` accepts. Each field's
grid is screened in one fresh process per value; the best screened value is
then judged against the current tuning with the lab protocol: fresh
processes, 3 warmups, at least 10 interleaved pairs in alternating order and
an independent replication, a paired log-ratio bootstrap (bench.statistics)
and exact greedy tokens. A value is kept only when both replications' 95%
lower bounds clear the threshold on the objective and the other phase's
upper bounds show no regression past it. Every value in a field's range
computes the same results up to rounding (the engine's parity suites run the
range ends), so a kept value changes speed only. The result is hardware
evidence for this chip, model, workload and engine build, nothing wider.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from .bench.statistics import paired_comparison, summary

DEFAULTS = {'qmv_mma_min_rows': 3, 'moe_grouped_min_tokens': 16, 'gdn_chunk_min_rows': 16,
            'attn_mm_min_rows': 16, 'decode_groups_per_core': 8, 'decode_max_blocks': 64,
            'prefill_max_blocks': 8, 'latent_min_block_keys': 64, 'inflight_steps': 2}
GRID = {
    'qmv_mma_min_rows': (2, 3, 4, 6, 9),
    'moe_grouped_min_tokens': (4, 8, 16, 32, 64),
    'gdn_chunk_min_rows': (4, 8, 16, 32, 64),
    'attn_mm_min_rows': (4, 8, 16, 32, 64),
    'decode_groups_per_core': (2, 4, 8, 16, 32),
    'decode_max_blocks': (16, 32, 64, 128, 256),
    'prefill_max_blocks': (2, 4, 8, 16, 32),
    'latent_min_block_keys': (16, 32, 64, 128, 256),
    'inflight_steps': (1, 2, 3, 4),
}
METRIC = {'decode': 'decode_tokens_per_second', 'prefill': 'prompt_tokens_per_second'}
WARMUPS, PAIRS, REPLICATIONS, THRESHOLD = 3, 10, 2, 0.02


def relevant_fields(config):
    """The fields that can change this architecture's dispatch."""
    text = config.get('text_config') or config
    fields = ['qmv_mma_min_rows', 'attn_mm_min_rows', 'decode_groups_per_core', 'decode_max_blocks',
              'prefill_max_blocks', 'inflight_steps']
    if text.get('num_experts') or text.get('num_local_experts') or text.get('n_routed_experts'):
        fields.append('moe_grouped_min_tokens')
    if text.get('linear_num_value_heads') or 'linear_attention' in (text.get('layer_types') or ()):
        fields.append('gdn_chunk_min_rows')
    if text.get('kv_lora_rank'):
        fields.append('latent_min_block_keys')
    return fields


def tokens_of(run):
    return [rep['token_ids'] for rep in run['repetitions']]


def judge_value(pairs_by_replication, objective, threshold=THRESHOLD):
    """Keep or reject one candidate from its replications of paired runs.

    Each replication is a list of (control_report, candidate_report) from
    fresh processes of one repetition each.
    """
    other = 'prefill' if objective == 'decode' else 'decode'
    decisions = []
    for pairs in pairs_by_replication:
        control = [a['repetitions'][0] for a, _ in pairs]
        candidate = [b['repetitions'][0] for _, b in pairs]
        if any(a['token_ids'] != b['token_ids'] for a, b in zip(control, candidate)):
            return {'keep': False, 'reason': 'greedy tokens changed', 'replications': decisions}
        main = paired_comparison([r[METRIC[objective]] for r in control], [r[METRIC[objective]] for r in candidate],
                                 larger_is_better=True)
        side = paired_comparison([r[METRIC[other]] for r in control], [r[METRIC[other]] for r in candidate],
                                 larger_is_better=True)
        decisions.append({objective: main, other: side})
    for d in decisions:
        if d[other]['upper'] < 1 - threshold:
            return {'keep': False, 'reason': f'{other} regressed', 'replications': decisions}
    if all(d[objective]['lower'] > 1 + threshold for d in decisions):
        return {'keep': True, 'reason': f'{objective} faster in every replication', 'replications': decisions}
    return {'keep': False, 'reason': 'not faster beyond the threshold in every replication', 'replications': decisions}


def search(bench, fields, objective, start=None, *, pairs=PAIRS, replications=REPLICATIONS, threshold=THRESHOLD,
           log=lambda record: None):
    """Coordinate descent; `bench(tuning, warmups, repetitions)` runs one fresh process."""
    if pairs < 10 or replications < 2:
        raise ValueError('the lab protocol needs at least 10 pairs and 2 replications')
    current = dict(DEFAULTS, **(start or {}))
    steps = []
    def median(run):
        return summary([rep[METRIC[objective]] for rep in run['repetitions']])['median']

    for field in fields:
        # Each value is screened beside a run of the current tuning, in
        # alternating order, so drift over the field (thermal state, other
        # work) cancels in the ratio; one process per value alone swung 30%.
        screened = []
        for i, value in enumerate(v for v in GRID[field] if v != current[field]):
            trial = dict(current, **{field: value})
            first, second = (current, trial) if i % 2 == 0 else (trial, current)
            a, b = bench(first, 1, 2), bench(second, 1, 2)
            base, run = (a, b) if first is current else (b, a)
            screened.append((median(run) / median(base), value))
            log({'kind': 'screen', 'field': field, 'value': value, 'control': base, 'candidate': run})
        step = {'field': field, 'from': current[field], 'screened': [{'value': v, 'ratio': r} for r, v in screened]}
        ratio, best = max(screened)
        if ratio <= 1 + threshold / 2:
            steps.append(dict(step, keep=False, reason='no value screened faster than the current one'))
            continue
        candidate = dict(current, **{field: best})
        blocks = []
        for replication in range(replications):
            block = []
            for i in range(pairs):
                # Alternate which side runs first, so drift does not favour one.
                first, second = (current, candidate) if (i + replication) % 2 == 0 else (candidate, current)
                a, b = bench(first, WARMUPS, 1), bench(second, WARMUPS, 1)
                pair = (a, b) if first is current else (b, a)
                block.append(pair)
                log({'kind': 'pair', 'field': field, 'value': best, 'replication': replication, 'index': i,
                     'control': pair[0], 'candidate': pair[1]})
            blocks.append(block)
        verdict = judge_value(blocks, objective, threshold)
        steps.append(dict(step, to=best, **verdict))
        if verdict['keep']:
            current = candidate
    changed = {k: v for k, v in current.items() if v != DEFAULTS[k]}
    return {'tuning': changed, 'effective': current, 'steps': steps}


def engine_bench(engine, model, tokens_file, max_new):
    def bench(tuning, warmups, repetitions):
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as handle:
            json.dump(tuning, handle)
        try:
            result = subprocess.run([str(engine), 'bench', '--model', str(model), '--backend', 'metal',
                                     '--tokens-file', str(tokens_file), '--max-new', str(max_new),
                                     '--warmups', str(warmups), '--repetitions', str(repetitions),
                                     '--tuning', handle.name], capture_output=True, text=True, timeout=1800)
        finally:
            Path(handle.name).unlink()
        if result.returncode != 0:
            raise RuntimeError(f'engine bench failed: {result.stderr.strip()[-400:]}')
        report = json.loads(result.stdout.strip().splitlines()[-1])
        if report['tuning'] != tuning:
            raise RuntimeError('the engine ran a different tuning than requested')
        return report
    return bench


def prompt_tokens(engine, model, length):
    """`length` tokens of the engine's own encoding of repeated text."""
    text = ' The quick brown fox jumps over the lazy dog.' * (length // 4 + 8)
    result = subprocess.run([str(engine), 'tokenize', '--model', str(model), '--text', text],
                            capture_output=True, text=True, check=True, timeout=120)
    ids = json.loads(result.stdout.strip().splitlines()[-1])['ids']
    if len(ids) < length:
        raise RuntimeError('the prompt text encodes to too few tokens')
    return ids[:length]


def run(engine, model, *, prompt_length, max_new, objective, fields=None, start=None, out=None):
    from .system import admission, command
    engine, model = Path(engine).absolute(), Path(model).absolute()
    config = json.loads((model / 'config.json').read_text())
    fields = fields or relevant_fields(config)
    unknown = sorted(set(fields) - set(GRID))
    if unknown:
        raise ValueError('unknown tuning fields: ' + ', '.join(unknown))
    guard = admission()
    tokens = prompt_tokens(engine, model, prompt_length)
    raw = []
    with tempfile.TemporaryDirectory(prefix='lokahi-lab-tune-') as folder:
        tokens_file = Path(folder) / 'prompt.txt'
        tokens_file.write_text(','.join(map(str, tokens)))
        result = search(engine_bench(engine, model, tokens_file, max_new), fields, objective, start, log=raw.append)
    device = raw[0]['report']['device'] if raw else None
    report = {
        'evidence_kind': 'hardware', 'kind': 'lokahi_tuning', 'objective': objective,
        'device': device, 'os_build': command(['sw_vers', '-buildVersion']),
        'gpu_cores': _gpu_cores(), 'engine_sha256': hashlib.sha256(engine.read_bytes()).hexdigest(),
        'tuner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model': {'directory_name': model.name, 'architecture': raw[0]['report']['architecture'] if raw else None,
                  'config_sha256': hashlib.sha256((model / 'config.json').read_bytes()).hexdigest()},
        'workload': {'prompt_tokens': prompt_length, 'max_new': max_new, 'prompt_sha256':
                     hashlib.sha256(','.join(map(str, tokens)).encode()).hexdigest()},
        'protocol': {'warmups': WARMUPS, 'pairs': PAIRS, 'replications': REPLICATIONS, 'threshold': THRESHOLD},
        'admission': guard, **result,
        'raw_sha256': hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
    }
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + '\n')
        out.with_suffix('.raw.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in raw))
    return report


def _gpu_cores():
    from .system import command
    try:
        text = command(['ioreg', '-rc', 'AGXAccelerator', '-d', '1'])
    except Exception:
        return None
    for line in text.splitlines():
        if '"gpu-core-count"' in line:
            return int(line.rsplit('=', 1)[1].strip())
    return None
