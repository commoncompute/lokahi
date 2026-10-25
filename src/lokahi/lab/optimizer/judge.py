"""Protected experimental acceptance, not Strata production promotion."""
from . import integrity
from ..bench.statistics import paired_comparison
from ..bench.workloads import INITIAL_PAIRS

POLICY = {'version': 1, 'minimum_ratio': 1.02, 'maximum_cv': 0.10,
          'maximum_mlx_memory_ratio': 1.05, 'maximum_other_latency_ratio': 1.05,
          # 2 generations x 3 plans x 2 cells x 2 looks x 2 references.
          'alpha_per_cell': 0.05 / 48, 'fresh_process_replications': 2,
          'production_promotion_eligible': False}


def judge(report, objective='decode', *, integrity_ok):
    def result(decision, reason, **extra):
        return {'decision': decision, 'reason': reason, 'policy': POLICY,
                'production_promotion_eligible': False, **extra}
    if not integrity_ok:
        return result('INVALID', 'protected files or experimental identity changed')
    if objective not in ('decode', 'ttft'):
        return result('INVALID', 'unknown protected objective')
    if report.get('correctness') != 'PASS':
        return result('REJECT', 'correctness did not pass')
    if report.get('backend_verified') is not True:
        return result('INVALID', 'requested candidate was not executed')
    if report.get('replications', 0) < POLICY['fresh_process_replications']:
        return result('INCONCLUSIVE', 'missing independent fresh-process replication')
    if report.get('unstable') is not False:
        return result('INCONCLUSIVE', 'thermal, swap or admission instability')
    memory = report.get('process_memory')
    if not isinstance(memory, dict) or set(memory) != {'baseline', 'candidate'}:
        return result('INCONCLUSIVE', 'missing matched fresh-process memory probes')
    from ..bench.statistics import summary
    summary(memory.values())
    if memory['candidate'] > memory['baseline'] * 1.05:
        return result('REJECT', 'fresh-process peak memory regression')
    rows = report.get('pairs', [])
    comparisons = []
    from ..bench.workloads import WORKLOADS
    for workload in WORKLOADS:
        selected = [row for row in rows if row['workload'] == workload.name]
        if len(selected) < INITIAL_PAIRS:
            return result('INCONCLUSIVE', 'missing protected workload pairs')
        for replication in range(POLICY['fresh_process_replications']):
            if sum(row['replication'] == replication for row in selected) < INITIAL_PAIRS // 2:
                return result('INCONCLUSIVE', 'missing a fresh-process workload cell')
        for row in selected:
            a, b = row['baseline'], row['candidate']
            if a['tokens'] != b['tokens'] or len(a['tokens']) != workload.output_tokens:
                return result('REJECT', 'complete-request greedy token mismatch')
            if b['peak_mlx_bytes'] > a['peak_mlx_bytes'] * POLICY['maximum_mlx_memory_ratio']:
                return result('REJECT', 'MLX allocation regression')
            for key in ('ttft_ms', 'wall_ms'):
                if b[key] > a[key] * 1.25:
                    return result('REJECT', 'major request latency regression')
        key = 'decode_tps' if objective == 'decode' else 'ttft_ms'
        comp = paired_comparison([r['baseline'][key] for r in selected],
                                 [r['candidate'][key] for r in selected],
                                 larger_is_better=objective == 'decode', alpha=POLICY['alpha_per_cell'])
        comparisons.append({'workload': workload.name, **comp})
        if max(comp['baseline']['cv'], comp['candidate']['cv']) > POLICY['maximum_cv']:
            return result('INCONCLUSIVE', 'timing noise exceeds protected bound', comparisons=comparisons)
        if comp['upper'] < POLICY['minimum_ratio']:
            return result('REJECT', 'no meaningful objective improvement', comparisons=comparisons)
        if comp['lower'] < POLICY['minimum_ratio']:
            return result('INCONCLUSIVE', 'confidence interval does not establish >=2% improvement', comparisons=comparisons)
        import statistics
        for key in ('ttft_ms', 'wall_ms'):
            baseline = statistics.median(r['baseline'][key] for r in selected)
            candidate = statistics.median(r['candidate'][key] for r in selected)
            if candidate > baseline * POLICY['maximum_other_latency_ratio']:
                return result('REJECT', 'non-objective latency regression', comparisons=comparisons)
            a_tail = summary(r['baseline'][key] for r in selected)['p95']
            b_tail = summary(r['candidate'][key] for r in selected)['p95']
            if b_tail > a_tail * POLICY['maximum_other_latency_ratio']:
                return result('REJECT', 'p95 latency regression', comparisons=comparisons)
    return result('ACCEPT', 'replicated experimental improvement; production gates remain pending',
                  comparisons=comparisons)
