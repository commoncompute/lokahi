"""Deterministic paired log-ratio bootstrap, no LLM-derived statistics."""
import math
import random
import statistics


def summary(values):
    values = list(values)
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, float))
                         or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('samples must be finite positive measurements')
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    sorted_ = sorted(values)
    return {'n': len(values), 'mean': mean, 'median': statistics.median(values),
            'stddev': sd, 'min': min(values), 'max': max(values), 'cv': sd / mean,
            'p95': sorted_[min(len(sorted_) - 1, math.ceil(0.95 * len(sorted_)) - 1)]}


def paired_comparison(control, candidate, *, larger_is_better=False, alpha=0.05):
    control, candidate = list(control), list(candidate)
    a, b = summary(control), summary(candidate)
    if len(control) != len(candidate) or len(control) < 10:
        raise ValueError('at least ten matched pairs are required')
    ratios = [y / x if larger_is_better else x / y for x, y in zip(control, candidate)]
    logs = [math.log(ratio) for ratio in ratios]
    rng = random.Random(730)
    bootstrap = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs))))
                       for _ in range(2000))
    low = bootstrap[int(alpha / 2 * len(bootstrap))]
    high = bootstrap[min(len(bootstrap) - 1, math.ceil((1 - alpha / 2) * len(bootstrap)) - 1)]
    return {'baseline': a, 'candidate': b, 'paired_ratios': ratios,
            'geometric_ratio': math.exp(statistics.mean(logs)), 'lower': low, 'upper': high,
            'confidence': 1 - alpha, 'method': 'paired_log_ratio_percentile_bootstrap_2000_seed730'}
