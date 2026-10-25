"""Synthetic contract tests for the lab's tuning search; they prove the
decision rules, not any speed."""
import random

import pytest

from lokahi.lab.tune import DEFAULTS, GRID, judge_value, relevant_fields, search


def fake_engine(speed, tokens=lambda tuning: [1, 2, 3], seed=7):
    """bench(tuning, warmups, repetitions) whose decode/prefill rates are speed(tuning) with 0.5% noise."""
    rng = random.Random(seed)
    calls = []

    def bench(tuning, warmups, repetitions):
        calls.append((dict(tuning), warmups, repetitions))
        decode, prefill = speed(tuning)
        reps = [{'decode_tokens_per_second': decode * rng.uniform(0.995, 1.005),
                 'prompt_tokens_per_second': prefill * rng.uniform(0.995, 1.005),
                 'token_ids': tokens(tuning)} for _ in range(repetitions)]
        return {'device': 'Apple M1', 'architecture': 'llama', 'tuning': dict(tuning), 'repetitions': reps}
    return bench, calls


def test_keeps_a_value_faster_in_every_replication_and_defaults_elsewhere():
    bench, calls = fake_engine(lambda t: (100.0 * (1.2 if t['decode_groups_per_core'] == 16 else 1.0), 500.0))
    result = search(bench, ['decode_groups_per_core', 'inflight_steps'], 'decode')
    assert result['tuning'] == {'decode_groups_per_core': 16}
    kept = [s for s in result['steps'] if s.get('keep')]
    assert [s['field'] for s in kept] == ['decode_groups_per_core']
    # Judged pairs ran with 3 warmups and one measured repetition each.
    assert all(w == 3 and r == 1 for _, w, r in calls if r == 1)


def test_rejects_a_faster_value_that_changes_greedy_tokens():
    bench, _ = fake_engine(lambda t: (150.0 if t['inflight_steps'] == 4 else 100.0, 500.0),
                           tokens=lambda t: [1, 2, 4] if t['inflight_steps'] == 4 else [1, 2, 3])
    result = search(bench, ['inflight_steps'], 'decode')
    assert result['tuning'] == {}
    assert result['steps'][0]['reason'] == 'greedy tokens changed'


def test_rejects_a_decode_win_that_costs_prefill():
    bench, _ = fake_engine(lambda t: (130.0, 400.0) if t['attn_mm_min_rows'] == 64 else (100.0, 500.0))
    result = search(bench, ['attn_mm_min_rows'], 'decode')
    assert result['tuning'] == {}
    assert result['steps'][0]['reason'] == 'prefill regressed'


def test_noise_alone_is_not_a_win():
    bench, _ = fake_engine(lambda t: (100.0, 500.0), seed=11)
    result = search(bench, list(GRID), 'decode')
    assert result['tuning'] == {} and result['effective'] == DEFAULTS


def test_protocol_minimums_are_enforced():
    bench, _ = fake_engine(lambda t: (100.0, 500.0))
    with pytest.raises(ValueError, match='10 pairs'):
        search(bench, ['inflight_steps'], 'decode', pairs=5)
    with pytest.raises(ValueError, match='ten matched pairs'):
        judge_value([[({'repetitions': [{'decode_tokens_per_second': 1.0, 'prompt_tokens_per_second': 1.0,
                                         'token_ids': []}]},) * 2] * 3], 'decode')


def test_fields_follow_the_architecture():
    dense = relevant_fields({'model_type': 'llama'})
    assert 'moe_grouped_min_tokens' not in dense and 'gdn_chunk_min_rows' not in dense
    assert 'moe_grouped_min_tokens' in relevant_fields({'num_experts': 128})
    assert 'gdn_chunk_min_rows' in relevant_fields({'text_config': {'layer_types': ['linear_attention']}})
    assert 'latent_min_block_keys' in relevant_fields({'kv_lora_rank': 512})
    assert set(DEFAULTS) == set(GRID)
