"""Canary selection is policy/synthetic evidence, never performance or admission."""
from dataclasses import FrozenInstanceError
import re

import pytest

from miki.bench.families import CANARIES, GIB, local_plan, plan
from miki.runtime import MODES


def test_m1_canaries_do_not_bypass_catalog_capacity_or_claim_scaling():
    result = plan(16 * GIB, ['Qwen3-0.6B-4bit', 'Llama-3.2-3B-Instruct-4bit'])
    rows = {r['key']: r for r in result['canaries']}
    assert rows['qwen3-dense']['controller_model_allowlisted'] is True
    assert rows['llama-current']['controller_model_allowlisted'] is True
    assert rows['qwen-hybrid']['controller_model_allowlisted'] is False
    assert rows['deepseek-distill']['architecture'] == 'qwen2'
    assert 'DeepSeek MLA' in rows['deepseek-distill']['does_not_cover']
    for key in ('glm-chat-fleet', 'gpt-oss-fleet'):
        assert rows[key]['status'] == 'FLEET_OR_COMPONENT_ONLY'
        assert rows[key]['nominal_tier_fits'] is False
    for row in rows.values():
        assert row['performance_qualified'] is False
        assert row['fleet_scaling_verified'] is False
        assert row['artifact_and_live_admission_required'] is True
    assert result['production_promotion_eligible'] is False
    assert result['download_performed'] is False
    assert MODES == ('control', 'compile_decode', 'compile_prefill', 'compile_both')


def test_family_snapshots_are_immutable_pinned_and_distinct():
    assert len({c.key for c in CANARIES}) == len(CANARIES)
    assert {c.family for c in CANARIES} == {'qwen', 'llama', 'gemma', 'deepseek', 'glm', 'gpt-oss'}
    assert all(re.fullmatch('[0-9a-f]{40}', c.revision) for c in CANARIES)
    assert all(c.does_not_cover for c in CANARIES)
    assert all(not r['fleet_scaling_verified'] for r in plan(512 * GIB)['canaries'])
    with pytest.raises(FrozenInstanceError):
        CANARIES[0].revision = 'main'


@pytest.mark.parametrize('memory', [0, -1, True, 16.0, 2**64])
def test_invalid_hardware_does_not_become_capacity(memory):
    with pytest.raises(ValueError):
        plan(memory)


def test_local_plan_inventory_is_not_artifact_qualification(tmp_path):
    name = CANARIES[0].local_name
    (tmp_path / name).mkdir()  # intentionally empty: presence is not validity
    result = local_plan(16 * GIB, tmp_path)
    row = result['canaries'][0]
    assert row['snapshot_present'] is True and row['performance_qualified'] is False
    assert row['status'] == 'LOCAL_SNAPSHOT_NEEDS_QUALIFICATION'
    (tmp_path / CANARIES[1].local_name).symlink_to(tmp_path / name)
    with pytest.raises(ValueError, match='symlink'):
        local_plan(16 * GIB, tmp_path)
