"""Catalog fixtures are planning evidence, not runtime support or speed."""
import pytest

from miki.bench.catalog_coverage import coverage, read_coverage
from miki.bench.families import GIB


def row(identifier='tiny', architecture='llama', memory=8, weights=.7,
        loadable='true', kind='llm', repo='mlx-community/Tiny', revision='a' * 40):
    return (f'  {{ id: "{identifier}", repo: "{repo}", revision: "{revision}", '
            f'kind: \'{kind}\', modelType: "{architecture}", minMemoryGb: {memory}, '
            f'weightsGb: {weights}, loadable: {loadable} }},')


def test_all_architecture_and_modality_lanes_remain_distinct():
    value = coverage('\n'.join([row(), row('large', weights=2),
                              row('hybrid', architecture='qwen3_5', memory=24),
                              row('vision', architecture='llama', kind='vlm'),
                              row('unsupported', architecture='qwen3_vl_moe', loadable='false')]), 16 * GIB)
    assert value['catalog_rows'] == 5 and value['architecture_lanes'] == 4
    lanes = {r['id']: r for r in value['lanes']}
    assert lanes['tiny']['status'] == 'LOCAL_CAPACITY_CANDIDATE'
    assert lanes['hybrid']['status'] == 'FLEET_OR_COMPONENT_ONLY'
    assert lanes['unsupported']['status'] == 'UNSUPPORTED_IN_CATALOG'
    assert 'vision' in lanes and 'large' not in lanes
    assert not value['production_promotion_eligible'] and not value['download_performed']
    assert all(not r['performance_qualified'] and not r['miki_execution_qualified'] for r in lanes.values())


def test_live_sizing_finding_requires_exact_artifact_pin():
    qwen = row('qwen3.8-27b', 'qwen3_5', 24, 4.67,
               repo='mlx-community/Qwen3.8-27B-4bit',
               revision='3e6447f082e89cc7f0bc6e5441afd38dfce760ff')
    finding = coverage(qwen, 16 * GIB)['sizing_findings'][0]
    assert finding['verified_remote_weight_bytes'] == 16054541349
    assert coverage(qwen.replace('3e6447f0', 'ae6447f0'), 16 * GIB)['sizing_findings'] == []


@pytest.mark.parametrize('text', ['', row() + '\n' + row(), row(revision='main'),
                                 row(memory=0), row(weights=0), row().replace('{ id:', '{id:'),
                                 'x' * (512 * 1024 + 1)])
def test_malformed_or_unbounded_catalog_rejected(text):
    with pytest.raises(ValueError):
        coverage(text, 16 * GIB)


def test_no_follow_reading_and_generated_text_is_never_executed(tmp_path):
    catalog = tmp_path / 'catalog.ts'
    catalog.write_text('throw new Error("must never execute");\n' + row())
    value = read_coverage(catalog, 16 * GIB)
    assert value['architecture_lanes'] == 1
    alias = tmp_path / 'alias.ts'
    alias.symlink_to(catalog)
    with pytest.raises(ValueError, match='symlink'):
        read_coverage(alias, 16 * GIB)
    with pytest.raises(ValueError, match='regular'):
        read_coverage(tmp_path, 16 * GIB)
