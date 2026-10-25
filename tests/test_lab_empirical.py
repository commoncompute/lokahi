"""Synthetic controller contracts. These fixtures are not hardware evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

import pytest

from lokahi.lab.bench.statistics import paired_comparison, summary
from lokahi.lab.bench.workloads import WORKLOADS
from lokahi.lab.experiments.database import History
from lokahi.lab.experiments.artifacts import Artifacts
from lokahi.lab.optimizer.integrity import snapshot, verify
from lokahi.lab.optimizer.judge import judge
from lokahi.lab.optimizer.researcher import validate_proposals
from lokahi.lab.optimizer.orchestrator import unstable
from lokahi.lab.system import real_path


def report(ratio=1.1):
    pairs = []
    for workload in WORKLOADS:
        for index in range(10):
            baseline = {'ttft_ms': 100.0, 'wall_ms': 1000.0, 'decode_tps': 32.0,
                        'tokens': [3] * workload.output_tokens, 'peak_mlx_bytes': 1024}
            measured = {**baseline, 'ttft_ms': 100 / ratio, 'wall_ms': 1000 / ratio,
                        'decode_tps': 32 * ratio}
            pairs.append({'workload': workload.name, 'replication': index % 2,
                          'baseline': baseline, 'candidate': measured})
    return {'correctness': 'PASS', 'backend_verified': True, 'replications': 2,
            'unstable': False, 'pairs': pairs,
            'process_memory': {'baseline': 1024, 'candidate': 1024}}


def hypothesis(mode='compile_decode'):
    return {**{field: 'synthetic fixture' for field in (
        'title', 'mechanism', 'expected_benefit', 'downside', 'conditions', 'experiment', 'critique')},
            'affected_code': 'miki_candidate.json', 'mode': mode, 'discard': False}


def test_statistics_are_deterministic_and_do_not_invent_confidence():
    a = paired_comparison([100] * 10, [90] * 10)
    assert a == paired_comparison([100] * 10, [90] * 10)
    assert a['lower'] == pytest.approx(100 / 90)
    assert summary([1, 2, 3])['median'] == 2
    assert summary([1, 2, 3])['p95'] == 3
    with pytest.raises(ValueError):
        paired_comparison([100], [90])


@pytest.mark.parametrize('values', [[], [0], [-1], [True], [float('nan')], [float('inf')]])
def test_invalid_timing_samples_fail_closed(values):
    with pytest.raises(ValueError):
        summary(values)


def test_synthetic_acceptance_never_means_production_promotion():
    result = judge(report(), integrity_ok=True)
    assert result['decision'] == 'ACCEPT'
    assert result['production_promotion_eligible'] is False
    assert judge(report(ratio=1.01), integrity_ok=True)['decision'] == 'REJECT'
    assert judge(report(), integrity_ok=False)['decision'] == 'INVALID'


@pytest.mark.parametrize('mutation,decision', [
    (lambda r: r.update(correctness='FAIL'), 'REJECT'),
    (lambda r: r.update(backend_verified=False), 'INVALID'),
    (lambda r: r.update(replications=1), 'INCONCLUSIVE'),
    (lambda r: r.update(unstable=True), 'INCONCLUSIVE'),
    (lambda r: r.pop('process_memory'), 'INCONCLUSIVE'),
    (lambda r: r['process_memory'].update(candidate=2048), 'REJECT'),
    (lambda r: r['pairs'][-1]['candidate'].update(tokens=[8]), 'REJECT'),
    (lambda r: r['pairs'][-1]['candidate'].update(peak_mlx_bytes=2048), 'REJECT'),
    (lambda r: r['pairs'].pop(), 'INCONCLUSIVE'),
])
def test_deterministic_judge_rejects_invalid_evidence(mutation, decision):
    value = report()
    mutation(value)
    assert judge(value, integrity_ok=True)['decision'] == decision


def test_noise_is_inconclusive_not_a_forced_winner():
    value = report()
    for index, row in enumerate(value['pairs']):
        row['candidate']['decode_tps'] = 25 if index % 2 else 60
    assert judge(value, integrity_ok=True)['decision'] == 'INCONCLUSIVE'


def test_database_preserves_failures_and_rejects_update_delete(tmp_path):
    history = History(tmp_path.resolve() / 'history.sqlite')
    history.append('fixture', 'COMPILE_FAILED', {'reason': 'synthetic failure'})
    history.append('fixture', 'INCONCLUSIVE', {'reason': 'synthetic noise'})
    assert len(history.show('fixture')) == 2
    with pytest.raises(sqlite3.IntegrityError):
        history.connection.execute('DELETE FROM events')
    history.connection.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        history.connection.execute("UPDATE events SET phase='ACCEPT'")
    history.connection.rollback()
    history.close()


def protected_repo(root):
    root = root.resolve()
    root.mkdir(exist_ok=True)
    (root / 'AGENTS.md').write_text('correctness first')
    (root / 'pyproject.toml').write_text('')
    for name in ('src', 'tests', 'benchmarks', 'native'):
        (root / name).mkdir()
        (root / name / 'reference.py').write_text('protected = True\n')
    return root


def test_protected_reference_and_added_source_are_detected(tmp_path):
    root = protected_repo(tmp_path / 'parent')
    frozen = snapshot(root)
    assert verify(root, frozen)
    (root / 'miki_candidate.json').write_text('{"mode":"compile_decode"}')
    assert verify(root, frozen)
    (root / 'tests/reference.py').write_text('weakened = True')
    with pytest.raises(ValueError, match='INVALID_EXPERIMENT'):
        verify(root, frozen)


def test_failed_drive_symlinks_rejected_before_traversal(tmp_path):
    link = tmp_path.resolve() / 'failed'
    link.symlink_to('/Volumes/Tyler HDD', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        real_path(link / 'strata', exists=False)
    with pytest.raises(ValueError, match='failed HDD'):
        real_path(Path('/Volumes/Tyler HDD/strata'), exists=False)


def test_artifacts_are_exclusive_and_hash_verified(tmp_path):
    artifacts = Artifacts(tmp_path.resolve() / 'artifacts')
    ref = artifacts.put('fixture', 'failure', {'success': False})
    assert artifacts.read(ref) == {'success': False}
    with pytest.raises(FileExistsError):
        artifacts.put('fixture', 'failure', {'success': False})
    Path(ref['path']).write_text('{"success":true}')
    with pytest.raises(ValueError, match='hash mismatch'):
        artifacts.read(ref)


def test_proposal_contract_prevents_policy_or_code_changes():
    value = {'hypotheses': [hypothesis()], 'evidence_limits': ['synthetic fixture, not measured']}
    assert validate_proposals(value) is value
    for field, replacement in (('affected_code', 'src/lokahi/lab/optimizer/judge.py'),
                               ('mode', 'invented'), ('discard', 0)):
        changed = copy.deepcopy(value)
        changed['hypotheses'][0][field] = replacement
        with pytest.raises(ValueError):
            validate_proposals(changed)
    with pytest.raises(ValueError):
        validate_proposals({**value, 'hypotheses': [hypothesis()] * 4})


def test_thermal_and_swap_rejections():
    before = {'swap': 'used = 1000.00M', 'thermal': ''}
    assert not unstable(before, {'swap': 'used = 1001.00M', 'thermal': 'CPU_Speed_Limit = 100'})
    assert unstable(before, {'swap': 'used = 1500.00M', 'thermal': ''})
    assert unstable(before, {'swap': 'used = 1000.00M', 'thermal': 'CPU_Speed_Limit = 80'})
    assert unstable({**before, 'thermal': 'CPU_Speed_Limit = 80'}, before)


def test_candidate_isolated_and_only_exact_manifest_committed(tmp_path):
    from lokahi.lab.optimizer.candidate import create, git, retain, verify_candidate
    root = protected_repo(tmp_path / 'repo')
    git(root, 'init', '-b', 'fixture')
    git(root, 'config', 'user.name', 'Lab fixture')
    git(root, 'config', 'user.email', 'lab-fixture@example.invalid')
    git(root, 'add', '--', 'AGENTS.md', 'pyproject.toml', 'src', 'tests', 'benchmarks', 'native')
    git(root, 'commit', '-m', 'synthetic protected fixture')
    parent = git(root, 'rev-parse', 'HEAD')
    frozen = snapshot(root)
    ref = 'refs/heads/fixture-verified'
    git(root, 'update-ref', ref, parent)
    info = create(root, tmp_path.resolve() / 'candidate', parent,
                  {'schema_version': 1, 'mode': 'compile_decode'}, frozen)
    assert git(root, 'rev-parse', 'HEAD') == parent
    assert retain(root, info, parent, ref) == info['commit']
    assert git(root, 'rev-parse', ref) == info['commit']
    assert not (root / 'miki_candidate.json').exists()
    assert verify_candidate(info, frozen)
    with pytest.raises(ValueError, match='concurrently'):
        retain(root, info, parent, ref)
    (Path(info['path']) / 'miki_candidate.json').write_text('{"mode":"compile_both"}')
    with pytest.raises(ValueError, match='INVALID_EXPERIMENT'):
        verify_candidate(info, frozen)


def test_unknown_objective_and_unbalanced_replication_fail_closed():
    assert judge(report(), objective='invented', integrity_ok=True)['decision'] == 'INVALID'
    value = report()
    for workload in WORKLOADS:
        rows = [p for p in value['pairs'] if p['workload'] == workload.name]
        for index, row in enumerate(rows):
            row['replication'] = 1 if index == 0 else 0
    assert judge(value, integrity_ok=True)['decision'] == 'INCONCLUSIVE'


def test_p95_regression_rejected_even_when_median_is_good():
    value = report()
    value['pairs'][0]['candidate']['ttft_ms'] = 110.0
    assert judge(value, integrity_ok=True)['reason'] == 'p95 latency regression'


def test_unchanged_mlx_comparison_is_required_after_parent_win():
    from lokahi.lab.optimizer.orchestrator import combine_reference_decisions
    parent = {'decision': 'ACCEPT', 'reason': 'synthetic parent win'}
    for decision in ('REJECT', 'INCONCLUSIVE', 'INVALID'):
        mlx = {'decision': decision, 'reason': 'synthetic MLX comparison'}
        assert combine_reference_decisions(parent, mlx) is mlx


def test_history_detects_tampering_before_append(tmp_path):
    history = History(tmp_path.resolve() / 'history.sqlite')
    history.append('fixture', 'FAILED', {'reason': 'synthetic'})
    history.connection.execute('DROP TRIGGER no_update')
    history.connection.execute("UPDATE events SET phase='ACCEPT'")
    history.connection.commit()
    with pytest.raises(ValueError, match='integrity'):
        history.append('new', 'STARTED', {})
    history.close()


def test_research_history_contains_failures_but_not_personal_paths(tmp_path):
    history = History(tmp_path.resolve() / 'history.sqlite')
    history.append('fixture', 'STARTED', {'model': 'fixture'})
    history.append('fixture', 'FAILED', {'reason': str(Path.home() / 'private/example'),
                                       'traceback': 'private traceback', 'stdout_tail': 'private output'})
    value = history.retrieve('fixture')
    encoded = json.dumps(value)
    assert 'FAILED' in encoded and '[local-home]' in encoded
    assert str(Path.home()) not in encoded and 'private traceback' not in encoded
    history.close()


def test_prepared_control_plan_checks_identity_and_restores_without_mlx():
    from lokahi.lab.runtime import prepare, candidate
    model = object()
    plan = prepare(model, 'control')
    with candidate(model, 'control', prepared=plan) as stats:
        assert stats['candidate_calls'] == 0
    with pytest.raises(ValueError, match='identity'):
        with candidate(object(), 'control', prepared=plan):
            pass


def synthetic_controller(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from lokahi.lab.optimizer import orchestrator
    from lokahi.lab.optimizer.candidate import git
    from lokahi.lab.optimizer.researcher import ResearchModel
    root = protected_repo(tmp_path / 'repo')
    (root / 'src/lokahi/lab').mkdir(parents=True)
    (root / 'src/lokahi/lab/runtime.py').write_text('synthetic_fixture = True\n')
    git(root, 'init', '-b', 'fixture')
    git(root, 'config', 'user.name', 'Lab fixture')
    git(root, 'config', 'user.email', 'lab-fixture@example.invalid')
    git(root, 'add', '--', 'AGENTS.md', 'pyproject.toml', 'src', 'tests', 'benchmarks', 'native')
    git(root, 'commit', '-m', 'synthetic protected fixture')
    monkeypatch.setattr(orchestrator, 'ROOT', root)
    monkeypatch.setattr(orchestrator, 'admission', lambda: {})
    baseline = {'success': True, 'hardware': {'evidence_kind': 'synthetic'},
                'artifact': {'model': 'synthetic'}, 'summary': {},
                'profile': {'next_search_family': 'native_mlx_mlp_compiler_plans'}}
    monkeypatch.setattr(orchestrator, 'worker', lambda *a, **k: baseline)
    class FakeResearch(ResearchModel):
        def request(self, stage, context):
            mode = 'compile_both' if context['current_verified_mode'] == 'compile_decode' else 'compile_decode'
            return {'hypotheses': [hypothesis(mode)], 'evidence_limits': ['synthetic only']}
    args = SimpleNamespace(model='fixture', population=1, generations=2,
                           objective='decode', time_budget_seconds=60)
    return (orchestrator, root, args, FakeResearch(),
            History(tmp_path.resolve() / 'history.sqlite'),
            Artifacts(tmp_path.resolve() / 'artifacts'))


def test_synthetic_two_generation_controller_keeps_ancestry_and_reprofiles(tmp_path, monkeypatch):
    from lokahi.lab.optimizer.candidate import git
    module, root, args, researcher, history, artifacts = synthetic_controller(tmp_path, monkeypatch)
    before = git(root, 'rev-parse', 'HEAD')
    evaluated = []
    def evaluator(model, info, baseline_mode, frozen, **kwargs):
        evaluated.append((info['plan']['mode'], baseline_mode))
        return judge(report(), integrity_ok=True)
    result = module.evolve(args, history, artifacts, researcher=researcher, evaluator=evaluator,
                           candidate_root=tmp_path.resolve() / 'candidates')
    assert result['accepted'] == result['retained'] == 2
    assert evaluated == [('compile_decode', 'control'), ('compile_both', 'compile_decode')]
    assert git(root, 'rev-parse', 'HEAD') == before
    assert git(root, 'rev-parse', result['verified_ref']) != before
    assert len(module.lineage(history, result['outcomes'][-1]['experiment'])) == 3
    assert len([e for e in history.show(result['session']) if e['phase'] == 'REPROFILE']) == 2
    assert result['production_promotion_eligible'] is False
    history.close()


def test_synthetic_resource_block_preserves_failure_without_verified_ref(tmp_path, monkeypatch):
    from lokahi.lab.optimizer.candidate import git
    module, root, args, researcher, history, artifacts = synthetic_controller(tmp_path, monkeypatch)
    monkeypatch.setattr(module, 'worker', lambda *a, **k: {'success': False, 'reason': 'synthetic reserve blocked'})
    result = module.evolve(args, history, artifacts, researcher=researcher,
                           candidate_root=tmp_path.resolve() / 'candidates')
    assert result['accepted'] == 0
    assert result['stop'] == 'synthetic reserve blocked'
    assert any(e['phase'] == 'FAILED' for e in history.show(result['session']))
    assert not git(root, 'branch', '--list', 'miki/verified-*')
    history.close()


@pytest.mark.parametrize('reference,expected', [('missing', 'INVALID'), ('slower', 'REJECT'), ('passing', 'ACCEPT')])
def test_actual_evaluator_requires_mlx_reference_in_later_generation(tmp_path, monkeypatch, reference, expected):
    from lokahi.lab.optimizer.candidate import create, git
    module, root, args, researcher, history, artifacts = synthetic_controller(tmp_path, monkeypatch)
    frozen = snapshot(root)
    info = create(root, tmp_path.resolve() / 'candidate', git(root, 'rev-parse', 'HEAD'),
                  {'schema_version': 1, 'mode': 'compile_both'}, frozen)
    stable = {'swap': 'used = 1000.00M', 'thermal': 'CPU_Speed_Limit = 100'}
    memory_routes = []
    def fake_worker(model, *, mode, baseline_mode, memory_only=False, pairs=5, replication=0, **kwargs):
        packet = {'success': True, 'hardware': {'class_fingerprint': 'synthetic'},
                  'artifact': {'model': 'synthetic'}, 'mode': mode, 'baseline_mode': baseline_mode,
                  'backend_verified': True, 'admission_before': stable, 'admission_after': stable,
                  'peak_process_memory_bytes': 1024}
        if memory_only:
            memory_routes.append((mode, baseline_mode))
            return packet
        rows = [copy.deepcopy(row) for row in report()['pairs'] if row['replication'] == replication]
        for row in rows:
            if reference != 'missing':
                row['mlx_reference'] = copy.deepcopy(row['baseline'])
                if reference == 'slower':
                    row['mlx_reference']['decode_tps'] = 40.0
        packet['pairs'] = rows
        return packet
    monkeypatch.setattr(module, 'worker', fake_worker)
    result = module.evaluate('fixture', info, 'compile_decode', frozen)
    assert result['decision'] == expected
    assert set(memory_routes) == {('compile_decode', 'control'), ('compile_both', 'control'), ('control', 'control')}
    assert result['production_promotion_eligible'] is False
    history.close()


def test_codex_structured_contract_no_live_provider_call(monkeypatch):
    from types import SimpleNamespace
    from lokahi.lab.optimizer import researcher as module
    monkeypatch.setattr(module.shutil, 'which', lambda name: '/synthetic/codex')
    value = {'hypotheses': [hypothesis()], 'evidence_limits': ['synthetic only']}
    captured = []
    def fake_run(command, **kwargs):
        captured.append((command, kwargs))
        event = {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': json.dumps(value)}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(event), stderr='')
    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    provider = module.CodexResearchModel()
    assert provider.generate_hypotheses({'evidence_kind': 'synthetic'}) == value
    command, kwargs = captured[0]
    assert '--ignore-user-config' in command and '--ephemeral' in command
    assert command[command.index('--sandbox') + 1] == 'read-only'
    assert command[command.index('--model') + 1] == 'gpt-6-astra'
    assert '--output-schema' in command and kwargs['timeout'] <= 180
    def forbidden(*a, **k):
        event = {'type': 'item.completed', 'item': {'type': 'command_execution', 'command': 'synthetic'}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(event), stderr='')
    monkeypatch.setattr(module.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='CONTRACT_VIOLATION'):
        provider.generate_hypotheses({'evidence_kind': 'synthetic'})
