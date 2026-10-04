"""Finite, serial empirical generations; LLM text never decides acceptance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time

from ..bench.runner import ROOT, worker
from ..experiments.database import new_id
from ..runtime import MODES
from ..system import admission
from .candidate import create, git, retain, verify_candidate
from .integrity import snapshot, verify, manifest_digest
from .judge import judge
from .researcher import CodexResearchModel, validate_proposals


def unstable(before, after):
    def swap_used(value):
        found = re.search(r'used\s*=\s*([\d.]+)M', value)
        return float(found[1]) if found else None
    a, b = swap_used(before['swap']), swap_used(after['swap'])
    thermal = before['thermal'] + '\n' + after['thermal']
    limits = [int(value) for value in re.findall(r'CPU_(?:Speed|Scheduler)_Limit\s*=\s*(\d+)', thermal)]
    return (a is None or b is None or b - a > 128 or
            any(value < 100 for value in limits))


def evaluate(model, info, baseline_mode, frozen, *, pairs=5, objective='decode', timeout=600):
    deadline = time.monotonic() + timeout
    def remaining():
        seconds = int(deadline - time.monotonic())
        if seconds < 1:
            raise TimeoutError('candidate evaluation time budget exhausted')
        return seconds
    reports = []
    for replication in (0, 1):
        verify(ROOT, frozen)
        verify_candidate(info, frozen)
        report = worker(model, mode=info['plan']['mode'], baseline_mode=baseline_mode,
                        pairs=pairs, replication=replication, timeout=remaining())
        reports.append(report)
        verify(ROOT, frozen)
        verify_candidate(info, frozen)
        if report.get('success') is not True:
            resource_block = any(word in report.get('reason', '').lower() for word in ('reserve', 'pressure', 'timeout'))
            return {'decision': 'INCONCLUSIVE' if resource_block else 'REJECT',
                    'blocked': resource_block, 'reason': report.get('reason', 'compile/correctness/worker failure'),
                    'reports': reports, 'production_promotion_eligible': False}
    first = reports[0]
    if any(r['hardware']['class_fingerprint'] != first['hardware']['class_fingerprint'] or
           r['artifact'] != first['artifact'] or r['mode'] != info['plan']['mode'] or
           r['baseline_mode'] != baseline_mode for r in reports):
        return {'decision': 'INVALID', 'reason': 'hardware/model/plan identity mismatch', 'reports': reports}
    combined = {'correctness': 'PASS', 'backend_verified': all(r['backend_verified'] for r in reports),
                'replications': len(reports), 'pairs': [p for r in reports for p in r['pairs']],
                'unstable': any(unstable(r['admission_before'], r['admission_after']) for r in reports)}
    memory_reports = []
    memory_by_mode = {}
    for mode in dict.fromkeys((baseline_mode, info['plan']['mode'], 'control')):
        verify(ROOT, frozen)
        # One route per fresh process: avoid contaminating a control memory
        # high-water mark with compilation of a previously retained parent.
        memory = worker(model, mode=mode, baseline_mode='control',
                        memory_only=True, timeout=remaining())
        memory_reports.append(memory)
        verify(ROOT, frozen)
        verify_candidate(info, frozen)
        if (memory.get('success') is not True or memory['hardware']['class_fingerprint'] != first['hardware']['class_fingerprint']
                or memory['artifact'] != first['artifact'] or memory['mode'] != mode):
            return {'decision': 'INCONCLUSIVE', 'reason': 'matched memory probe failed',
                    'reports': reports, 'memory_reports': memory_reports, 'production_promotion_eligible': False}
        memory_by_mode[mode] = memory['peak_process_memory_bytes']
        combined['unstable'] |= unstable(memory['admission_before'], memory['admission_after'])
    combined['process_memory'] = {'baseline': memory_by_mode[baseline_mode],
                                  'candidate': memory_by_mode[info['plan']['mode']]}
    parent_result = judge(combined, objective, integrity_ok=True)
    # The immutable MLX-LM route is mandatory in EVERY generation. An accepted
    # parent never silently becomes the only comparison reference.
    if any('mlx_reference' not in pair for pair in combined['pairs']):
        mlx_result = {'decision': 'INVALID', 'reason': 'unchanged MLX reference missing'}
    else:
        mlx_report = {**combined,
                      'pairs': [{**p, 'baseline': p['mlx_reference']} for p in combined['pairs']],
                      'process_memory': {'baseline': memory_by_mode['control'],
                                         'candidate': memory_by_mode[info['plan']['mode']]}}
        mlx_result = judge(mlx_report, objective, integrity_ok=True)
    selected = combine_reference_decisions(parent_result, mlx_result)
    return {**selected, 'reference_decisions': {'parent': parent_result, 'mlx_llm': mlx_result},
            'reports': reports, 'memory_reports': memory_reports, 'production_promotion_eligible': False}


def combine_reference_decisions(parent, mlx):
    """A parent win never overrides a rejection or missing MLX comparison."""
    priority = {'INVALID': 0, 'REJECT': 1, 'INCONCLUSIVE': 2, 'ACCEPT': 3}
    selected = min((parent, mlx), key=lambda r: priority[r['decision']])
    if parent['decision'] == mlx['decision'] == 'ACCEPT':
        return {**selected, 'reason': 'replicated improvement against parent AND unchanged MLX-LM',
                'comparisons': [{**c, 'reference': name}
                                for name, value in (('parent', parent), ('mlx_llm', mlx))
                                for c in value['comparisons']]}
    return selected


def evolve(args, history, artifacts, *, researcher=None, evaluator=evaluate, candidate_root=None):
    session = new_id()
    started = time.monotonic()
    frozen = snapshot(ROOT)
    history.verify()
    digest = manifest_digest(frozen)
    commit = git(ROOT, 'rev-parse', 'HEAD')
    if git(ROOT, 'status', '--porcelain', '--untracked-files=all'):
        history.append(session, 'BLOCKED', {'reason': 'freeze a clean local controller checkpoint first'})
        return {'session': session, 'stop': 'uncommitted controller source; no candidate executed'}
    reference = 'refs/heads/miki/verified-' + args.model.lower().replace('.', '-')
    new_reference = False
    if git(ROOT, 'branch', '--list', reference.removeprefix('refs/heads/')):
        parent = git(ROOT, 'rev-parse', '--verify', reference)
        tree_names = git(ROOT, 'ls-tree', '--name-only', parent).splitlines()
        baseline_mode = json.loads(git(ROOT, 'show', parent + ':miki_candidate.json'))['mode'] if 'miki_candidate.json' in tree_names else 'control'
        if baseline_mode not in MODES:
            raise ValueError('invalid existing verified plan; preserve it for inspection')
    else:
        parent, baseline_mode = commit, 'control'
        new_reference = True
    history.append(session, 'STARTED', {'model': args.model, 'git_commit': commit,
                                       'protected_digest': digest, 'frozen': frozen,
                                       'generation_budget': args.generations, 'population': args.population,
                                       'time_budget_seconds': args.time_budget_seconds, 'objective': args.objective,
                                       'verified_ref': reference, 'parent_commit': parent})
    outcomes = []
    baseline = worker(args.model, baseline=True, timeout=min(600, args.time_budget_seconds))
    baseline_artifact = artifacts.put(session, 'baseline', baseline)
    history.append(session, 'BASELINE' if baseline.get('success') else 'FAILED',
                   {'artifact': baseline_artifact, 'reason': baseline.get('reason')})
    if not baseline.get('success'):
        return {'session': session, 'stop': baseline.get('reason'), 'accepted': 0, 'baseline': baseline_artifact}
    verify(ROOT, frozen)
    if new_reference:
        git(ROOT, 'update-ref', reference, commit, '0' * 40)
    try:
        researcher = researcher or CodexResearchModel()
    except Exception as error:
        history.append(session, 'BLOCKED', {'reason': str(error), 'stage': 'research_provider'})
        return {'session': session, 'stop': str(error), 'accepted': 0, 'retained': 0,
                'production_promotion_eligible': False}
    parent_experiment = session
    retained_count = 0
    for generation in range(args.generations):
        if time.monotonic() - started >= args.time_budget_seconds:
            break
        try:
            admission()
            verify(ROOT, frozen)
            if baseline['profile'].get('next_search_family') != 'native_mlx_mlp_compiler_plans':
                raise ValueError('PROFILE_SCOPE_UNSUPPORTED: no evidence for this bounded MLP search family')
            context = {'generation': generation, 'hardware': baseline['hardware'], 'model': baseline['artifact'],
                       'baseline': baseline['summary'], 'profile': baseline['profile'],
                       'history_successes_and_failures': history.retrieve(args.model, limit=12),
                       'source': (ROOT / 'src/miki/runtime.py').read_text(),
                       'candidate_bounds': {'population': 3, 'immutable_scope': 'miki_candidate.json'},
                       'current_verified_mode': baseline_mode}
            def set_research_deadline():
                remaining = int(args.time_budget_seconds - (time.monotonic() - started))
                if remaining < 1:
                    raise TimeoutError('research time budget exhausted')
                if hasattr(researcher, 'timeout'):
                    researcher.timeout = min(180, remaining)
            set_research_deadline()
            proposals = validate_proposals(researcher.generate_hypotheses(context))
            history.append(session, 'HYPOTHESES', {'generation': generation, 'proposals': proposals,
                                                 'provider': getattr(researcher, 'last_evidence', {})})
            set_research_deadline()
            critique = validate_proposals(researcher.critique({**context, 'proposals': proposals}))
            history.append(session, 'CRITIQUE', {'generation': generation, 'proposals': critique,
                                               'provider': getattr(researcher, 'last_evidence', {})})
        except Exception as error:
            history.append(session, 'BLOCKED', {'reason': str(error), 'generation': generation})
            return {'session': session, 'stop': str(error), 'outcomes': outcomes,
                    'accepted': sum(o['decision'] == 'ACCEPT' for o in outcomes)}
        survivors = []
        proposed_modes = {h['mode'] for h in proposals['hypotheses']}
        for hypothesis in critique['hypotheses'][:args.population]:
            identifier = new_id()
            history.append(identifier, 'STARTED', {'parent_experiment': parent_experiment, 'generation': generation,
                                                 'model': args.model, 'parent_commit': parent,
                                                 'controller_session': session,
                                                 'protected_digest': digest, 'objective': args.objective,
                                                 'baseline_mode': baseline_mode, 'hypothesis': hypothesis})
            if hypothesis['mode'] not in proposed_modes or hypothesis['discard'] or hypothesis['mode'] == baseline_mode:
                outcome = {'decision': 'REJECT', 'reason': 'critique discarded, invented or duplicate parent plan'}
            elif time.monotonic() - started >= args.time_budget_seconds:
                outcome = {'decision': 'INCONCLUSIVE', 'reason': 'time budget exhausted before evaluation'}
            else:
                try:
                    plan = researcher.generate_candidate(hypothesis)
                    base_directory = candidate_root or (Path.home() / 'Library/Application Support/Strata/miki/candidates')
                    directory = Path(base_directory) / identifier
                    info = create(ROOT, directory, parent, plan, frozen)
                    history.append(identifier, 'CANDIDATE', info)
                    remaining = max(1, int(args.time_budget_seconds - (time.monotonic() - started)))
                    outcome = evaluator(args.model, info, baseline_mode, frozen, objective=args.objective,
                                        timeout=min(600, remaining))
                    verify(ROOT, frozen)
                    verify_candidate(info, frozen)
                    history.verify()
                    first_ref = artifacts.put(identifier, 'evaluation_initial', outcome)
                    history.append(identifier, 'EVALUATION', {'artifact': first_ref})
                    if (outcome['decision'] == 'INCONCLUSIVE' and not outcome.get('blocked')
                            and time.monotonic() - started < args.time_budget_seconds):
                        remaining = max(1, int(args.time_budget_seconds - (time.monotonic() - started)))
                        outcome = evaluator(args.model, info, baseline_mode, frozen, pairs=15,
                                            objective=args.objective, timeout=min(600, remaining))
                        verify(ROOT, frozen)
                        verify_candidate(info, frozen)
                        history.verify()
                        history.append(identifier, 'REPLICATION_EXPANDED',
                                       {'artifact': artifacts.put(identifier, 'evaluation_expanded', outcome)})
                    if outcome['decision'] == 'ACCEPT':
                        survivors.append((info, identifier, min(c['lower'] for c in outcome['comparisons'])))
                except Exception as error:
                    outcome = {'decision': 'INCONCLUSIVE' if isinstance(error, TimeoutError) else ('INVALID' if 'INVALID' in str(error) else 'REJECT'),
                               'reason': str(error), 'production_promotion_eligible': False}
            # Retain all failures and unselected wins; no destructive worktree cleanup.
            conclusion = {k: v for k, v in outcome.items() if k not in ('reports', 'memory_reports')}
            history.append(identifier, 'DECISION', conclusion)
            outcomes.append({'experiment': identifier, **conclusion})
        if survivors:
            # Choose only among gated survivors; maximize the worst-cell lower
            # confidence bound for the explicit objective, not a blended score.
            accepted, accepted_id, _ = max(survivors, key=lambda item: item[2])
            verify(ROOT, frozen)
            verify_candidate(accepted, frozen)
            history.verify()
            retain(ROOT, accepted, parent, reference)
            history.append(accepted_id, 'RETAINED', {'commit': accepted['commit'], 'verified_ref': reference,
                                                     'protected_digest': digest,
                                                     'scope': 'local experimental compiler plan; not production promotion'})
            parent, baseline_mode = accepted['commit'], accepted['plan']['mode']
            parent_experiment = accepted_id
            retained_count += 1
            # Re-observe the retained plan without rewriting the original MLX
            # baseline. Subsequent evaluations still measure the control route.
            remaining_seconds = int(args.time_budget_seconds - (time.monotonic() - started))
            if remaining_seconds < 1:
                break
            refreshed = worker(args.model, baseline=True, baseline_mode=baseline_mode,
                               timeout=min(600, remaining_seconds))
            history.append(session, 'REPROFILE', {'artifact': artifacts.put(session, 'reprofile_' + str(generation), refreshed),
                                                'scope': 'new retained experimental plan observation; original MLX reference remains unchanged'})
            if not refreshed.get('success'):
                break
            baseline = refreshed
        else:
            history.append(session, 'NO_IMPROVEMENT', {'generation': generation})
    final = {'session': session, 'outcomes': outcomes, 'accepted': sum(o['decision'] == 'ACCEPT' for o in outcomes),
             'retained': retained_count,
             'elapsed_seconds': time.monotonic() - started, 'verified_ref': reference, 'verified_commit': parent,
             'stop': 'bounded generation/time/no-improvement budget', 'production_promotion_eligible': False}
    history.append(session, 'FINISHED', final)
    return final


def lineage(history, identifier):
    result, seen = [], set()
    while identifier:
        if identifier in seen:
            raise ValueError('experiment ancestry cycle')
        seen.add(identifier)
        events = history.show(identifier)
        if not events:
            raise ValueError('unknown experiment')
        result.append({'experiment': identifier, 'events': events})
        started = next((e['payload'] for e in events if e['phase'] == 'STARTED'), {})
        identifier = started.get('parent_experiment')
    return result


def reproduce(history, artifacts, identifier):
    events = history.show(identifier)
    candidate_event = next((e['payload'] for e in events if e['phase'] == 'CANDIDATE'), None)
    started = next((e['payload'] for e in events if e['phase'] == 'STARTED'), None)
    if started and started.get('kind') == 'baseline' and 'frozen' in started:
        verify(ROOT, started['frozen'])
        replay = new_id()
        history.append(replay, 'STARTED', {'parent_experiment': identifier, 'kind': 'baseline',
                                         'model': started['model'], 'frozen': started['frozen'],
                                         'protected_digest': started['protected_digest'],
                                         'git_commit': git(ROOT, 'rev-parse', 'HEAD')})
        result = worker(started['model'], baseline=True)
        verify(ROOT, started['frozen'])
        reference = artifacts.put(replay, 'baseline_reproduction', result)
        history.append(replay, 'BASELINE' if result.get('success') else 'FAILED',
                       {'artifact': reference, 'reason': result.get('reason')})
        return {'experiment_id': replay, 'success': result.get('success'), 'artifact': reference,
                'automatically_retained': False}
    if not candidate_event or not started:
        raise ValueError('experiment has no runnable candidate; show its preserved failure instead')
    session = history.show(started['controller_session'])
    frozen = next(e['payload']['frozen'] for e in session if e['phase'] == 'STARTED')
    if manifest_digest(snapshot(ROOT)) != started['protected_digest']:
        raise ValueError('protected implementation changed; restore the recorded controller before reproduction')
    replay = new_id()
    history.append(replay, 'STARTED', {'parent_experiment': identifier, 'kind': 'reproduction'})
    result = evaluate(started['model'], candidate_event, started['baseline_mode'], frozen,
                      pairs=15, objective=started['objective'])
    reference = artifacts.put(replay, 'reproduction', result)
    history.append(replay, 'DECISION', {'decision': result['decision'], 'artifact': reference})
    return {'experiment_id': replay, 'decision': result['decision'], 'artifact': reference,
            'automatically_retained': False}
