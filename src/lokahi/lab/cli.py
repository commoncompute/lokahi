"""Lōkahi lab research CLI; inference does not depend on a reasoning provider."""
import argparse
import json
from pathlib import Path

from .system import system_info


def main():
    parser = argparse.ArgumentParser(prog='lokahi-lab')
    parser.add_argument('--database', type=Path, default=Path.home() / 'Library/Application Support/Strata/miki/experiments.sqlite')
    parser.add_argument('--artifacts', type=Path, default=Path.home() / 'Library/Application Support/Strata/miki/artifacts')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('system-info')
    family = commands.add_parser('family-plan')
    family.add_argument('--catalog', type=Path, help='read-only generated CommonCompute mlxCatalog.gen.ts')
    from .bench.families import BASELINE_MODELS, EVOLVE_MODELS
    baseline = commands.add_parser('baseline')
    baseline.add_argument('model', choices=sorted(BASELINE_MODELS))
    experiment = commands.add_parser('experiment')
    actions = experiment.add_subparsers(dest='action', required=True)
    for action in ('show', 'reproduce', 'lineage'):
        sub = actions.add_parser(action)
        sub.add_argument('id')
    evolve = commands.add_parser('evolve')
    evolve.add_argument('model', choices=sorted(EVOLVE_MODELS))
    evolve.add_argument('--objective', choices=('decode', 'ttft'), default='decode')
    evolve.add_argument('--population', type=int, choices=(1, 2, 3), default=3)
    evolve.add_argument('--generations', type=int, choices=(1, 2), default=2)
    evolve.add_argument('--time-budget-seconds', type=int, choices=range(60, 3601), default=1800)
    args = parser.parse_args()
    if args.command == 'system-info':
        print(json.dumps(system_info(), indent=2))
        return 0
    if args.command == 'family-plan':
        from .bench.families import local_plan
        from benchmarks.recovered_mlx_baseline import ROOT as model_root
        memory = system_info()['memory_bytes']
        result = local_plan(memory, model_root)
        if args.catalog:
            from .bench.catalog_coverage import read_coverage
            result['catalog_coverage'] = read_coverage(args.catalog, memory)
        print(json.dumps(result, indent=2))
        return 0
    from .experiments.database import History, new_id
    from .experiments.artifacts import Artifacts
    from .bench.runner import worker
    history, artifacts = History(args.database), Artifacts(args.artifacts)
    try:
        if args.command == 'baseline':
            identifier = new_id()
            from .optimizer.integrity import snapshot, manifest_digest, verify
            from .optimizer.candidate import git
            from .bench.runner import ROOT
            frozen = snapshot(ROOT)
            history.append(identifier, 'STARTED', {'model': args.model, 'kind': 'baseline',
                                                   'git_commit': git(ROOT, 'rev-parse', 'HEAD'),
                                                   'protected_digest': manifest_digest(frozen),
                                                   'frozen': frozen,
                                                   'hardware': system_info()})
            report = worker(args.model, baseline=True)
            try:
                verify(ROOT, frozen)
            except ValueError as error:
                report.update(success=False, reason=str(error), evidence_invalid=True)
            reference = artifacts.put(identifier, 'baseline', report)
            history.append(identifier, 'BASELINE' if report.get('success') else 'FAILED',
                           {'artifact': reference, 'reason': report.get('reason'),
                            'summary': report.get('summary'), 'hardware': report.get('hardware')})
            print(json.dumps({'experiment_id': identifier, 'artifact': reference,
                              'success': report.get('success'), 'summary': report.get('summary'),
                              'reason': report.get('reason')}, indent=2))
            return 0 if report.get('success') else 1
        if args.command == 'experiment':
            if args.action == 'reproduce':
                from .optimizer.orchestrator import reproduce
                result = reproduce(history, artifacts, args.id)
            elif args.action == 'lineage':
                from .optimizer.orchestrator import lineage
                result = lineage(history, args.id)
            else:
                result = history.show(args.id)
            print(json.dumps(result, indent=2))
            return 0
        from .optimizer.orchestrator import evolve
        result = evolve(args, history, artifacts)
        print(json.dumps(result, indent=2))
        return 0 if result.get('stop') == 'bounded generation/time/no-improvement budget' else 1
    finally:
        history.close()
