"""Isolated local Git worktrees; only a bounded plan manifest is mutable."""
import hashlib
import json
from pathlib import Path
import subprocess

from ..runtime import MODES
from ..system import real_path
from .integrity import CANDIDATE_FILE, verify


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, timeout=30).strip()


def create(root, directory, parent, plan, frozen):
    root, directory = real_path(Path(root)), real_path(Path(directory), exists=False)
    if plan.get('schema_version') != 1 or set(plan) != {'schema_version', 'mode'} or plan['mode'] not in MODES[1:]:
        raise ValueError('candidate plan is outside its immutable contract')
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise ValueError('candidate workspace already exists')
    git(root, 'worktree', 'add', '--detach', str(directory), parent)
    verify(directory, frozen)
    path = directory / CANDIDATE_FILE
    # Only controlled plan metadata is generated; no arbitrary source execution.
    data = json.dumps(plan, sort_keys=True, indent=2) + '\n'
    if path.exists():
        real_path(path)
    path.write_text(data, encoding='utf-8')
    git(directory, 'add', '--', CANDIDATE_FILE)
    git(directory, 'commit', '-m', 'experiment(miki): ' + plan['mode'])
    verify(directory, frozen)
    changes = git(directory, 'diff', '--name-only', parent, 'HEAD').splitlines()
    if changes != [CANDIDATE_FILE]:
        raise ValueError('INVALID_EXPERIMENT: candidate escaped its exact allowlist')
    diff = git(directory, 'diff', '--binary', parent, 'HEAD').encode()
    return {'path': str(directory), 'commit': git(directory, 'rev-parse', 'HEAD'),
            'parent_commit': parent, 'diff_sha256': hashlib.sha256(diff).hexdigest(), 'plan': plan}


def retain(root, info, expected_parent, reference):
    # Local compare-and-swap checkpoint; never force-push or alter a checked-out
    # branch. Accepted research plans do not become CommonCompute defaults.
    if git(root, 'rev-parse', reference) != expected_parent:
        raise ValueError('verified checkpoint advanced concurrently')
    git(root, 'merge-base', '--is-ancestor', expected_parent, info['commit'])
    git(root, 'update-ref', reference, info['commit'], expected_parent)
    return info['commit']


def verify_candidate(info, frozen):
    directory = real_path(Path(info['path']))
    verify(directory, frozen)
    if (git(directory, 'rev-parse', 'HEAD') != info['commit'] or
            git(directory, 'status', '--porcelain', '--untracked-files=all')):
        raise ValueError('INVALID_EXPERIMENT: candidate checkpoint changed')
    path = real_path(directory / CANDIDATE_FILE)
    if json.loads(path.read_text()) != info['plan']:
        raise ValueError('INVALID_EXPERIMENT: candidate plan changed')
    diff = git(directory, 'diff', '--binary', info['parent_commit'], 'HEAD').encode()
    if hashlib.sha256(diff).hexdigest() != info['diff_sha256']:
        raise ValueError('INVALID_EXPERIMENT: candidate diff changed')
    return True
