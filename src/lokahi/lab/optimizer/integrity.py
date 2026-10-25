"""Trusted-parent no-follow allowlist/integrity protection, not an OS sandbox."""
import hashlib
from pathlib import Path
import stat

from ..system import real_path

# The Miki-era name stays: retained checkpoints and private history refer to it.
CANDIDATE_FILE = 'miki_candidate.json'


def digest(path):
    real_path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('only real regular protected files are permitted')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root):
    root = real_path(Path(root))
    paths = [root / 'AGENTS.md', root / 'pyproject.toml']
    for name in ('src', 'tests', 'benchmarks', 'native'):
        directory = root / name
        real_path(directory)
        # Directory symlinks are never traversed; file symlinks fail closed.
        import os
        for folder, directories, files in os.walk(directory, followlinks=False):
            directories[:] = [d for d in directories if d not in ('__pycache__', 'build')]
            for d in directories:
                real_path(Path(folder) / d)
            for file in files:
                path = Path(folder) / file
                if path.suffix in ('.py', '.metal', '.mm', '.m', '.h', '.cpp', '.c', '.json', '.toml') or file.startswith('Makefile') or path.suffix == '.sh':
                    paths.append(path)
    return {str(path.relative_to(root)): digest(path) for path in sorted(paths)}


def verify(root, frozen):
    current = snapshot(root)
    if current != frozen:
        changed = sorted(path for path in set(current) | set(frozen) if current.get(path) != frozen.get(path))
        raise ValueError('INVALID_EXPERIMENT: protected source changed: ' + ', '.join(changed[:8]))
    return True


def manifest_digest(frozen):
    import json
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
