"""Exclusive, hashed artifacts in explicitly approved project storage."""
import hashlib
import json
import os
from pathlib import Path

from ..system import real_path


class Artifacts:
    def __init__(self, root):
        self.root = real_path(Path(root), exists=False)
        self.root.mkdir(parents=True, exist_ok=True)
        real_path(self.root)
        self.device = self.root.stat().st_dev

    def put(self, experiment, kind, payload):
        real_path(self.root)
        if self.root.stat().st_dev != self.device:
            raise ValueError('artifact storage mount changed')
        if not experiment.isalnum() or not kind.replace('_', '').isalnum():
            raise ValueError('invalid artifact identifier')
        data = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / f'{experiment}_{kind}_{digest}.json'
        with path.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return {'path': str(path), 'sha256': digest, 'bytes': len(data)}

    def read(self, reference):
        path = real_path(Path(reference['path']))
        if not path.is_relative_to(self.root):
            raise ValueError('artifact escaped its approved root')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != reference['sha256']:
            raise ValueError('artifact hash mismatch')
        return json.loads(data)
