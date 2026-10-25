"""SQLite history on local APFS; large artifacts live in approved storage."""
import hashlib
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
import uuid

from ..system import real_path


class History:
    def __init__(self, path):
        self.path = real_path(Path(path), exists=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # New records only. Externally modifying SQLite is not a supported API.
        self.connection = sqlite3.connect(self.path, timeout=10)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS events '
                                '(sequence INTEGER PRIMARY KEY, experiment TEXT NOT NULL, '
                                'time TEXT NOT NULL, phase TEXT NOT NULL, payload TEXT NOT NULL, '
                                'previous TEXT NOT NULL, digest TEXT UNIQUE NOT NULL)')
        self.connection.execute('CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON events '
                                'BEGIN SELECT RAISE(ABORT,"history is append-only"); END')
        self.connection.execute('CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON events '
                                'BEGIN SELECT RAISE(ABORT,"history is append-only"); END')
        self.connection.commit()
        self.verify()

    def append(self, experiment, phase, payload):
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
        timestamp = datetime.now(timezone.utc).isoformat()
        self.connection.execute('BEGIN IMMEDIATE')
        try:
            self.verify()
            row = self.connection.execute('SELECT digest FROM events ORDER BY sequence DESC LIMIT 1').fetchone()
            previous = row[0] if row else '0' * 64
            value = json.dumps([experiment, timestamp, phase, encoded, previous], separators=(',', ':')).encode()
            digest = hashlib.sha256(value).hexdigest()
            self.connection.execute('INSERT INTO events(experiment,time,phase,payload,previous,digest) VALUES(?,?,?,?,?,?)',
                                    (experiment, timestamp, phase, encoded, previous, digest))
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        return digest

    def verify(self):
        previous = '0' * 64
        for experiment, timestamp, phase, payload, expected_previous, digest in self.connection.execute(
                'SELECT experiment,time,phase,payload,previous,digest FROM events ORDER BY sequence'):
            value = json.dumps([experiment, timestamp, phase, payload, previous], separators=(',', ':')).encode()
            if expected_previous != previous or hashlib.sha256(value).hexdigest() != digest:
                raise ValueError('experiment history integrity failure')
            previous = digest
        return previous

    def show(self, experiment):
        self.verify()
        return [{'time': time, 'phase': phase, 'payload': json.loads(payload), 'digest': digest}
                for time, phase, payload, digest in self.connection.execute(
                    'SELECT time,phase,payload,digest FROM events WHERE experiment=? ORDER BY sequence', (experiment,))]

    def recent(self, limit=20):
        self.verify()
        rows = self.connection.execute('SELECT experiment,phase,payload FROM events ORDER BY sequence DESC LIMIT ?', (limit,))
        return [{'experiment': e, 'phase': phase, 'payload': json.loads(p)} for e, phase, p in rows]

    def retrieve(self, model, limit=12):
        """Small matched-model history, including failures; no private paths to LLM."""
        self.verify()
        matched = {e for e, p in self.connection.execute('SELECT experiment,payload FROM events WHERE phase="STARTED"')
                   if json.loads(p).get('model') == model}
        result = []
        for e, phase, payload in self.connection.execute('SELECT experiment,phase,payload FROM events ORDER BY sequence DESC'):
            if e not in matched or phase not in ('DECISION', 'FAILED', 'NO_IMPROVEMENT', 'RETAINED', 'BASELINE'):
                continue
            parsed = json.loads(payload)
            # Artifact paths are for local reproduction, not reasoning context.
            public = {k: v for k, v in parsed.items()
                      if k not in ('artifact', 'frozen', 'path', 'traceback', 'stdout_tail', 'stderr_tail')}
            if 'artifact' in parsed:
                public['artifact_sha256'] = parsed['artifact']['sha256']
            # Reasons can contain local paths even when artifact fields are
            # omitted. Keep failure mechanisms, redact the personal home path.
            encoded = json.dumps(public).replace(str(Path.home()), '[local-home]')
            result.append({'experiment': e, 'phase': phase, 'payload': json.loads(encoded)})
            if len(result) >= limit:
                break
        return result

    def close(self):
        self.connection.close()


def new_id():
    return uuid.uuid4().hex
