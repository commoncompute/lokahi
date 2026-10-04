"""Serial fresh-process worker dispatcher, with preserved failures and timeout."""
import json
import os
from pathlib import Path
import subprocess
import sys

from ..system import real_path

ROOT = Path(__file__).resolve().parents[3]


def worker(model, *, mode='control', baseline_mode='control', baseline=False, memory_only=False,
           pairs=5, replication=0, timeout=600):
    real_path(ROOT)
    args = [sys.executable, '-m', 'miki.bench.worker', '--model', model,
            '--mode', mode, '--baseline-mode', baseline_mode,
            '--pairs', str(pairs), '--replication', str(replication)]
    if baseline:
        args.append('--baseline')
    if memory_only:
        args.append('--memory-only')
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / 'src'), str(ROOT))),
                       PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = None
    try:
        result = subprocess.run(args, cwd=ROOT, env=environment, capture_output=True,
                                text=True, timeout=timeout)
        report = json.loads(result.stdout)
        if not isinstance(report, dict) or type(report.get('success')) is not bool:
            raise ValueError('worker did not return an explicit success/failure packet')
        if result.returncode and report.get('success') is not False:
            raise ValueError('worker exited without failure evidence')
        return report
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError, ValueError) as error:
        def tail(value):
            if isinstance(value, bytes):
                value = value.decode(errors='replace')
            return (value or '')[-16000:]
        return {'success': False, 'failure': type(error).__name__, 'reason': str(error),
                'stdout_tail': tail(result.stdout if result else getattr(error, 'stdout', '')),
                'stderr_tail': tail(result.stderr if result else getattr(error, 'stderr', '')),
                'worker_returncode': result.returncode if result else None,
                'production_promotion_eligible': False}
