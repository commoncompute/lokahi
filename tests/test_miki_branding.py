"""Rename compatibility: no implicit installation, model loading or duplicate implementation."""
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_public_facade_preserves_legacy_identity_and_lazy_imports():
    code = '''
import sys
import miki_engine
assert 'mlx.core' not in sys.modules
assert 'transformers' not in sys.modules
import ollm
assert miki_engine.TensorTracer is ollm.TensorTracer
assert miki_engine.TraceEvent is ollm.TraceEvent
from miki_engine.native_session import NativeSession
from ollm.runtime.native_session import NativeSession as LegacySession
assert NativeSession is LegacySession
assert 'mlx.core' not in sys.modules
assert 'transformers' not in sys.modules
try:
    miki_engine.not_a_runtime
except AttributeError:
    pass
else:
    raise AssertionError('unknown facade name accepted')
'''
    subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
                   capture_output=True, check=True, timeout=20)
    assert 'name = "miki-engine"' in (ROOT / 'pyproject.toml').read_text()
