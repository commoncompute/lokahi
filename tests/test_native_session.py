"""Native CPU lifecycle correctness, never LLM speed or security evidence."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from lokahi.runtime.native_session import Config, Request, Snapshot, NativeSession, NativeSessionError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp('native-session').resolve()
    target = root / 'libstrata_session.dylib'
    subprocess.run(['xcrun', 'clang++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-pthread', '-dynamiclib', str(ROOT / 'native/runtime/strata_session.cpp'),
                    '-o', str(target)], check=True, capture_output=True)
    return target, hashlib.sha256(target.read_bytes()).hexdigest()


def test_exact_abi_and_native_cancel_accounting(library):
    assert (ctypes.sizeof(Config), ctypes.sizeof(Request), ctypes.sizeof(Snapshot)) == (32, 200, 208)
    with NativeSession(*library, memory_limit_bytes=1024) as owner:
        owner.submit('fixture', '07' * 32, reserve_bytes=512, deadline_ns=1000,
                     prompt_tokens=64, output_tokens=32, context_tokens=128, now_ns=1)
        assert owner.next(2).state == 2
        owner.add_tokens('fixture', 8)
        owner.cancel('fixture')
        assert owner.totals().reserved_bytes == 512
        assert owner.totals().active == 1
        with pytest.raises(NativeSessionError) as failure:
            owner.close()
        assert failure.value.code == 5
        with pytest.raises(NativeSessionError):
            owner.add_tokens('fixture', 1)
        owner.finish('fixture', 6, 4)
        assert owner.read('fixture').completion_tokens == 8
        assert owner.read('fixture').native_runtime_qualified == 0
        assert owner.totals().reserved_bytes == 0
        with pytest.raises(NativeSessionError) as failure:
            owner.finish('fixture', 6, 4)
        assert failure.value.code == 6


def test_invalid_wrapped_inputs_never_truncate(library):
    with pytest.raises(ValueError, match='hash mismatch'):
        NativeSession(library[0], '0' * 64, memory_limit_bytes=1000)
    with pytest.raises(ValueError, match='digest required'):
        NativeSession(library[0], None, memory_limit_bytes=1000)
    with pytest.raises(ValueError):
        NativeSession(*library, memory_limit_bytes=True)
    with NativeSession(*library, memory_limit_bytes=1000) as owner:
        for changes in ({'reserve_bytes': -1}, {'output_tokens': 2**32}, {'context_tokens': True},
                        {'identifier': 'x\0y'}, {'identifier': 'x' * 128}, {'digest': 'wrong'}):
            values = {'identifier': 'fixture', 'digest': '07' * 32, 'reserve_bytes': 100,
                      'deadline_ns': 1000, 'prompt_tokens': 64, 'output_tokens': 32,
                      'context_tokens': 128, 'now_ns': 1, **changes}
            with pytest.raises(ValueError):
                owner.submit(**values)
        assert owner.totals().records == 0


def test_swift_linked_native_binding_uses_same_abi(library, tmp_path):
    executable = tmp_path.resolve() / 'swift-native-session'
    subprocess.run(['xcrun', 'swiftc', '-swift-version', '6', '-I', str(ROOT / 'native/runtime'),
                    str(ROOT / 'integrations/commoncompute/native-session-v1/StrataSessionOwner.swift'),
                    str(ROOT / 'integrations/commoncompute/native-session-v1/VerifySession.swift'),
                    str(library[0]), '-o', str(executable)], check=True, capture_output=True, timeout=120)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=20)
    assert 'Swift/native lifecycle parity passed' in result.stdout


def test_typescript_consumes_real_native_snapshots(library, tmp_path):
    with NativeSession(*library, memory_limit_bytes=1024) as owner:
        owner.submit('fixture', '07' * 32, reserve_bytes=512, deadline_ns=1000,
                     prompt_tokens=64, output_tokens=32, context_tokens=128, now_ns=1)
        rows = [owner.read('fixture').diagnostic(), owner.next(2).diagnostic()]
        owner.add_tokens('fixture', 8)
        rows.append(owner.read('fixture').diagnostic())
        owner.cancel('fixture')
        rows.append(owner.read('fixture').diagnostic())
        owner.finish('fixture', 6, 4)
        rows.append(owner.read('fixture').diagnostic())
    fixture = tmp_path.resolve() / 'native-snapshots.json'
    fixture.write_text(json.dumps(rows))
    result = subprocess.run(['node', '--experimental-strip-types',
                             str(ROOT / 'integrations/commoncompute/native-session-v1/verify-diagnostic.ts'),
                             str(fixture)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'TypeScript/native diagnostic parity passed' in result.stdout
