"""Thin trusted-host ctypes wrapper: lifecycle math runs in the native C ABI.

Not a model runner. No remote input, plaintext, weights, keys or filesystem
authority belongs in a session descriptor. Caller owns signature verification,
complete model demand, GPU synchronization and joined teardown.
"""
import ctypes as C
import hashlib
from pathlib import Path
import re
import threading

from lokahi.lab.system import real_path


class NativeSessionError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(f'native session status {code}')


class Config(C.Structure):
    _fields_ = [('abi_version', C.c_uint32), ('struct_bytes', C.c_uint32),
                ('memory_limit_bytes', C.c_uint64), ('max_queued', C.c_uint32),
                ('max_active', C.c_uint32), ('max_records', C.c_uint32), ('reserved', C.c_uint32)]


class Request(C.Structure):
    _fields_ = [('abi_version', C.c_uint32), ('struct_bytes', C.c_uint32),
                ('request_id', C.c_char * 128), ('model_digest', C.c_uint8 * 32),
                ('reserve_bytes', C.c_uint64), ('deadline_monotonic_ns', C.c_uint64),
                ('prompt_tokens', C.c_uint32), ('max_output_tokens', C.c_uint32),
                ('context_tokens', C.c_uint32), ('reserved', C.c_uint32)]


class Snapshot(C.Structure):
    _fields_ = [('abi_version', C.c_uint32), ('struct_bytes', C.c_uint32),
                ('request_id', C.c_char * 128), ('model_digest', C.c_uint8 * 32),
                ('reserved_bytes', C.c_uint64), ('revision', C.c_uint64),
                ('state', C.c_uint32), ('reason', C.c_uint32), ('prompt_tokens', C.c_uint32),
                ('completion_tokens', C.c_uint32), ('max_output_tokens', C.c_uint32),
                ('native_runtime_qualified', C.c_uint32)]

    def diagnostic(self):
        """Lossless metadata-only JSON; uint64 values are decimal strings."""
        return {'schema_version': 1, 'request_id': self.request_id.decode('ascii'),
                'model_digest': bytes(self.model_digest).hex(),
                'reserved_bytes': str(self.reserved_bytes), 'revision': str(self.revision),
                'state': self.state, 'reason': self.reason, 'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens, 'max_output_tokens': self.max_output_tokens,
                'native_runtime_qualified': bool(self.native_runtime_qualified)}


class Totals(C.Structure):
    _fields_ = [('memory_limit_bytes', C.c_uint64), ('reserved_bytes', C.c_uint64),
                ('queued', C.c_uint32), ('active', C.c_uint32), ('records', C.c_uint32), ('draining', C.c_uint32)]


def _integer(value, bits):
    if type(value) is not int or not 0 <= value < 2**bits:
        raise ValueError('native integer outside its exact ABI range')
    return value


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,126}', value):
        raise ValueError('invalid opaque request identifier')
    return value.encode('ascii')


class NativeSession:
    def __init__(self, library, expected_sha256, *, memory_limit_bytes, max_queued=4, max_records=32):
        path = real_path(Path(library))
        if not isinstance(expected_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
            raise ValueError('trusted native library digest required')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError('native library hash mismatch')
        self.library = C.CDLL(str(path))
        self.lock = threading.RLock()
        self.handle = C.c_void_p()
        signatures = {
            'create_v1': [C.POINTER(Config), C.POINTER(C.c_void_p)],
            'destroy_v1': [C.POINTER(C.c_void_p)],
            'submit_v1': [C.c_void_p, C.POINTER(Request), C.c_uint64],
            'next_v1': [C.c_void_p, C.c_uint64, C.POINTER(Snapshot)],
            'read_v1': [C.c_void_p, C.c_char_p, C.POINTER(Snapshot)],
            'add_tokens_v1': [C.c_void_p, C.c_char_p, C.c_uint32],
            'cancel_v1': [C.c_void_p, C.c_char_p],
            'check_v1': [C.c_void_p, C.c_char_p, C.c_uint64],
            'finish_v1': [C.c_void_p, C.c_char_p, C.c_uint32, C.c_uint32],
            'forget_v1': [C.c_void_p, C.c_char_p],
            'drain_v1': [C.c_void_p],
            'totals_v1_read': [C.c_void_p, C.POINTER(Totals)],
        }
        for name, arguments in signatures.items():
            function = getattr(self.library, 'strata_session_' + name)
            function.argtypes, function.restype = arguments, C.c_int
        config = Config(1, C.sizeof(Config), _integer(memory_limit_bytes, 64),
                        _integer(max_queued, 32), 1, _integer(max_records, 32), 0)
        code = self.library.strata_session_create_v1(C.byref(config), C.byref(self.handle))
        if code:
            raise NativeSessionError(code)

    def _call(self, name, *arguments):
        with self.lock:
            if not self.handle.value:
                raise ValueError('native session already closed')
            code = getattr(self.library, 'strata_session_' + name)(self.handle, *arguments)
            if code:
                raise NativeSessionError(code)

    def submit(self, identifier, digest, *, reserve_bytes, deadline_ns, prompt_tokens, output_tokens, context_tokens, now_ns):
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('exact model digest required')
        request = Request(1, C.sizeof(Request), _id(identifier), (C.c_uint8 * 32).from_buffer_copy(bytes.fromhex(digest)),
                          _integer(reserve_bytes, 64), _integer(deadline_ns, 64), _integer(prompt_tokens, 32),
                          _integer(output_tokens, 32), _integer(context_tokens, 32), 0)
        self._call('submit_v1', C.byref(request), _integer(now_ns, 64))

    def next(self, now_ns):
        value = Snapshot()
        self._call('next_v1', _integer(now_ns, 64), C.byref(value))
        return value

    def read(self, identifier):
        value = Snapshot()
        self._call('read_v1', _id(identifier), C.byref(value))
        return value

    def add_tokens(self, identifier, delta):
        self._call('add_tokens_v1', _id(identifier), _integer(delta, 32))

    def cancel(self, identifier):
        self._call('cancel_v1', _id(identifier))

    def check(self, identifier, now_ns):
        self._call('check_v1', _id(identifier), _integer(now_ns, 64))

    def finish(self, identifier, state, reason):
        self._call('finish_v1', _id(identifier), _integer(state, 32), _integer(reason, 32))

    def forget(self, identifier):
        self._call('forget_v1', _id(identifier))

    def drain(self):
        self._call('drain_v1')

    def totals(self):
        value = Totals()
        self._call('totals_v1_read', C.byref(value))
        return value

    def close(self):
        with self.lock:
            if not self.handle.value:
                return
            code = self.library.strata_session_destroy_v1(C.byref(self.handle))
            if code:
                raise NativeSessionError(code)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
