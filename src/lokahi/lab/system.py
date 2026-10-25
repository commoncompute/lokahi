"""Non-identifying hardware class and fail-closed local experiment admission."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys

RESERVE = 40 * 1024**3


def command(args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def real_path(path: Path, *, exists: bool = True) -> Path:
    path = path.absolute()
    if '..' in path.parts:
        raise ValueError('experiment paths must not contain parent traversal')
    for ancestor in (*reversed(path.parents), path):
        if ancestor == Path('/Volumes/Tyler HDD'):
            raise ValueError('the failed HDD is forbidden')
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            if exists:
                raise
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError('experiment paths must not contain symlinks')
    return path


def system_info():
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        raise ValueError('physical Apple-Silicon macOS is required; no simulated M1')
    info = {
        'chip': command(['sysctl', '-n', 'machdep.cpu.brand_string']),
        'memory_bytes': int(command(['sysctl', '-n', 'hw.memsize'])),
        'cpu_cores': int(command(['sysctl', '-n', 'hw.ncpu'])),
        'performance_cores': int(command(['sysctl', '-n', 'hw.perflevel0.physicalcpu'])),
        'efficiency_cores': int(command(['sysctl', '-n', 'hw.perflevel1.physicalcpu'])),
        'os_version': command(['sw_vers', '-productVersion']),
        'os_build': command(['sw_vers', '-buildVersion']),
        'machine': platform.machine(),
        'compiler': command(['xcrun', 'clang', '--version']).splitlines()[0],
        'xcode': command(['xcodebuild', '-version']).splitlines(),
        'python_version': platform.python_version(),
        'software': {},
        'gpu': None,
        'evidence_kind': 'physical_hardware_class',
    }
    for name in ('mlx', 'mlx-lm', 'numpy', 'transformers', 'tokenizers', 'safetensors'):
        try:
            info['software'][name] = version(name)
        except PackageNotFoundError:
            info['software'][name] = None
    # Device discovery is not kernel execution or an ANE capability claim.
    import mlx.core as mx
    device = mx.device_info()
    info['gpu'] = {key: device.get(key) for key in (
        'device_name', 'architecture', 'memory_size',
        'max_recommended_working_set_size', 'max_buffer_length')}
    encoded = json.dumps(info, sort_keys=True, separators=(',', ':')).encode()
    return {**info, 'class_fingerprint': hashlib.sha256(encoded).hexdigest()}


def admission():
    home = real_path(Path.home())
    free = shutil.disk_usage(home).free
    if free < RESERVE:
        raise ValueError(f'internal SSD reserve below 40 GiB: {free} < {RESERVE} bytes')
    pressure = command(['memory_pressure', '-Q'])
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', pressure)
    if not match or int(match[1]) < 20:
        raise ValueError('current memory pressure is unsafe or unavailable')
    return {'free_internal_ssd_bytes': free, 'memory_free_percent': int(match[1]),
            'swap': command(['sysctl', '-n', 'vm.swapusage']),
            'thermal': command(['pmset', '-g', 'therm'])}
