"""Read-only bounded parser of generated CommonCompute catalog data, never JS.

Selects the lightest estimated artifact per (kind, architecture), not the
smallest parameter release on the entire Hub. Capacity is planning, not admission.
"""
import hashlib
import math
from pathlib import Path
import re
import stat

from .families import CANARIES, GIB, plan
from ..system import real_path

ROW = re.compile(
    r'\s*\{ id: "([a-z0-9.-]+)", repo: "([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)", '
    r'revision: "([0-9a-f]{40})", kind: \'(llm|vlm)\', modelType: "([a-z0-9_-]+)", '
    r'minMemoryGb: ([0-9]+), weightsGb: ([0-9]+(?:\.[0-9]+)?), loadable: (true|false) \},\s*')


def coverage(text, memory_bytes):
    plan(memory_bytes)  # Use the same strict hardware input contract.
    if not isinstance(text, str) or len(text.encode()) > 512 * 1024:
        raise ValueError('catalog must be bounded UTF-8 generated data')
    rows, ids = [], set()
    for line in text.splitlines():
        if not re.match(r'\s*\{\s*id\s*:', line):
            continue
        match = ROW.fullmatch(line)
        if not match:
            raise ValueError('unrecognized generated catalog row; never evaluate TypeScript')
        identifier, repo, revision, kind, architecture, memory, weights, loadable = match.groups()
        memory, weights = int(memory), float(weights)
        if identifier in ids or not 0 < memory <= 1024 or not math.isfinite(weights) or not 0 < weights <= 10000:
            raise ValueError('duplicate or invalid catalog row')
        ids.add(identifier)
        rows.append({'id': identifier, 'repository': repo, 'revision': revision,
                     'kind': kind, 'architecture': architecture,
                     'catalog_memory_tier_gib': memory, 'catalog_weights_gb_estimate': weights,
                     'catalog_loadable': loadable == 'true'})
    if not rows or len(rows) > 500:
        raise ValueError('empty or oversized catalog inventory')
    by_architecture = {}
    for row in rows:
        key = row['kind'], row['architecture']
        old = by_architecture.get(key)
        if old is None or (row['catalog_weights_gb_estimate'], row['id']) < (old['catalog_weights_gb_estimate'], old['id']):
            by_architecture[key] = row
    selected = []
    for key in sorted(by_architecture):
        row = by_architecture[key]
        fits = memory_bytes >= row['catalog_memory_tier_gib'] * GIB
        selected.append({**row, 'status': 'UNSUPPORTED_IN_CATALOG' if not row['catalog_loadable'] else
                         'LOCAL_CAPACITY_CANDIDATE' if fits else 'FLEET_OR_COMPONENT_ONLY',
                         'miki_execution_qualified': False, 'performance_qualified': False,
                         'artifact_and_live_admission_required': True})
    findings = []
    for row in rows:
        for canary in CANARIES:
            if (row['repository'], row['revision']) != (canary.repository, canary.revision):
                continue
            actual = canary.verified_remote_weight_bytes
            if actual and abs(row['catalog_weights_gb_estimate'] * 10**9 - actual) / actual > .05:
                findings.append({'id': row['id'], 'repository': row['repository'], 'revision': row['revision'],
                                 'catalog_weights_gb_estimate': row['catalog_weights_gb_estimate'],
                                 'verified_remote_weight_bytes': actual,
                                 'action': 'do_not_admit_from_catalog_weight_estimate; catalog_not_modified'})
    return {'schema_version': 1, 'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'selection': 'lightest catalog weight estimate per kind and model_type; not latest or Hub-wide minimum',
            'catalog_rows': len(rows), 'architecture_lanes': len(selected), 'lanes': selected,
            'sizing_findings': findings, 'download_performed': False, 'production_promotion_eligible': False}


def read_coverage(path, memory_bytes):
    path = real_path(Path(path))
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 512 * 1024:
        raise ValueError('catalog must be a bounded nonsymlink regular file')
    return coverage(path.read_text(encoding='utf-8'), memory_bytes)
