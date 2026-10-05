"""Tiny hash/provenance fixtures; never actual model or fit evidence."""
import hashlib

import pytest

from benchmarks.recovered_mlx_baseline import verify_snapshot

REPO = 'mlx-community/Tiny'
REVISION = 'a' * 40


def fixture(path):
    provenance = path / '.cache/huggingface/download'
    provenance.mkdir(parents=True)
    data = b'not-real-safetensors; content-verification-fixture-only'
    weights = path / 'model.safetensors'
    weights.write_bytes(data)
    metadata = provenance / (weights.name + '.metadata')
    metadata.write_text(REVISION + '\n' + hashlib.sha256(data).hexdigest() + '\n0\n')
    return weights, metadata


def test_cold_content_check_is_not_ssd_admission(tmp_path):
    weights, _ = fixture(tmp_path)
    path, result = verify_snapshot(tmp_path, REPO, REVISION)
    assert path == tmp_path
    assert result['files'][0]['bytes'] == weights.stat().st_size
    assert result['repository'] == REPO
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_snapshot(tmp_path, REPO, REVISION)


def test_provenance_pin_and_links_fail_closed(tmp_path):
    _, metadata = fixture(tmp_path)
    with pytest.raises(ValueError, match='unpinned'):
        verify_snapshot(tmp_path, REPO, 'b' * 40)
    moved = metadata.with_suffix('.original')
    metadata.rename(moved)
    metadata.symlink_to(moved)
    with pytest.raises(ValueError, match='symlink'):
        verify_snapshot(tmp_path, REPO, REVISION)


@pytest.mark.parametrize('repo,revision', [(REPO, 'main'), ('../Tiny', REVISION)])
def test_mutable_or_invalid_repo_pin_rejected(tmp_path, repo, revision):
    fixture(tmp_path)
    with pytest.raises(ValueError, match='immutable'):
        verify_snapshot(tmp_path, repo, revision)
