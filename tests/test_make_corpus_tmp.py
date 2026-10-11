"""Offline EXDEV / staging tests for scripts/make_corpus.download_shard.

No network: urlopen + CDN resolve are mocked; the tests simulate a
cross-filesystem publish (os.replace raising EXDEV) and assert the
.part file is staged in tmp_dir (same fs as dest) with a shutil.move
fallback.
"""

import errno
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import scripts.make_corpus as mc


class _FakeResp:
    status = 200

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        if self._pos >= len(self._data):
            return b""
        chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk


def _mock_net(monkeypatch, payload: bytes = b"x" * 64):
    monkeypatch.setattr(mc, "_resolve_cdn", lambda url: url)
    monkeypatch.setattr(mc.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp(payload))


def test_part_staged_in_tmp_dir_not_part_dir(tmp_path, monkeypatch):
    """part_dir on another 'filesystem' must not receive the .part file."""
    _mock_net(monkeypatch)
    tmp_dir = str(tmp_path / "tmpfs")
    part_dir = str(tmp_path / "outfs")
    os.makedirs(tmp_dir)
    os.makedirs(part_dir)
    name, dest = mc.download_shard(7, tmp_dir, part_dir=part_dir)
    assert dest == os.path.join(tmp_dir, name)
    assert os.path.exists(dest)
    # No .part residue anywhere; nothing staged in part_dir.
    assert os.listdir(part_dir) == []
    assert not os.path.exists(os.path.join(tmp_dir, name + ".part"))


def test_exdev_falls_back_to_shutil_move(tmp_path, monkeypatch):
    """os.replace raising EXDEV still publishes via shutil.move."""
    _mock_net(monkeypatch, payload=b"y" * 32)
    tmp_dir = str(tmp_path / "tmp")
    os.makedirs(tmp_dir)
    real_replace = os.replace

    def fake_replace(src, dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "replace", fake_replace)
    # mc references os.replace via the os module attribute.
    monkeypatch.setattr(mc.os, "replace", fake_replace)
    name, dest = mc.download_shard(9, tmp_dir)
    assert os.path.exists(dest)
    assert open(dest, "rb").read() == b"y" * 32
    monkeypatch.setattr(mc.os, "replace", real_replace)
