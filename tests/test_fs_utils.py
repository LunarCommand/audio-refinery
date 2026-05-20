"""Tests for `src.fs_utils`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.fs_utils import detect_fstype


def test_detect_fstype_reads_proc_mounts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """On Linux, /proc/mounts is the canonical filesystem-type source. We can't
    actually mount tmpfs in a test, but we can prove the parser finds the
    deepest matching mountpoint."""
    fake_mounts = (
        f"tmpfs / tmpfs rw 0 0\noverlay {tmp_path} overlay rw 0 0\ntmpfs {tmp_path / 'scratch'} tmpfs rw,size=4G 0 0\n"
    )
    fake_proc = tmp_path / "fake_proc_mounts"
    fake_proc.write_text(fake_mounts)

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/mounts":
            return real_open(fake_proc, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # Deepest match for scratch should be the tmpfs entry.
    assert detect_fstype(scratch) == "tmpfs"
    # A path inside scratch also resolves to tmpfs.
    inner = scratch / "subdir"
    inner.mkdir()
    assert detect_fstype(inner) == "tmpfs"
    # A path outside scratch falls back to the overlay entry.
    other = tmp_path / "other"
    other.mkdir()
    assert detect_fstype(other) == "overlay"


def test_detect_fstype_returns_none_when_proc_mounts_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/mounts":
            raise OSError("not Linux")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    assert detect_fstype(tmp_path) is None
