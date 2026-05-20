"""Filesystem utilities — small helpers used by both CLI and service modes."""

from __future__ import annotations

from pathlib import Path


def detect_fstype(path: Path) -> str | None:
    """Return the filesystem type of ``path`` on Linux, or None when undetectable.

    Walks /proc/mounts and matches the deepest mount-point prefix of ``path``.
    Returns the fstype string (e.g. ``"tmpfs"``, ``"ext4"``, ``"overlay"``).
    Returns ``None`` on non-Linux, when /proc/mounts is unreadable, or when
    the path's filesystem doesn't appear in /proc/mounts.

    Used by both modes to label Demucs scratch paths as RAM-backed vs.
    disk-backed in user-facing output and structured logs.
    """
    try:
        with open("/proc/mounts") as f:
            entries = [line.split() for line in f if line.strip()]
    except OSError:
        return None

    try:
        target = path.resolve()
    except OSError:
        return None

    best: tuple[Path, str] | None = None
    for parts in entries:
        if len(parts) < 3:
            continue
        try:
            mountpoint = Path(parts[1]).resolve()
        except OSError:
            continue
        fstype = parts[2]
        is_match = mountpoint == target or mountpoint in target.parents
        is_deeper = best is None or len(str(mountpoint)) > len(str(best[0]))
        if is_match and is_deeper:
            best = (mountpoint, fstype)
    return best[1] if best is not None else None


__all__ = ["detect_fstype"]
