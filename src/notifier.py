"""Slack webhook notifications for pipeline events.

Set ``SLACK_WEBHOOK_URL`` in your ``.env`` file or the shell environment
to enable.  All functions are fire-and-forget — errors are silently ignored, so
a notification failure can never block or abort the pipeline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None  # type: ignore[assignment]


def _send(text: str) -> None:
    """POST a plain-text Slack message to the configured webhook URL, if any."""
    if _load_dotenv is not None:
        _load_dotenv()

    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return
    try:
        data = json.dumps({"text": text}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
    except (urllib.error.URLError, OSError, ValueError):
        pass  # Never raise — notification failure must not interrupt the pipeline


def _fmt_elapsed(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s" if mins else f"{secs}s"


_STAGE_LABELS = {
    "separation": "Vocal separation",
    "diarization": "Speaker diarization",
    "transcription": "Transcription",
    "sentiment": "Text sentiment",
}


def notify_pipeline_complete(
    device: str,
    total: int,
    completed: int,
    failures: int,
    elapsed_seconds: float,
    stages: dict[str, dict[str, int]] | None = None,
    avg_per_file_seconds: float = 0.0,
) -> None:
    """Send a Slack notification when a single-GPU pipeline run ends.

    Args:
        device: PyTorch device string, e.g. ``"cuda:0"``.
        total: Total files discovered.
        completed: Files successfully transcribed (processed + skipped).
        failures: Total stage-level failures across all stages.
        elapsed_seconds: Wall-clock runtime in seconds.
        stages: Optional per-stage counts keyed by stage name, each containing
            ``"processed"``, ``"skipped"``, and ``"failed"`` counts, e.g.
            ``{"separation": {"processed": 12, "skipped": 0, "failed": 0}, ...}``.
        avg_per_file_seconds: Average wall-clock time per file across the full pipeline.
    """
    elapsed = _fmt_elapsed(elapsed_seconds)
    if failures == 0:
        icon, status = ":white_check_mark:", "Pipeline complete"
    else:
        icon, status = ":warning:", f"Pipeline complete with {failures} failure(s)"
    avg_str = f"  ·  avg/file: {_fmt_elapsed(avg_per_file_seconds)}" if avg_per_file_seconds else ""
    lines = [
        f"{icon} *{status}* on `{device}`",
        f"{completed}/{total} files transcribed in {elapsed}{avg_str}",
    ]
    if stages:
        for stage_key, stage_label in _STAGE_LABELS.items():
            s = stages.get(stage_key)
            if s is None:
                continue
            lines.append(
                f"{stage_label}: {s.get('processed', 0)} processed, "
                f"{s.get('skipped', 0)} skipped, {s.get('failed', 0)} failed"
            )
    _send("\n".join(lines))


def notify_thermal_shutdown(device: str, temp: int, limit: int) -> None:
    """Send a Slack notification when the pipeline is aborted due to GPU overheating."""
    _send(f":thermometer: *Thermal shutdown* — `{device}` at {temp}\u00b0C (limit {limit}\u00b0C). Pipeline aborted.")


def notify_pipeline_parallel_complete(
    worker_statuses: list[tuple[str, str, int, int]],
    total_discovered: int,
    total_processed: int,
    failures: int,
    elapsed_seconds: float,
    stages: dict[str, dict[str, int]] | None = None,
    avg_per_file_seconds: float = 0.0,
) -> None:
    """Send a Slack notification when a multi-GPU pipeline-parallel run ends.

    Args:
        worker_statuses: One ``(label, device, exit_code, n_failures)`` tuple per worker,
            e.g. ``[("W0", "cuda:0", 0, 0), ("W1", "cuda:1", 1, 3)]``.
        total_discovered: Total files discovered across all workers.
        total_processed: Files successfully transcribed (processed + skipped).
        failures: Total stage-level failures across all workers.
        elapsed_seconds: Wall-clock runtime in seconds.
        stages: Optional aggregated per-stage counts keyed by stage name, each
            containing ``"processed"``, ``"skipped"``, and ``"failed"`` counts.
        avg_per_file_seconds: Combined average wall-clock time per file.
    """
    elapsed = _fmt_elapsed(elapsed_seconds)
    all_ok = all(rc == 0 for _, _, rc, _ in worker_statuses) and failures == 0
    icon = ":white_check_mark:" if all_ok else ":warning:"
    worker_parts = []
    for label, device, rc, n_fail in worker_statuses:
        if rc == 0:
            worker_icon = ":white_check_mark:"
        elif n_fail > 0:
            worker_icon = f":warning: ({n_fail} failure{'s' if n_fail != 1 else ''})"
        else:
            worker_icon = f":x: (exit {rc})"
        worker_parts.append(f"{label} (`{device}`): {worker_icon}")
    avg_str = f"  ·  avg/file: {_fmt_elapsed(avg_per_file_seconds)}" if avg_per_file_seconds else ""
    lines = [
        f"{icon} *Pipeline-parallel {'complete' if all_ok else 'finished with issues'}*",
        "  |  ".join(worker_parts),
        f"{total_processed}/{total_discovered} files transcribed in {elapsed}{avg_str}",
    ]
    if stages:
        for stage_key, stage_label in _STAGE_LABELS.items():
            s = stages.get(stage_key)
            if s is None:
                continue
            lines.append(
                f"{stage_label}: {s.get('processed', 0)} processed, "
                f"{s.get('skipped', 0)} skipped, {s.get('failed', 0)} failed"
            )
    _send("\n".join(lines))
