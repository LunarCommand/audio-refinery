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


def notify_pipeline_complete(
    device: str,
    total: int,
    completed: int,
    failures: int,
    elapsed_seconds: float,
) -> None:
    """Send a Slack notification when a single-GPU pipeline run ends."""
    elapsed = _fmt_elapsed(elapsed_seconds)
    if failures == 0:
        icon, status = ":white_check_mark:", "Pipeline complete"
    else:
        icon, status = ":warning:", f"Pipeline complete with {failures} failure(s)"
    _send(f"{icon} *{status}* on `{device}`\n{completed}/{total} files processed in {elapsed}")


def notify_thermal_shutdown(device: str, temp: int, limit: int) -> None:
    """Send a Slack notification when the pipeline is aborted due to GPU overheating."""
    _send(f":thermometer: *Thermal shutdown* — `{device}` at {temp}\u00b0C (limit {limit}\u00b0C). Pipeline aborted.")


def notify_pipeline_parallel_complete(
    worker_statuses: list[tuple[str, str, bool]],
    total_discovered: int,
    total_processed: int,
    failures: int,
    elapsed_seconds: float,
) -> None:
    """Send a Slack notification when a multi-GPU pipeline-parallel run ends.

    Args:
        worker_statuses: One ``(label, device, ok)`` tuple per worker,
            e.g. ``[("W0", "cuda:0", True), ("W1", "cuda:1", False)]``.
        total_discovered: Total files discovered across all workers.
        total_processed: Files successfully transcribed (processed + skipped).
        failures: Total stage-level failures across all workers.
        elapsed_seconds: Wall-clock runtime in seconds.
    """
    elapsed = _fmt_elapsed(elapsed_seconds)
    all_ok = all(ok for _, _, ok in worker_statuses) and failures == 0
    icon = ":white_check_mark:" if all_ok else ":warning:"
    worker_parts = []
    for label, device, ok in worker_statuses:
        status_icon = ":white_check_mark:" if ok else ":x:"
        worker_parts.append(f"{label} (`{device}`): {status_icon}")
    lines = [
        f"{icon} *Pipeline-parallel {'complete' if all_ok else 'finished with issues'}*",
        "  |  ".join(worker_parts),
        f"{total_processed}/{total_discovered} files processed in {elapsed}",
    ]
    if failures > 0:
        lines.append(f"{failures} file(s) failed")
    _send("\n".join(lines))
