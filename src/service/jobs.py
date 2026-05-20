"""Job registry, FIFO queue, background-thread worker, and schema factories.

This module owns the in-memory job/batch state and the background worker
that processes jobs through the warm pipeline. The schema factories
(:func:`build_combined`, :func:`build_summary`) also live here because
they are pure assembly functions tightly coupled to this module's worker.

The dataclass + queue + worker implementations land in Phase 5; this module
currently exposes only the factories so Phase 3's schema work can be wired
up end-to-end ahead of the worker.
"""

from __future__ import annotations

from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from src.models.diarization import DiarizationResult
from src.models.sentiment import SentimentResult
from src.models.transcription import TranscriptionResult
from src.service.schemas import (
    BatchSummary,
    BatchTotals,
    CombinedTranscript,
    JobSummaryEntry,
)


def _audio_refinery_version() -> str:
    """Return the installed audio-refinery package version, or 'unknown' if not packaged."""
    try:
        return _pkg_version("audio-refinery")
    except PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------
# Schema factories
# ---------------------------------------------------------------------------


def build_combined(
    diarization: DiarizationResult,
    transcription: TranscriptionResult,
    sentiment: SentimentResult | None = None,
    *,
    processed_at: datetime | None = None,
) -> CombinedTranscript:
    """Assemble a ``CombinedTranscript`` from the existing per-stage results.

    Args:
        diarization: The diarization stage output. Its ``input_info`` is
            used as the canonical audio metadata for the combined document
            (all three stages process the same file, so any of their
            ``input_info`` fields would do).
        transcription: The transcription stage output.
        sentiment: The sentiment stage output, when enabled.
        processed_at: Override for the assembly timestamp. Useful in tests;
            defaults to ``datetime.now()`` when omitted.

    Returns:
        A populated ``CombinedTranscript`` ready to JSON-serialize and upload.
    """
    model_versions = {
        "diarization": diarization.model_name,
        "transcription": transcription.model_name,
    }
    if sentiment is not None:
        model_versions["sentiment"] = sentiment.model_name

    return CombinedTranscript(
        audio_refinery_version=_audio_refinery_version(),
        processed_at=processed_at if processed_at is not None else datetime.now(),
        audio=diarization.input_info,
        diarization=diarization,
        transcription=transcription,
        sentiment=sentiment,
        model_versions=model_versions,
    )


def build_summary(
    batch_id: str,
    submitted_at: datetime,
    completed_at: datetime,
    jobs: list[JobSummaryEntry],
) -> BatchSummary:
    """Assemble a ``BatchSummary`` from terminal per-job records.

    Callers are responsible for constructing ``JobSummaryEntry`` instances
    from their internal job state. Totals are derived from the entries to
    avoid the caller having to track them separately.

    Args:
        batch_id: The batch identifier returned to the caller in the
            ``POST /transcribe`` response (e.g. ``"btc_..."``).
        submitted_at: When the batch was accepted by the service.
        completed_at: When the final job in the batch reached a terminal
            state.
        jobs: Per-job terminal entries in submission order.

    Returns:
        A populated ``BatchSummary`` ready to JSON-serialize and upload.
    """
    completed = sum(1 for j in jobs if j.status == "completed")
    failed = sum(1 for j in jobs if j.status == "failed")
    return BatchSummary(
        batch_id=batch_id,
        submitted_at=submitted_at,
        completed_at=completed_at,
        jobs=jobs,
        totals=BatchTotals(submitted=len(jobs), completed=completed, failed=failed),
    )


__all__ = [
    "build_combined",
    "build_summary",
]
