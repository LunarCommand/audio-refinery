"""Combined transcript schema (v1.0.0) and batch summary schema (v1.0.0).

These are **service-mode** documents — they envelope the existing per-stage
pipeline outputs so callers receive one transcript per successful job and one
summary per accepted batch. The schemas live here rather than in
``src/models/`` because they are a service-layer concern, not a stage output;
the CLI continues to emit the per-stage JSONs directly.

Schema versioning is independent per document type. Both schemas start at
v1.0.0 in this release. They are expected to bump (the transcript shape will
change when v0.3.0 alignment lands and splits aligned words into a separate
array). Consumers MUST key off ``schema_version`` to handle evolution.
"""

from __future__ import annotations

from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Literal

from pydantic import BaseModel, Field

from src.models.audio import AudioFileInfo
from src.models.diarization import DiarizationResult
from src.models.sentiment import SentimentResult
from src.models.transcription import TranscriptionResult

TRANSCRIPT_SCHEMA_VERSION = "1.0.0"
BATCH_SUMMARY_SCHEMA_VERSION = "1.0.0"


def _audio_refinery_version() -> str:
    """Return the installed audio-refinery package version, or 'unknown' if not packaged."""
    try:
        return _pkg_version("audio-refinery")
    except PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------
# CombinedTranscript — one document per successful job, uploaded to output_uri
# ---------------------------------------------------------------------------


class CombinedTranscript(BaseModel):
    """One combined-output JSON per successful job.

    Wraps the existing per-stage Pydantic models so the consuming analysis
    pipeline reads a single document instead of stitching diarization +
    transcription + sentiment together itself.
    """

    schema_version: str = Field(default=TRANSCRIPT_SCHEMA_VERSION, description="Semver of the transcript schema.")
    audio_refinery_version: str = Field(description="Version of audio-refinery that produced this document.")
    processed_at: datetime = Field(description="When this combined document was assembled (UTC).")
    audio: AudioFileInfo = Field(description="Input audio metadata.")
    diarization: DiarizationResult = Field(description="Full pyannote diarization result.")
    transcription: TranscriptionResult = Field(description="Full WhisperX transcription result.")
    sentiment: SentimentResult | None = Field(default=None, description="Sentiment result, present only when enabled.")
    model_versions: dict[str, str] = Field(description="Per-stage model identifiers for quick lookup.")


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


# ---------------------------------------------------------------------------
# JobSummaryEntry + BatchSummary — one document per accepted batch,
# uploaded to summary_uri after every job in the batch has settled.
# ---------------------------------------------------------------------------


JobStatus = Literal["completed", "failed"]
"""Terminal per-job status. Non-terminal states (``queued``, ``processing``)
never appear in the summary — by the time the summary is written, every job
has reached one of these two outcomes."""


JobFailureStage = Literal["download", "transcribe", "upload", "thermal_shutdown"]
"""Pipeline stage at which a failure occurred. Used only on failed entries."""


class JobSummaryEntry(BaseModel):
    """One per-job record inside a ``BatchSummary``.

    The ``status`` field discriminates between success and failure shapes.
    On ``completed``, ``completed_at`` and ``duration_seconds`` are populated
    and ``failed_at``/``stage``/``error``/``retryable`` are None. On ``failed``,
    the reverse holds.
    """

    job_id: str
    input_uri: str
    output_uri: str
    status: JobStatus
    started_at: datetime

    # Populated on status == "completed"
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    # Populated on status == "failed"
    failed_at: datetime | None = None
    stage: JobFailureStage | None = None
    error: str | None = None
    retryable: bool | None = None


class BatchTotals(BaseModel):
    """Per-batch aggregate counts. Always sums to ``submitted``."""

    submitted: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)


class BatchSummary(BaseModel):
    """One summary JSON per accepted batch.

    Written to the caller-supplied ``summary_uri`` after every job in the
    batch has reached a terminal state. The canonical surface for the
    orchestrator to discover batch outcomes — successful job transcripts are
    already on the analysis pipeline's wake signal by the time this is
    written.
    """

    schema_version: str = Field(default=BATCH_SUMMARY_SCHEMA_VERSION, description="Semver of the summary schema.")
    batch_id: str
    submitted_at: datetime
    completed_at: datetime
    jobs: list[JobSummaryEntry]
    totals: BatchTotals


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
    "BATCH_SUMMARY_SCHEMA_VERSION",
    "TRANSCRIPT_SCHEMA_VERSION",
    "BatchSummary",
    "BatchTotals",
    "CombinedTranscript",
    "JobFailureStage",
    "JobStatus",
    "JobSummaryEntry",
    "build_combined",
    "build_summary",
]
