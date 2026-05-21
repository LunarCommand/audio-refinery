"""Combined transcript schema (v1.0.0) and batch summary schema (v1.0.0).

These are **service-mode** documents — they envelope the existing per-stage
pipeline outputs so callers receive one transcript per successful job and one
summary per accepted batch. The schemas live here rather than in
``src/models/`` because they are a service-layer concern, not a stage output;
the CLI continues to emit the per-stage JSONs directly.

Pure data — no behavior. Factories that assemble these documents from the
existing per-stage results live in :mod:`src.service.jobs` alongside the
worker that calls them, matching the project convention that data models
are separate from the modules that use them.

Schema versioning is independent per document type. Both schemas start at
v1.0.0 in this release. They are expected to bump (the transcript shape will
change when v0.3.0 alignment lands and splits aligned words into a separate
array). Consumers MUST key off ``schema_version`` to handle evolution.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.models.audio import AudioFileInfo
from src.models.diarization import DiarizationResult
from src.models.sentiment import SentimentResult
from src.models.transcription import TranscriptionResult

TRANSCRIPT_SCHEMA_VERSION = "1.0.0"
BATCH_SUMMARY_SCHEMA_VERSION = "1.0.0"


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


__all__ = [
    "BATCH_SUMMARY_SCHEMA_VERSION",
    "TRANSCRIPT_SCHEMA_VERSION",
    "BatchSummary",
    "BatchTotals",
    "CombinedTranscript",
    "JobFailureStage",
    "JobStatus",
    "JobSummaryEntry",
]
