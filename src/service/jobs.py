"""Job registry, FIFO queue, background-thread worker, and schema factories.

This module owns the in-memory job/batch state and the background worker
that processes jobs through the warm pipeline. The schema factories
(:func:`build_combined`, :func:`build_summary`) also live here because
they are pure assembly functions tightly coupled to this module's worker.
"""

from __future__ import annotations

import queue
import secrets
import threading
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# In-memory state — Job and Batch records, registries, queue
# ---------------------------------------------------------------------------

JobStatus = "queued | processing | completed | failed"  # documentation alias only


def make_job_id() -> str:
    """Return a fresh ``rfj_<16-hex>`` identifier."""
    return f"rfj_{secrets.token_hex(8)}"


def make_batch_id() -> str:
    """Return a fresh ``btc_<16-hex>`` identifier."""
    return f"btc_{secrets.token_hex(8)}"


@dataclass
class Job:
    """Per-job in-memory record. The worker mutates this through registries."""

    job_id: str
    batch_id: str
    input_uri: str
    output_uri: str
    status: str = "queued"  # queued | processing | completed | failed
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    stage: str | None = None  # set on failure
    error: str | None = None  # set on failure
    retryable: bool | None = None  # set on failure
    duration_seconds: float | None = None  # set on success


@dataclass
class Batch:
    """Per-batch in-memory record. ``pending_count`` reaches 0 when every job
    has settled, which is the worker's cue to call ``finalize_batch``."""

    batch_id: str
    summary_uri: str
    job_ids: list[str]
    submitted_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    pending_count: int = 0  # initialized to len(job_ids) at construction


class JobRegistry:
    """Thread-safe in-memory dict of ``Job`` records.

    Reads and writes serialize through a single lock. The hot path
    (``mark_terminal``) is intentionally co-located with the batch registry's
    pending-count decrement to keep the per-batch state machine atomic.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def add(self, job: Job) -> None:
        with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"job_id {job.job_id!r} already registered")
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job  # dataclass; caller treats as read-only snapshot

    def update(self, job_id: str, **fields: object) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            for name, value in fields.items():
                setattr(job, name, value)
            return job

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def all_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def __contains__(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs


class BatchRegistry:
    """Thread-safe in-memory dict of ``Batch`` records.

    ``decrement_pending`` returns the new ``pending_count`` so the worker can
    detect the transition to zero without a separate read.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._batches: dict[str, Batch] = {}

    def add(self, batch: Batch) -> None:
        with self._lock:
            if batch.batch_id in self._batches:
                raise ValueError(f"batch_id {batch.batch_id!r} already registered")
            batch.pending_count = len(batch.job_ids)
            self._batches[batch.batch_id] = batch

    def get(self, batch_id: str) -> Batch | None:
        with self._lock:
            return self._batches.get(batch_id)

    def decrement_pending(self, batch_id: str) -> int:
        """Atomically decrement ``pending_count`` and return its new value.

        Raises KeyError if the batch is unknown. Never decrements below zero
        — a guard against double-decrement bugs in the worker.
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise KeyError(batch_id)
            if batch.pending_count > 0:
                batch.pending_count -= 1
            return batch.pending_count

    def mark_completed(self, batch_id: str, completed_at: datetime) -> None:
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise KeyError(batch_id)
            batch.completed_at = completed_at

    def delete(self, batch_id: str) -> bool:
        with self._lock:
            return self._batches.pop(batch_id, None) is not None

    def all_batches(self) -> list[Batch]:
        with self._lock:
            return list(self._batches.values())


class JobQueue:
    """FIFO bounded queue wrapping ``queue.Queue``.

    The HTTP endpoint puts job_ids onto the queue with ``put_nowait``; the
    worker pulls with a blocking ``get``. An over-limit ``put_nowait`` raises
    ``queue.Full``, which the endpoint translates to ``429 Too Many Requests``.
    """

    def __init__(self, maxsize: int = 100) -> None:
        self._q: queue.Queue[str] = queue.Queue(maxsize=maxsize)

    @property
    def maxsize(self) -> int:
        return self._q.maxsize

    def qsize(self) -> int:
        return self._q.qsize()

    def put_nowait(self, job_id: str) -> None:
        """Enqueue a job_id; raises ``queue.Full`` on over-limit."""
        self._q.put_nowait(job_id)

    def get(self, timeout: float | None = None) -> str | None:
        """Block until a job_id is available, or return None on timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


@dataclass
class Registries:
    """Bundle of in-memory state passed to the worker and the HTTP endpoints."""

    jobs: JobRegistry = field(default_factory=JobRegistry)
    batches: BatchRegistry = field(default_factory=BatchRegistry)
    queue: JobQueue = field(default_factory=JobQueue)


__all__ = [
    "Batch",
    "BatchRegistry",
    "Job",
    "JobQueue",
    "JobRegistry",
    "Registries",
    "build_combined",
    "build_summary",
    "make_batch_id",
    "make_job_id",
]
