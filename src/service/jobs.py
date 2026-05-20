"""Job registry, FIFO queue, background-thread worker, and schema factories.

This module owns the in-memory job/batch state and the background worker
that processes jobs through the warm pipeline. The schema factories
(:func:`build_combined`, :func:`build_summary`) also live here because
they are pure assembly functions tightly coupled to this module's worker.
"""

from __future__ import annotations

import contextlib
import queue
import secrets
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import structlog

from src.models.diarization import DiarizationResult
from src.models.sentiment import SentimentResult
from src.models.transcription import TranscriptionResult
from src.notifier import notify_job_failed
from src.pipeline import PipelineResult, run_pipeline
from src.service.lifecycle import PipelineHandles, ServiceConfig
from src.service.schemas import (
    BatchSummary,
    BatchTotals,
    CombinedTranscript,
    JobFailureStage,
    JobSummaryEntry,
)
from src.service.uri_io import (
    FetchError,
    UnsupportedScheme,
    UploadError,
    fetch_input,
    upload,
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


# ---------------------------------------------------------------------------
# Worker — pulls jobs, runs the pipeline, uploads, and triggers batch summary
# ---------------------------------------------------------------------------


def _content_id_from_job_id(job_id: str) -> str:
    """Strip the ``rfj_`` prefix so the per-job temp file matches the
    ``audio_<content_id>.wav`` naming convention that ``run_pipeline`` expects."""
    return job_id.removeprefix("rfj_")


def _stage_to_failure_attribution(stage_name: str) -> JobFailureStage:
    """Map an internal pipeline stage name to the public summary stage label.

    All upstream stage failures (separation, diarization, transcription,
    sentiment) collapse to ``transcribe`` in the public summary. The
    distinction matters for ops via structured logs; consumers only need to
    know the work didn't produce a transcript."""
    # Every supported value resolves to "transcribe" today. The mapping shape
    # exists so future work (e.g., separating a separation-stage failure
    # category in v0.3.0) is a one-line change.
    _: dict[str, JobFailureStage] = {
        "separation": "transcribe",
        "diarization": "transcribe",
        "transcription": "transcribe",
        "sentiment": "transcribe",
    }
    return _.get(stage_name, "transcribe")


def _find_failed_stage(result: PipelineResult, content_id: str) -> tuple[str, str] | None:
    """Return ``(stage_label, error_message)`` for the first failed stage of
    ``content_id`` in ``result``, or ``None`` if every stage succeeded."""
    stage_attr_names = ("separation", "diarization", "transcription", "sentiment")
    for stage_attr in stage_attr_names:
        stage_result = getattr(result, stage_attr)
        for outcome in stage_result.outcomes:
            if outcome.content_id == content_id and not outcome.success:
                return stage_attr, outcome.error or "unknown error"
    return None


def _persist_intermediates(temp_root: Path, dest_root: Path) -> None:
    """Best-effort copy of per-stage JSONs from the per-job temp dir to a
    persistent location. Failures are caller-logged; this function only does
    the file moves."""
    dest_root.mkdir(parents=True, exist_ok=True)
    for sub in ("diarization", "transcription", "sentiment"):
        src = temp_root / sub
        if not src.is_dir():
            continue
        for f in src.glob("*.json"):
            shutil.copy2(f, dest_root / f.name)


def process_job(
    job: Job,
    handles: PipelineHandles,
    config: ServiceConfig,
    registries: Registries,
) -> None:
    """Process one job end-to-end. Mutates the in-memory Job record via the
    registry; returns nothing. Never writes anything to ``job.output_uri`` on
    failure — the canonical failure surface is the batch summary.

    External dependencies (``run_pipeline``, ``fetch_input``, ``upload``,
    ``notify_job_failed``) are looked up at module scope so tests can patch
    them via the usual ``patch("src.service.jobs.<name>")`` pattern.
    """
    started_at = datetime.now()
    registries.jobs.update(job.job_id, status="processing", started_at=started_at)
    log = structlog.get_logger(__name__).bind(job_id=job.job_id, batch_id=job.batch_id)
    log.info("job.started", input_uri=job.input_uri)

    content_id = _content_id_from_job_id(job.job_id)

    with tempfile.TemporaryDirectory(prefix="rfj_") as tmpdir_name:
        tmp = Path(tmpdir_name)
        source_dir = tmp / "source"
        source_dir.mkdir()
        target_audio = source_dir / f"audio_{content_id}.wav"

        # ── Fetch ─────────────────────────────────────────────────────────
        try:
            fetched = fetch_input(job.input_uri, target_audio)
        except (FetchError, UnsupportedScheme) as exc:
            _record_failure(registries, job, started_at, "download", str(exc), retryable=True, log=log)
            return

        # For file:// fetch_input returns the original path (no copy). Symlink
        # it into source_dir so run_pipeline's discover_files picks it up via
        # the expected naming convention.
        if fetched != target_audio:
            try:
                target_audio.symlink_to(fetched)
            except OSError:
                shutil.copy2(fetched, target_audio)

        # ── Run pipeline ──────────────────────────────────────────────────
        diar_dir = tmp / "diarization"
        tx_dir = tmp / "transcription"
        sent_dir = tmp / "sentiment"
        demucs_dir = tmp / "stems"
        try:
            result = run_pipeline(
                source_dir=source_dir,
                demucs_output_dir=demucs_dir,
                diarization_dir=diar_dir,
                transcription_dir=tx_dir,
                sentiment_dir=sent_dir,
                device=config.device,
                compute_type=config.compute_type,
                batch_size=config.batch_size,
                language=config.language,
                hf_token=config.hf_token,
                enable_sentiment=config.sentiment_enabled,
                manifest=[content_id],
                whisper_model=config.whisper_model,
                model_handles=handles,
                resume=False,
            )
        except Exception as exc:  # noqa: BLE001 — top-level safety net
            _record_failure(registries, job, started_at, "transcribe", repr(exc), retryable=False, log=log)
            return

        failure = _find_failed_stage(result, content_id)
        if failure is not None:
            stage_name, error_msg = failure
            label = _stage_to_failure_attribution(stage_name)
            _record_failure(registries, job, started_at, label, error_msg, retryable=False, log=log)
            return

        # ── Assemble combined transcript ──────────────────────────────────
        try:
            diar_path = diar_dir / f"diarization_{content_id}.json"
            tx_path = tx_dir / f"transcription_{content_id}.json"
            diar = DiarizationResult.model_validate_json(diar_path.read_text())
            tx = TranscriptionResult.model_validate_json(tx_path.read_text())
            sentiment = None
            if config.sentiment_enabled:
                sent_path = sent_dir / f"sentiment_{content_id}.json"
                if sent_path.exists():
                    sentiment = SentimentResult.model_validate_json(sent_path.read_text())
            combined = build_combined(diar, tx, sentiment)
        except Exception as exc:  # noqa: BLE001 — assembly failures are local
            _record_failure(registries, job, started_at, "transcribe", repr(exc), retryable=False, log=log)
            return

        # ── Upload ────────────────────────────────────────────────────────
        try:
            upload(job.output_uri, combined.model_dump(mode="json"))
        except (UploadError, UnsupportedScheme) as exc:
            _record_failure(registries, job, started_at, "upload", str(exc), retryable=True, log=log)
            return

        # ── Optional intermediate persistence ────────────────────────────
        if config.intermediate_dir is not None:
            try:
                _persist_intermediates(tmp, config.intermediate_dir / job.job_id)
            except OSError as exc:
                log.warning("intermediate_persist.failed", error=repr(exc))

        # ── Mark complete ────────────────────────────────────────────────
        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()
        registries.jobs.update(
            job.job_id,
            status="completed",
            completed_at=completed_at,
            duration_seconds=duration,
        )
        log.info("job.completed", duration_seconds=duration)


def _record_failure(
    registries: Registries,
    job: Job,
    started_at: datetime,
    stage: JobFailureStage,
    error: str,
    *,
    retryable: bool,
    log: object | None = None,
) -> None:
    """Mark a job failed in the registry and fire the failure-only Slack hook."""
    failed_at = datetime.now()
    registries.jobs.update(
        job.job_id,
        status="failed",
        started_at=started_at,
        failed_at=failed_at,
        stage=stage,
        error=error,
        retryable=retryable,
    )
    if log is not None:
        log.error("job.failed", stage=stage, error=error, retryable=retryable)  # type: ignore[attr-defined]
    # Failure-only Slack policy: fires per-job on failure; success path is silent.
    # notifier already swallows internally; belt-and-suspenders here.
    with contextlib.suppress(Exception):
        notify_job_failed(job.job_id, stage, job.input_uri, error)


def _job_to_summary_entry(job: Job) -> JobSummaryEntry:
    """Convert a terminal Job record into a JobSummaryEntry for the batch summary."""
    if job.status == "completed":
        return JobSummaryEntry(
            job_id=job.job_id,
            input_uri=job.input_uri,
            output_uri=job.output_uri,
            status="completed",
            started_at=job.started_at or job.created_at,
            completed_at=job.completed_at,
            duration_seconds=job.duration_seconds,
        )
    return JobSummaryEntry(
        job_id=job.job_id,
        input_uri=job.input_uri,
        output_uri=job.output_uri,
        status="failed",
        started_at=job.started_at or job.created_at,
        failed_at=job.failed_at,
        stage=job.stage,  # type: ignore[arg-type]
        error=job.error,
        retryable=job.retryable,
    )


def finalize_batch(
    batch_id: str,
    registries: Registries,
    *,
    completed_at: datetime | None = None,
) -> None:
    """Assemble and upload the batch summary for ``batch_id``.

    Reads every terminal Job in the batch from the registry, builds a
    BatchSummary via the existing factory, and PUTs it to the caller-supplied
    summary_uri. Upload failure logs a structured warning but does not raise
    — the worker thread must never die because the summary upload failed.
    """
    batch = registries.batches.get(batch_id)
    if batch is None:
        return
    finished_at = completed_at if completed_at is not None else datetime.now()
    registries.batches.mark_completed(batch_id, finished_at)

    entries = []
    for job_id in batch.job_ids:
        job = registries.jobs.get(job_id)
        if job is None:
            continue  # already swept by retention or never registered
        entries.append(_job_to_summary_entry(job))

    summary = build_summary(
        batch_id=batch_id,
        submitted_at=batch.submitted_at,
        completed_at=finished_at,
        jobs=entries,
    )

    log = structlog.get_logger(__name__).bind(batch_id=batch_id)
    try:
        upload(batch.summary_uri, summary.model_dump(mode="json"))
        log.info("batch.summary_uploaded", n_jobs=len(entries))
    except (UploadError, UnsupportedScheme) as exc:
        log.warning("batch.summary_upload_failed", summary_uri=batch.summary_uri, error=repr(exc))


class Worker:
    """Background daemon that pulls job_ids off the queue and processes them.

    One worker per container — GPU is the bottleneck. Multiple workers on
    the same GPU would thrash. Scale via container replicas, not via more
    workers per process.
    """

    def __init__(
        self,
        registries: Registries,
        handles: PipelineHandles,
        config: ServiceConfig,
        *,
        get_timeout: float = 1.0,
    ) -> None:
        self._registries = registries
        self._handles = handles
        self._config = config
        self._stop = threading.Event()
        self._get_timeout = get_timeout
        self._thread: threading.Thread | None = None
        self._log = structlog.get_logger(__name__)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="rfj-worker")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            job_id = self._registries.queue.get(timeout=self._get_timeout)
            if job_id is None:
                continue
            self._process_one(job_id)

    def _process_one(self, job_id: str) -> None:
        job = self._registries.jobs.get(job_id)
        if job is None:
            self._log.warning("worker.unknown_job", job_id=job_id)
            return
        try:
            process_job(job, self._handles, self._config, self._registries)
        except Exception as exc:  # noqa: BLE001 — last-resort safety net
            self._log.exception("worker.uncaught_exception", job_id=job_id, error=repr(exc))
            # Best-effort failure record so the batch summary stays accurate.
            with contextlib.suppress(KeyError):
                self._registries.jobs.update(
                    job_id,
                    status="failed",
                    failed_at=datetime.now(),
                    stage="transcribe",
                    error=f"worker uncaught: {exc!r}",
                    retryable=False,
                )

        # Decrement the batch's pending counter; finalize when it hits zero.
        try:
            remaining = self._registries.batches.decrement_pending(job.batch_id)
        except KeyError:
            return
        if remaining == 0:
            finalize_batch(job.batch_id, self._registries)


# ---------------------------------------------------------------------------
# RetentionSweeper — periodically evicts old terminal Job and Batch records
# ---------------------------------------------------------------------------


class RetentionSweeper:
    """Background daemon that periodically evicts terminal records older than
    ``config.job_retention_seconds`` from both registries.

    After eviction ``GET /jobs/{id}`` returns 404 for the old id, matching the
    integration contract ("absence is not a contract violation"). The
    orchestrator already has the per-batch summary at that point, so per-job
    state is no longer needed.

    The sweep is best-effort: an exception during a single tick logs a
    warning and the loop continues. The thread never dies on its own.
    """

    def __init__(
        self,
        registries: Registries,
        config: ServiceConfig,
        *,
        tick_seconds: float = 60.0,
    ) -> None:
        self._registries = registries
        self._config = config
        self._tick = tick_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._log = structlog.get_logger(__name__)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="rfj-retention")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        # Initial wait so a freshly-started service doesn't immediately sweep.
        if self._stop.wait(self._tick):
            return
        while not self._stop.is_set():
            try:
                self.sweep_once()
            except Exception as exc:  # noqa: BLE001 — never die
                self._log.warning("retention.sweep_error", error=repr(exc))
            if self._stop.wait(self._tick):
                return

    def sweep_once(self) -> tuple[int, int]:
        """Evict terminal records older than the retention window.

        Returns ``(evicted_jobs, evicted_batches)``. Exposed for tests and
        operators (e.g., an admin endpoint can call this directly).
        """
        cutoff = datetime.now() - timedelta(seconds=self._config.job_retention_seconds)
        evicted_jobs = 0
        for job in self._registries.jobs.all_jobs():
            terminal_ts = job.completed_at or job.failed_at
            if terminal_ts is not None and terminal_ts < cutoff and self._registries.jobs.delete(job.job_id):
                evicted_jobs += 1
        evicted_batches = 0
        for batch in self._registries.batches.all_batches():
            if (
                batch.completed_at is not None
                and batch.completed_at < cutoff
                and self._registries.batches.delete(batch.batch_id)
            ):
                evicted_batches += 1
        if evicted_jobs or evicted_batches:
            self._log.info("retention.swept", jobs=evicted_jobs, batches=evicted_batches)
        return evicted_jobs, evicted_batches


__all__ = [
    "Batch",
    "BatchRegistry",
    "Job",
    "JobQueue",
    "JobRegistry",
    "Registries",
    "RetentionSweeper",
    "Worker",
    "build_combined",
    "build_summary",
    "finalize_batch",
    "make_batch_id",
    "make_job_id",
    "process_job",
]
