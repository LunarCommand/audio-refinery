"""FastAPI app, endpoints, lifespan wiring, and the `audio-refinery-service` entrypoint.

Wires together everything from Phases 2–6:
- Phase 2 (uri_io): scheme validation for incoming URIs.
- Phase 3 (schemas): per-job transcript + per-batch summary.
- Phase 4 (lifecycle): warm-up, ServiceReadiness, thermal guard.
- Phase 5 (jobs): in-memory registries, queue, Worker, RetentionSweeper.
- Phase 6 (auth): bearer-token dependency.

The integration contract this implements is documented in
``_docs/refinery-integration.md`` and ``_reqs/service-mode.md``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.fs_utils import detect_fstype
from src.service.api_schemas import (
    HealthResponse,
    JobStatusResponse,
    TranscribeRequest,
    TranscribeResponse,
)
from src.service.auth import (
    AllowlistError,
    load_allowlist_from_env,
    make_bearer_dependency,
)
from src.service.config import PipelineHandles, ServiceConfig
from src.service.jobs import (
    Batch,
    Job,
    Registries,
    RetentionSweeper,
    Worker,
    _audio_refinery_version,
    make_batch_id,
    make_job_id,
)
from src.service.lifecycle import (
    ServiceReadiness,
    WarmupError,
    default_thermal_trip,
    start_thermal_guard_from_config,
    warm_up,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_structlog(log_format: str) -> None:
    """Wire structlog with the configured renderer.

    ``json`` (default) emits one JSON object per log record — suitable for
    container log aggregators. ``console`` emits the developer-friendly
    pretty output structlog uses by default.
    """
    if log_format == "console":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


class _StripAuthorizationFilter(logging.Filter):
    """Logging filter that removes ``Authorization`` headers from uvicorn's
    access-log records. We don't rely on uvicorn's access log in production
    (set ``access_log=False`` in :func:`run`), but the filter is defensive —
    any code path that does log a request must not leak bearer tokens."""

    _BEARER_RE = re.compile(r"Bearer\s+\S+")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Bearer " in msg:
            record.msg = self._BEARER_RE.sub("Bearer <redacted>", msg)
            record.args = ()
        return True


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    config: ServiceConfig,
    *,
    registries: Registries | None = None,
    readiness: ServiceReadiness | None = None,
    api_keys: set[str] | None = None,
    handles: PipelineHandles | None = None,
    enable_lifespan_warmup: bool = True,
) -> FastAPI:
    """Build the FastAPI app with all routes wired up.

    Production callers (``run()``) pass only ``config``; the factory reads
    everything else from env or constructs fresh state. Tests can inject
    pre-built ``registries`` / ``readiness`` / ``handles`` / ``api_keys`` and
    set ``enable_lifespan_warmup=False`` so the lifespan handler skips the
    real model load.
    """
    if registries is None:
        # Default registries honor the configured queue size.
        from src.service.jobs import JobQueue

        registries = Registries(queue=JobQueue(maxsize=config.max_queue_size))
    readiness = readiness if readiness is not None else ServiceReadiness()
    api_keys = api_keys if api_keys is not None else load_allowlist_from_env()
    require_bearer = make_bearer_dependency(api_keys)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if enable_lifespan_warmup and handles is None:
            # Spawn warmup in a background thread so the event loop is free to
            # serve /health (returns 503 with "loading" until warmup completes).
            threading.Thread(target=_run_warmup, args=(app, config, registries, readiness), daemon=True).start()
        elif handles is not None:
            # Test path: handles already supplied. Mark readiness ready since
            # warm_up (which normally does this) is being skipped; otherwise
            # the readiness gate on /transcribe would reject every request.
            readiness.mark_ready()
            _start_background_workers(app, config, registries, readiness, handles)

        try:
            yield
        finally:
            if app.state.worker is not None:
                app.state.worker.stop(timeout=5.0)
            if app.state.sweeper is not None:
                app.state.sweeper.stop(timeout=5.0)
            if app.state.thermal_stop is not None:
                app.state.thermal_stop.set()

    app = FastAPI(title="Audio Refinery", version=_audio_refinery_version(), lifespan=lifespan)
    # Initialize state at construction time so endpoints work whether or not
    # the lifespan handler has run (TestClient without `with` doesn't run it).
    app.state.config = config
    app.state.registries = registries
    app.state.readiness = readiness
    app.state.handles = handles
    app.state.worker = None
    app.state.sweeper = None
    app.state.thermal_stop = None
    app.add_middleware(_StructLogContextMiddleware)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> JSONResponse:
        state, stage, detail = request.app.state.readiness.snapshot()
        body = HealthResponse(
            status="ok" if state == "ready" else state,
            stage=stage,
            detail=detail,
        ).model_dump()
        http_status = status.HTTP_200_OK if state == "ready" else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(content=body, status_code=http_status)

    @app.post(
        "/transcribe",
        response_model=TranscribeResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def transcribe(
        body: TranscribeRequest,
        request: Request,
        fp: str = Depends(require_bearer),
    ) -> TranscribeResponse:
        cfg: ServiceConfig = request.app.state.config
        regs: Registries = request.app.state.registries

        # Readiness gate. /transcribe must track /health so the surfaces stay
        # symmetric — accepting work while warming up risks zombie jobs in a
        # container whose warmup ultimately fails, and the caller's SQS-retry
        # path is the right primitive for transient unavailability anyway.
        ready_state, ready_stage, _ = request.app.state.readiness.snapshot()
        if ready_state != "ready":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": "service_not_ready", "state": ready_state, "stage": ready_stage},
                headers={"Retry-After": "5"},
            )

        # Server-side batch cap.
        if len(body.jobs) > cfg.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "batch_too_large",
                    "max": cfg.max_batch_size,
                    "submitted": len(body.jobs),
                },
            )

        # Pre-check queue capacity so we don't half-register a batch.
        if regs.queue.qsize() + len(body.jobs) > regs.queue.maxsize:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error": "queue_full"},
            )

        batch_id = make_batch_id()
        job_ids = [make_job_id() for _ in body.jobs]

        # Register jobs + batch first, then enqueue. Order ensures the worker
        # never pulls a job whose record isn't in the registry yet.
        for job_id, job_body in zip(job_ids, body.jobs, strict=True):
            regs.jobs.add(
                Job(
                    job_id=job_id,
                    batch_id=batch_id,
                    input_uri=job_body.input_uri,
                    output_uri=job_body.output_uri,
                )
            )
        regs.batches.add(
            Batch(
                batch_id=batch_id,
                summary_uri=body.summary_uri,
                job_ids=list(job_ids),
            )
        )
        for job_id in job_ids:
            regs.queue.put_nowait(job_id)

        logger.info(
            "transcribe.accepted",
            batch_id=batch_id,
            n_jobs=len(job_ids),
            caller_fp=fp,
        )
        return TranscribeResponse(batch_id=batch_id, job_ids=job_ids)

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job(
        job_id: str,
        request: Request,
        fp: str = Depends(require_bearer),  # noqa: ARG001 — auth side-effect
    ) -> JobStatusResponse:
        job = request.app.state.registries.jobs.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "job_not_found", "job_id": job_id},
            )
        return JobStatusResponse(
            job_id=job.job_id,
            batch_id=job.batch_id,
            status=job.status,
            input_uri=job.input_uri,
            output_uri=job.output_uri,
            started_at=job.started_at,
            completed_at=job.completed_at,
            failed_at=job.failed_at,
            stage=job.stage,
            error=job.error,
            retryable=job.retryable,
            duration_seconds=job.duration_seconds,
        )

    return app


# ---------------------------------------------------------------------------
# Scratch-location diagnostics
# ---------------------------------------------------------------------------


def _resolve_scratch_location(config: ServiceConfig) -> tuple[Path, str | None]:
    """Return the resolved scratch directory plus its filesystem type (or None).

    Resolution: ``config.scratch_dir`` if set, otherwise the tempfile module's
    default (``TMPDIR`` env or ``/tmp``). Filesystem type is detected by
    :func:`src.fs_utils.detect_fstype`; returns None on non-Linux.
    """
    import tempfile as _tempfile

    path = config.scratch_dir if config.scratch_dir is not None else Path(_tempfile.gettempdir())
    return path, detect_fstype(path)


# ---------------------------------------------------------------------------
# Background-thread orchestration helpers
# ---------------------------------------------------------------------------


def _run_warmup(
    app: FastAPI,
    config: ServiceConfig,
    registries: Registries,
    readiness: ServiceReadiness,
) -> None:
    """Lifespan-thread worker: loads models, then starts the workers."""
    try:
        handles = warm_up(config, readiness)
    except WarmupError as exc:
        # readiness already marked failed by warm_up.
        logger.error("warmup.failed", stage=exc.stage, error=repr(exc.original))
        return
    app.state.handles = handles
    _start_background_workers(app, config, registries, readiness, handles)


def _start_background_workers(
    app: FastAPI,
    config: ServiceConfig,
    registries: Registries,
    readiness: ServiceReadiness,  # noqa: ARG001 — kept for symmetry / future use
    handles: PipelineHandles,
) -> None:
    """Spin up the Worker, RetentionSweeper, and thermal guard."""
    worker = Worker(registries, handles, config)
    worker.start()
    app.state.worker = worker

    sweeper = RetentionSweeper(registries, config)
    sweeper.start()
    app.state.sweeper = sweeper

    app.state.thermal_stop = start_thermal_guard_from_config(config, default_thermal_trip)

    scratch_path, scratch_fstype = _resolve_scratch_location(config)
    logger.info(
        "service.ready",
        device=config.device,
        whisper_model=config.whisper_model,
        scratch_dir=str(scratch_path),
        scratch_fstype=scratch_fstype,
        scratch_is_tmpfs=(scratch_fstype == "tmpfs"),
    )
    if scratch_fstype is not None and scratch_fstype != "tmpfs":
        logger.warning(
            "scratch.not_tmpfs",
            scratch_dir=str(scratch_path),
            scratch_fstype=scratch_fstype,
            hint=(
                "Demucs scratch lives on a disk-backed filesystem. For better batch throughput, mount "
                "tmpfs at the scratch path and set REFINERY_SCRATCH_DIR to point there."
            ),
        )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class _StructLogContextMiddleware(BaseHTTPMiddleware):
    """Binds the request method + path into structlog's context vars so every
    log record emitted while processing a request includes them automatically.

    Deliberately does not log the ``Authorization`` header or any query string
    that might contain a token."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            http_method=request.method,
            http_path=request.url.path,
        )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Production entrypoint — `audio-refinery-service`
# ---------------------------------------------------------------------------


def _config_from_env() -> ServiceConfig:
    """Resolve a ServiceConfig from environment variables.

    Defaults match the ServiceConfig defaults; the service-mode reqs doc lists
    every recognized env var.
    """
    intermediate = os.getenv("REFINERY_INTERMEDIATE_DIR")
    scratch = os.getenv("REFINERY_SCRATCH_DIR")
    return ServiceConfig(
        device=os.getenv("REFINERY_DEVICE", ServiceConfig.__dataclass_fields__["device"].default),  # type: ignore[arg-type]
        whisper_model=os.getenv(
            "REFINERY_WHISPER_MODEL",
            ServiceConfig.__dataclass_fields__["whisper_model"].default,  # type: ignore[arg-type]
        ),
        compute_type=os.getenv(
            "REFINERY_COMPUTE_TYPE",
            ServiceConfig.__dataclass_fields__["compute_type"].default,  # type: ignore[arg-type]
        ),
        language=os.getenv(
            "REFINERY_DEFAULT_LANGUAGE",
            ServiceConfig.__dataclass_fields__["language"].default,  # type: ignore[arg-type]
        ),
        sentiment_enabled=os.getenv("REFINERY_SENTIMENT_ENABLED", "false").lower() == "true",
        hf_token=os.getenv("HF_TOKEN", ""),
        intermediate_dir=Path(intermediate) if intermediate else None,
        scratch_dir=Path(scratch) if scratch else None,
        max_queue_size=int(os.getenv("REFINERY_MAX_QUEUE_SIZE", "100")),
        max_batch_size=int(os.getenv("REFINERY_MAX_BATCH_SIZE", "25")),
        job_retention_seconds=int(os.getenv("REFINERY_JOB_RETENTION_SECONDS", "3600")),
        gpu_temp_limit_celsius=int(os.getenv("REFINERY_GPU_TEMP_LIMIT", "0")),
        gpu_temp_poll_seconds=float(os.getenv("REFINERY_GPU_TEMP_POLL_SECONDS", "5.0")),
    )


def run() -> None:
    """`audio-refinery-service` entry point."""
    log_format = os.getenv("REFINERY_LOG_FORMAT", "json")
    _configure_structlog(log_format)

    # Fail fast if the allowlist is unset — the alternative (a service no
    # caller can reach) is worse than crashlooping.
    try:
        load_allowlist_from_env()
    except AllowlistError as exc:
        logger.error("startup.allowlist_missing", error=str(exc))
        raise SystemExit(1) from exc

    config = _config_from_env()
    app = create_app(config)

    port = int(os.getenv("REFINERY_PORT", "8000"))

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False)  # noqa: S104 — service binds all interfaces by design


__all__ = [
    "create_app",
    "run",
]
