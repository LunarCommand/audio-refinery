"""Service-mode runtime configuration and pre-loaded model handles.

Both classes are pure-data containers — frozen / mutable dataclasses with no
behavior. They live here so the modules that act on them (``lifecycle.py``,
``jobs.py``, ``app.py``) stay focused on behavior, matching the project's
convention of separating data shapes from the code that uses them
(``src/models/*.py`` vs the stage modules, ``schemas.py`` vs ``jobs.py``).

``ServiceConfig`` is resolved from environment variables in
``src.service.app:_config_from_env``. ``PipelineHandles`` is produced by
``src.service.lifecycle.warm_up`` at container startup and consumed by the
worker on every job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.pipeline import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_SENTIMENT_MODEL,
    DIARIZER_DEFAULT_MODEL,
    TRANSCRIBER_DEFAULT_MODEL,
)


@dataclass(frozen=True)
class ServiceConfig:
    """Container-startup configuration. Resolved from env vars in app.py.

    Worker / queue / retention fields:
        ``intermediate_dir`` — when set (from ``REFINERY_INTERMEDIATE_DIR``),
            the worker copies per-stage JSONs to ``<dir>/<job_id>/`` after each
            successful job. Debugging-only; off by default.
        ``scratch_dir`` — base directory for per-job temp dirs that hold the
            input download + Demucs stems + per-stage JSONs. Set from
            ``REFINERY_SCRATCH_DIR``. When unset, the worker uses Python's
            tempfile default (TMPDIR env or /tmp). Recommended: mount a tmpfs
            volume here for the RAM-disk benefit Demucs benefits from. Each
            job uses its own subdirectory, cleaned up automatically when the
            job settles; peak transient space is one job's worth (~3–5 GB).
        ``max_queue_size`` — JobQueue cap from ``REFINERY_MAX_QUEUE_SIZE``.
            Default 100. Over-limit POSTs return 429.
        ``max_batch_size`` — server-side cap on jobs per POST /transcribe call
            from ``REFINERY_MAX_BATCH_SIZE``. Default 25.
        ``job_retention_seconds`` — retention window for terminal jobs/batches
            from ``REFINERY_JOB_RETENTION_SECONDS``. Default 3600 (1 hour after
            ``completed_at`` / ``failed_at``).

    Thermal-guard fields:
        ``gpu_temp_limit_celsius`` — °C threshold from ``REFINERY_GPU_TEMP_LIMIT``.
            ``0`` (default) disables the guard entirely. Any positive value spawns
            the daemon thread at lifespan startup.
        ``gpu_temp_poll_seconds`` — how often the guard polls nvidia-smi, from
            ``REFINERY_GPU_TEMP_POLL_SECONDS``. Default ``5.0`` matches the CLI's
            ``_run_temp_guard`` cadence. Tune higher on shared hosts where
            nvidia-smi polling is expensive.
    """

    device: str = DEFAULT_DEVICE
    whisper_model: str = TRANSCRIBER_DEFAULT_MODEL
    compute_type: str = DEFAULT_COMPUTE_TYPE
    batch_size: int = DEFAULT_BATCH_SIZE
    language: str = DEFAULT_LANGUAGE
    diarizer_model: str = DIARIZER_DEFAULT_MODEL
    sentiment_model: str = DEFAULT_SENTIMENT_MODEL
    sentiment_enabled: bool = False
    hf_token: str = ""
    intermediate_dir: Path | None = None
    scratch_dir: Path | None = None
    max_queue_size: int = 100
    max_batch_size: int = 25
    job_retention_seconds: int = 3600
    gpu_temp_limit_celsius: int = 0
    gpu_temp_poll_seconds: float = 5.0


@dataclass
class PipelineHandles:
    """Pre-loaded model handles passed into ``run_pipeline()`` by the service.

    The pipeline reads attributes on this bundle when a non-None ``model_handles``
    parameter is supplied. CLI mode never constructs one — it sticks with the
    existing per-call load path.
    """

    diarization: Any
    whisperx: Any
    sentiment: Any | None = None


__all__ = [
    "PipelineHandles",
    "ServiceConfig",
]
