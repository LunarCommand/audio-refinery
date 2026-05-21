"""Service-mode model lifecycle: warmup, readiness, thermal guard.

In CLI mode the pipeline loads pyannote, WhisperX, and (optionally) sentiment
once per ``run_pipeline()`` invocation. The service runs as a long-lived
container with serial job processing, so paying that load cost on every job
would push first-byte latency to ~10 seconds and defeat the ``202 Accepted``
contract.

This module owns the alternative: ``warm_up(config)`` loads every in-process
GPU model once at container startup and returns a ``PipelineHandles`` bundle
(defined in :mod:`src.service.config`) that gets passed into ``run_pipeline()``
for the duration of the container's lifetime. The existing per-call load
functions (`diarizer.load_pipeline`, ``transcriber._load_whisperx_model``,
``sentiment_analyzer.load_sentiment_pipeline``) are reused as-is — service
mode is additive, the CLI's lifecycle is unchanged.

Pure data lives in :mod:`src.service.config` (``ServiceConfig``,
``PipelineHandles``); this module holds only the behavior that consumes it.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from src.diarizer import DiarizationError, load_pipeline
from src.gpu_utils import query_gpu_temperature
from src.notifier import notify_thermal_shutdown
from src.pipeline import _load_whisperx_model, _parse_whisperx_device
from src.sentiment_analyzer import load_sentiment_pipeline
from src.service.config import PipelineHandles, ServiceConfig

ReadyState = Literal["loading", "ready", "failed"]


# ---------------------------------------------------------------------------
# Lifecycle behavior (data lives in src.service.config)
# ---------------------------------------------------------------------------


class WarmupError(RuntimeError):
    """Raised by ``warm_up`` when a model fails to load.

    Carries the stage name so the caller (and ``ServiceReadiness``) can attribute
    the failure cleanly.
    """

    def __init__(self, stage: str, original: BaseException) -> None:
        super().__init__(f"warmup failed at stage {stage!r}: {original!r}")
        self.stage = stage
        self.original = original


# ---------------------------------------------------------------------------
# Readiness state — read by the /health endpoint
# ---------------------------------------------------------------------------


@dataclass
class ServiceReadiness:
    """Thread-safe state object the `/health` endpoint reads.

    Three states:
      - ``"loading"``: warmup in progress, /health returns 503
      - ``"ready"``: warmup succeeded, /health returns 200
      - ``"failed"``: warmup raised, /health returns 503 with the failing stage
    """

    _state: ReadyState = "loading"
    _failed_stage: str | None = None
    _failed_detail: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> tuple[ReadyState, str | None, str | None]:
        """Return ``(state, failed_stage, failed_detail)`` atomically."""
        with self._lock:
            return self._state, self._failed_stage, self._failed_detail

    def mark_ready(self) -> None:
        with self._lock:
            self._state = "ready"
            self._failed_stage = None
            self._failed_detail = None

    def mark_failed(self, stage: str, detail: str) -> None:
        with self._lock:
            self._state = "failed"
            self._failed_stage = stage
            self._failed_detail = detail


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def warm_up(config: ServiceConfig, readiness: ServiceReadiness | None = None) -> PipelineHandles:
    """Load every in-process GPU model the service needs.

    Order is deterministic so failures pinpoint which stage broke:
      1. pyannote diarization
      2. WhisperX transcription
      3. (optional) sentiment

    On success, marks ``readiness`` as ``"ready"`` and returns a populated
    ``PipelineHandles``. On any failure, marks ``readiness`` as ``"failed"`` with
    the offending stage and re-raises as ``WarmupError`` so the caller (typically
    the FastAPI lifespan handler) can decide whether to exit the process.

    Args:
        config: Resolved container config.
        readiness: Optional readiness object to update. Pass ``None`` in tests
            that don't care about /health state.

    Returns:
        PipelineHandles with every configured model loaded.

    Raises:
        WarmupError: A stage failed to load. ``.stage`` identifies which.
    """
    try:
        diar = load_pipeline(config.diarizer_model, config.device, config.hf_token)
    except (DiarizationError, Exception) as exc:
        _record_failure(readiness, "diarization", exc)
        raise WarmupError("diarization", exc) from exc

    try:
        ct2_device, ct2_device_index = _parse_whisperx_device(config.device)
        wx_language = None if config.language == "auto" else config.language
        whisperx_model = _load_whisperx_model(
            config.whisper_model,
            ct2_device,
            ct2_device_index,
            config.compute_type,
            wx_language,
        )
    except Exception as exc:
        _record_failure(readiness, "transcription", exc)
        raise WarmupError("transcription", exc) from exc

    sentiment = None
    if config.sentiment_enabled:
        try:
            sentiment = load_sentiment_pipeline(config.sentiment_model, "cpu")
        except Exception as exc:
            _record_failure(readiness, "sentiment", exc)
            raise WarmupError("sentiment", exc) from exc

    if readiness is not None:
        readiness.mark_ready()
    return PipelineHandles(diarization=diar, whisperx=whisperx_model, sentiment=sentiment)


def _record_failure(readiness: ServiceReadiness | None, stage: str, exc: BaseException) -> None:
    if readiness is not None:
        readiness.mark_failed(stage, repr(exc))


# ---------------------------------------------------------------------------
# Thermal guard — ports the CLI pattern at src/cli.py:126 into the service
# ---------------------------------------------------------------------------


def start_thermal_guard(
    device: str,
    limit_celsius: int,
    on_trip: Callable[[str, int, int], None],
    *,
    poll_interval_seconds: float = 5.0,
) -> threading.Event | None:
    """Spawn a daemon thread that polls GPU temperature and trips the callback.

    Mirrors ``src/cli.py::_run_temp_guard``: 5-second polls of
    ``query_gpu_temperature``, callback when the limit is reached.

    Skipped (returns ``None``) when ``device == "cpu"`` or ``limit_celsius <= 0``.
    Callers store the returned ``threading.Event`` and ``.set()`` it to terminate
    the guard cleanly during shutdown.

    Args:
        device: PyTorch-style device string, e.g. ``"cuda:0"``.
        limit_celsius: Threshold; trip occurs at ``temp >= limit_celsius``.
        on_trip: Callback invoked with ``(device, temp, limit)``. Expected to
            mark the in-flight job failed, write the partial batch summary,
            fire ``notify_thermal_shutdown``, and exit the process. Provided
            by the worker in Phase 5; tests pass a recording callable.
        poll_interval_seconds: How often to query the GPU. Default 5s.

    Returns:
        ``threading.Event`` whose ``.set()`` cleanly terminates the guard,
        or ``None`` when the guard is skipped.
    """
    if device == "cpu" or limit_celsius <= 0:
        return None

    stop = threading.Event()
    device_index = int(device.split(":")[1]) if ":" in device else 0

    def _run() -> None:
        while not stop.wait(poll_interval_seconds):
            temp = query_gpu_temperature(device_index)
            if temp is not None and temp >= limit_celsius:
                on_trip(device, temp, limit_celsius)
                return

    threading.Thread(target=_run, daemon=True, name="thermal-guard").start()
    return stop


def start_thermal_guard_from_config(
    config: ServiceConfig,
    on_trip: Callable[[str, int, int], None],
) -> threading.Event | None:
    """Convenience wrapper that threads ``ServiceConfig`` fields into ``start_thermal_guard``.

    The lifespan handler in ``app.py`` calls this once at startup. Direct callers
    (tests, custom integrations) can still use ``start_thermal_guard`` with
    explicit args.

    Returns ``None`` when the guard is disabled (device is ``cpu`` or
    ``gpu_temp_limit_celsius <= 0``).
    """
    return start_thermal_guard(
        device=config.device,
        limit_celsius=config.gpu_temp_limit_celsius,
        on_trip=on_trip,
        poll_interval_seconds=config.gpu_temp_poll_seconds,
    )


def default_thermal_trip(device: str, temp: int, limit: int) -> None:
    """Default ``on_trip`` callback for the thermal guard.

    Fires the existing thermal-shutdown Slack notification and exits the
    process non-zero so the orchestrator restarts the container after
    cool-down. Phase 5's worker will replace this with a richer callback
    that first marks the in-flight job failed and writes the partial batch
    summary; this default is the fallback for setups (or tests) that don't
    inject a worker-aware callback.
    """
    notify_thermal_shutdown(device, temp, limit)
    os._exit(1)


__all__ = [
    "PipelineHandles",
    "ServiceConfig",
    "ServiceReadiness",
    "WarmupError",
    "default_thermal_trip",
    "start_thermal_guard",
    "start_thermal_guard_from_config",
    "warm_up",
]
