"""Tests for `src.service.lifecycle`.

Covers warmup orchestration, readiness state, and the thermal guard. The
actual model-loading functions (`diarizer.load_pipeline`,
`pipeline._load_whisperx_model`, `sentiment_analyzer.load_sentiment_pipeline`)
are patched so these tests run without a GPU or any heavy weights.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.service.lifecycle import (
    PipelineHandles,
    ServiceConfig,
    ServiceReadiness,
    WarmupError,
    default_thermal_trip,
    start_thermal_guard,
    start_thermal_guard_from_config,
    warm_up,
)

# --------------------------------------------------------------------------
# ServiceReadiness
# --------------------------------------------------------------------------


def test_readiness_defaults_to_loading():
    r = ServiceReadiness()
    state, stage, detail = r.snapshot()
    assert state == "loading"
    assert stage is None
    assert detail is None


def test_readiness_mark_ready_clears_failure_fields():
    r = ServiceReadiness()
    r.mark_failed("transcription", "boom")
    r.mark_ready()
    state, stage, detail = r.snapshot()
    assert state == "ready"
    assert stage is None
    assert detail is None


def test_readiness_mark_failed_records_stage_and_detail():
    r = ServiceReadiness()
    r.mark_failed("diarization", "RuntimeError('CUDA out of memory')")
    state, stage, detail = r.snapshot()
    assert state == "failed"
    assert stage == "diarization"
    assert "CUDA out of memory" in detail  # type: ignore[operator]


def test_readiness_is_thread_safe():
    """Hammer mark_ready / mark_failed from threads and verify snapshot is internally consistent."""
    r = ServiceReadiness()
    stop = threading.Event()

    def flipper():
        while not stop.is_set():
            r.mark_failed("transcription", "x")
            r.mark_ready()

    threads = [threading.Thread(target=flipper, daemon=True) for _ in range(8)]
    for t in threads:
        t.start()
    for _ in range(200):
        state, stage, detail = r.snapshot()
        # state == "ready" must imply both other fields are None
        if state == "ready":
            assert stage is None
            assert detail is None
        elif state == "failed":
            assert stage == "transcription"
            assert detail == "x"
    stop.set()
    for t in threads:
        t.join(timeout=1.0)


# --------------------------------------------------------------------------
# warm_up
# --------------------------------------------------------------------------


@pytest.fixture
def cfg_no_sentiment() -> ServiceConfig:
    return ServiceConfig(
        device="cuda:0",
        whisper_model="large-v3",
        compute_type="float16",
        sentiment_enabled=False,
        hf_token="hf_dummy",
    )


@pytest.fixture
def cfg_with_sentiment(cfg_no_sentiment: ServiceConfig) -> ServiceConfig:
    return ServiceConfig(
        device=cfg_no_sentiment.device,
        whisper_model=cfg_no_sentiment.whisper_model,
        compute_type=cfg_no_sentiment.compute_type,
        sentiment_enabled=True,
        hf_token=cfg_no_sentiment.hf_token,
    )


def test_warm_up_loads_diarizer_and_whisperx_skips_sentiment(cfg_no_sentiment):
    fake_diar = MagicMock(name="pyannote")
    fake_wx = MagicMock(name="whisperx")
    fake_sentiment_loader = MagicMock(name="load_sentiment_pipeline")

    with (
        patch("src.service.lifecycle.load_pipeline", return_value=fake_diar) as diar_load,
        patch("src.service.lifecycle._load_whisperx_model", return_value=fake_wx) as wx_load,
        patch("src.service.lifecycle.load_sentiment_pipeline", new=fake_sentiment_loader),
    ):
        handles = warm_up(cfg_no_sentiment)

    assert isinstance(handles, PipelineHandles)
    assert handles.diarization is fake_diar
    assert handles.whisperx is fake_wx
    assert handles.sentiment is None
    diar_load.assert_called_once()
    wx_load.assert_called_once()
    fake_sentiment_loader.assert_not_called()


def test_warm_up_loads_sentiment_when_enabled(cfg_with_sentiment):
    fake_sentiment = MagicMock(name="sentiment-pipeline")
    with (
        patch("src.service.lifecycle.load_pipeline", return_value=MagicMock()),
        patch("src.service.lifecycle._load_whisperx_model", return_value=MagicMock()),
        patch("src.service.lifecycle.load_sentiment_pipeline", return_value=fake_sentiment) as sent_load,
    ):
        handles = warm_up(cfg_with_sentiment)

    assert handles.sentiment is fake_sentiment
    sent_load.assert_called_once()


def test_warm_up_marks_readiness_ready_on_success(cfg_no_sentiment):
    readiness = ServiceReadiness()
    with (
        patch("src.service.lifecycle.load_pipeline", return_value=MagicMock()),
        patch("src.service.lifecycle._load_whisperx_model", return_value=MagicMock()),
    ):
        warm_up(cfg_no_sentiment, readiness=readiness)

    state, stage, _ = readiness.snapshot()
    assert state == "ready"
    assert stage is None


def test_warm_up_raises_warmup_error_with_stage_on_diarization_failure(cfg_no_sentiment):
    readiness = ServiceReadiness()
    with patch("src.service.lifecycle.load_pipeline", side_effect=RuntimeError("CUDA OOM")):
        with pytest.raises(WarmupError) as exc_info:
            warm_up(cfg_no_sentiment, readiness=readiness)

    assert exc_info.value.stage == "diarization"
    state, stage, detail = readiness.snapshot()
    assert state == "failed"
    assert stage == "diarization"
    assert "CUDA OOM" in detail  # type: ignore[operator]


def test_warm_up_raises_warmup_error_on_whisperx_failure(cfg_no_sentiment):
    readiness = ServiceReadiness()
    with (
        patch("src.service.lifecycle.load_pipeline", return_value=MagicMock()),
        patch("src.service.lifecycle._load_whisperx_model", side_effect=RuntimeError("WX OOM")),
    ):
        with pytest.raises(WarmupError) as exc_info:
            warm_up(cfg_no_sentiment, readiness=readiness)

    assert exc_info.value.stage == "transcription"
    state, stage, _ = readiness.snapshot()
    assert state == "failed"
    assert stage == "transcription"


def test_warm_up_raises_warmup_error_on_sentiment_failure(cfg_with_sentiment):
    readiness = ServiceReadiness()
    with (
        patch("src.service.lifecycle.load_pipeline", return_value=MagicMock()),
        patch("src.service.lifecycle._load_whisperx_model", return_value=MagicMock()),
        patch("src.service.lifecycle.load_sentiment_pipeline", side_effect=RuntimeError("sentiment fail")),
    ):
        with pytest.raises(WarmupError) as exc_info:
            warm_up(cfg_with_sentiment, readiness=readiness)

    assert exc_info.value.stage == "sentiment"
    state, stage, _ = readiness.snapshot()
    assert state == "failed"
    assert stage == "sentiment"


def test_warm_up_without_readiness_works(cfg_no_sentiment):
    """Tests that don't care about /health state can skip the readiness arg."""
    with (
        patch("src.service.lifecycle.load_pipeline", return_value=MagicMock()),
        patch("src.service.lifecycle._load_whisperx_model", return_value=MagicMock()),
    ):
        handles = warm_up(cfg_no_sentiment)  # no readiness arg
    assert handles.diarization is not None


# --------------------------------------------------------------------------
# start_thermal_guard
# --------------------------------------------------------------------------


def test_thermal_guard_returns_none_for_cpu_device():
    callback = MagicMock()
    result = start_thermal_guard(device="cpu", limit_celsius=85, on_trip=callback)
    assert result is None
    callback.assert_not_called()


def test_thermal_guard_returns_none_when_limit_disabled():
    callback = MagicMock()
    result = start_thermal_guard(device="cuda:0", limit_celsius=0, on_trip=callback)
    assert result is None
    callback.assert_not_called()


def test_thermal_guard_invokes_callback_when_temperature_exceeds_limit():
    """Use a fast poll interval and a mocked query that returns over-limit on the first call."""
    callback = MagicMock()
    with patch("src.service.lifecycle.query_gpu_temperature", return_value=95):
        stop = start_thermal_guard(
            device="cuda:0",
            limit_celsius=85,
            on_trip=callback,
            poll_interval_seconds=0.01,
        )
        assert stop is not None
        # Give the daemon a moment to tick
        time.sleep(0.1)
        stop.set()

    callback.assert_called()
    args = callback.call_args.args
    assert args[0] == "cuda:0"
    assert args[1] == 95
    assert args[2] == 85


def test_thermal_guard_does_not_trip_when_temperature_is_safe():
    callback = MagicMock()
    with patch("src.service.lifecycle.query_gpu_temperature", return_value=70):
        stop = start_thermal_guard(
            device="cuda:0",
            limit_celsius=85,
            on_trip=callback,
            poll_interval_seconds=0.01,
        )
        assert stop is not None
        time.sleep(0.1)
        stop.set()

    callback.assert_not_called()


def test_thermal_guard_skips_silently_when_temp_query_returns_none():
    """nvidia-smi unavailable / parse failure → query returns None; guard must not trip."""
    callback = MagicMock()
    with patch("src.service.lifecycle.query_gpu_temperature", return_value=None):
        stop = start_thermal_guard(
            device="cuda:0",
            limit_celsius=85,
            on_trip=callback,
            poll_interval_seconds=0.01,
        )
        assert stop is not None
        time.sleep(0.1)
        stop.set()

    callback.assert_not_called()


def test_thermal_guard_from_config_threads_fields_through(monkeypatch: pytest.MonkeyPatch):
    """The config-wrapper must pass `device`, `gpu_temp_limit_celsius`, and
    `gpu_temp_poll_seconds` straight to the underlying start_thermal_guard call."""
    captured = {}

    def fake_start(*, device, limit_celsius, on_trip, poll_interval_seconds):
        captured["device"] = device
        captured["limit_celsius"] = limit_celsius
        captured["poll_interval_seconds"] = poll_interval_seconds
        captured["on_trip"] = on_trip
        return threading.Event()

    monkeypatch.setattr("src.service.lifecycle.start_thermal_guard", fake_start)

    cb = MagicMock()
    cfg = ServiceConfig(
        device="cuda:1",
        gpu_temp_limit_celsius=82,
        gpu_temp_poll_seconds=2.5,
    )
    start_thermal_guard_from_config(cfg, cb)

    assert captured["device"] == "cuda:1"
    assert captured["limit_celsius"] == 82
    assert captured["poll_interval_seconds"] == 2.5
    assert captured["on_trip"] is cb


def test_thermal_guard_from_config_returns_none_when_disabled():
    """A config with default `gpu_temp_limit_celsius=0` skips the guard via the
    underlying start_thermal_guard behavior — verified end-to-end (no patch)."""
    cb = MagicMock()
    cfg = ServiceConfig()  # all defaults; limit is 0
    result = start_thermal_guard_from_config(cfg, cb)
    assert result is None
    cb.assert_not_called()


def test_thermal_guard_from_config_uses_configured_poll_interval():
    """Verify the configured poll interval reaches the daemon thread by mocking
    the temperature query and observing how quickly the trip callback fires."""
    cb = MagicMock()
    cfg = ServiceConfig(
        device="cuda:0",
        gpu_temp_limit_celsius=85,
        gpu_temp_poll_seconds=0.02,
    )
    with patch("src.service.lifecycle.query_gpu_temperature", return_value=95):
        stop = start_thermal_guard_from_config(cfg, cb)
        assert stop is not None
        time.sleep(0.1)  # plenty of ticks at 20ms cadence
        stop.set()
    cb.assert_called()


def test_default_thermal_trip_calls_notifier_and_exits(monkeypatch: pytest.MonkeyPatch):
    exit_calls = []

    def fake_exit(code: int) -> None:
        exit_calls.append(code)

    monkeypatch.setattr("src.service.lifecycle.os._exit", fake_exit)
    notify = MagicMock()
    monkeypatch.setattr("src.service.lifecycle.notify_thermal_shutdown", notify)

    default_thermal_trip("cuda:0", 95, 85)

    notify.assert_called_once_with("cuda:0", 95, 85)
    assert exit_calls == [1]
