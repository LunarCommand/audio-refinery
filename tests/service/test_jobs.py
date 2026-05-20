"""Tests for the in-memory data primitives in `src.service.jobs`.

Worker behavior tests land in a subsequent commit. This file covers the
dataclasses, registries, queue, and ID-generation helpers in isolation.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline import FileOutcome, PipelineResult, StageResult
from src.service.config import PipelineHandles, ServiceConfig
from src.service.jobs import (
    Batch,
    BatchRegistry,
    Job,
    JobQueue,
    JobRegistry,
    Registries,
    RetentionSweeper,
    Worker,
    finalize_batch,
    make_batch_id,
    make_job_id,
    process_job,
)

# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


def test_make_job_id_format():
    jid = make_job_id()
    assert re.fullmatch(r"rfj_[0-9a-f]{16}", jid) is not None


def test_make_batch_id_format():
    bid = make_batch_id()
    assert re.fullmatch(r"btc_[0-9a-f]{16}", bid) is not None


def test_make_job_id_collision_resistance():
    """secrets.token_hex(8) is 64 bits — collisions in a small loop are
    statistically impossible. This is a sanity-check, not a probability test."""
    ids = {make_job_id() for _ in range(1000)}
    assert len(ids) == 1000


# ---------------------------------------------------------------------------
# JobRegistry
# ---------------------------------------------------------------------------


def _make_job(job_id: str = "rfj_1", batch_id: str = "btc_1") -> Job:
    return Job(
        job_id=job_id,
        batch_id=batch_id,
        input_uri="https://x/in.wav",
        output_uri="https://x/out.json",
    )


def test_job_registry_add_get_roundtrip():
    reg = JobRegistry()
    job = _make_job()
    reg.add(job)
    fetched = reg.get(job.job_id)
    assert fetched is job
    assert job.job_id in reg


def test_job_registry_get_missing_returns_none():
    reg = JobRegistry()
    assert reg.get("rfj_nonexistent") is None


def test_job_registry_duplicate_add_raises():
    reg = JobRegistry()
    reg.add(_make_job())
    with pytest.raises(ValueError, match="already registered"):
        reg.add(_make_job())


def test_job_registry_update_mutates_fields():
    reg = JobRegistry()
    reg.add(_make_job())
    now = datetime(2026, 5, 19, 22, 0, 0)
    reg.update("rfj_1", status="processing", started_at=now)
    fetched = reg.get("rfj_1")
    assert fetched is not None
    assert fetched.status == "processing"
    assert fetched.started_at == now


def test_job_registry_update_missing_raises():
    reg = JobRegistry()
    with pytest.raises(KeyError):
        reg.update("rfj_nope", status="completed")


def test_job_registry_delete():
    reg = JobRegistry()
    reg.add(_make_job())
    assert reg.delete("rfj_1") is True
    assert reg.get("rfj_1") is None
    assert reg.delete("rfj_1") is False  # second delete is a no-op


def test_job_registry_all_jobs_returns_snapshot():
    reg = JobRegistry()
    reg.add(_make_job("rfj_1"))
    reg.add(_make_job("rfj_2"))
    snap = reg.all_jobs()
    assert {j.job_id for j in snap} == {"rfj_1", "rfj_2"}


def test_job_registry_thread_safety_under_contention():
    reg = JobRegistry()
    # Pre-populate.
    for i in range(50):
        reg.add(_make_job(job_id=f"rfj_{i:04x}"))

    stop = threading.Event()

    def update_loop():
        i = 0
        while not stop.is_set():
            reg.update(f"rfj_{i % 50:04x}", status="processing")
            i += 1

    def get_loop():
        while not stop.is_set():
            for j in reg.all_jobs():
                # snapshot consistency: status is always one of the legal values
                assert j.status in {"queued", "processing", "completed", "failed"}

    threads = [threading.Thread(target=update_loop, daemon=True) for _ in range(4)]
    threads += [threading.Thread(target=get_loop, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    threading.Event().wait(0.1)
    stop.set()
    for t in threads:
        t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# BatchRegistry
# ---------------------------------------------------------------------------


def _make_batch(batch_id: str = "btc_1", job_ids: list[str] | None = None) -> Batch:
    return Batch(
        batch_id=batch_id,
        summary_uri="https://x/summary.json",
        job_ids=job_ids if job_ids is not None else ["rfj_1", "rfj_2"],
    )


def test_batch_registry_add_initializes_pending_count():
    reg = BatchRegistry()
    reg.add(_make_batch(job_ids=["a", "b", "c"]))
    batch = reg.get("btc_1")
    assert batch is not None
    assert batch.pending_count == 3


def test_batch_registry_decrement_pending_returns_new_value():
    reg = BatchRegistry()
    reg.add(_make_batch(job_ids=["a", "b"]))
    assert reg.decrement_pending("btc_1") == 1
    assert reg.decrement_pending("btc_1") == 0


def test_batch_registry_decrement_pending_never_goes_negative():
    reg = BatchRegistry()
    reg.add(_make_batch(job_ids=["a"]))
    assert reg.decrement_pending("btc_1") == 0
    assert reg.decrement_pending("btc_1") == 0  # second decrement is a no-op


def test_batch_registry_decrement_pending_unknown_raises():
    reg = BatchRegistry()
    with pytest.raises(KeyError):
        reg.decrement_pending("btc_nope")


def test_batch_registry_mark_completed():
    reg = BatchRegistry()
    reg.add(_make_batch())
    ts = datetime(2026, 5, 19, 22, 5, 0)
    reg.mark_completed("btc_1", ts)
    batch = reg.get("btc_1")
    assert batch is not None and batch.completed_at == ts


def test_batch_registry_decrement_atomic_across_threads():
    """Hammer decrement_pending from many threads and confirm the final count
    matches the expected number of decrements (no lost updates)."""
    reg = BatchRegistry()
    n_jobs = 200
    reg.add(_make_batch(job_ids=[f"rfj_{i}" for i in range(n_jobs)]))

    barrier = threading.Barrier(8)
    results: list[int] = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        for _ in range(25):  # 8 * 25 = 200 decrements total
            r = reg.decrement_pending("btc_1")
            with lock:
                results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    batch = reg.get("btc_1")
    assert batch is not None
    assert batch.pending_count == 0
    assert 0 in results  # we hit zero exactly once when the last decrement landed


# ---------------------------------------------------------------------------
# JobQueue
# ---------------------------------------------------------------------------


def test_job_queue_fifo_order():
    q = JobQueue(maxsize=10)
    q.put_nowait("rfj_a")
    q.put_nowait("rfj_b")
    q.put_nowait("rfj_c")
    assert q.get(timeout=0.1) == "rfj_a"
    assert q.get(timeout=0.1) == "rfj_b"
    assert q.get(timeout=0.1) == "rfj_c"


def test_job_queue_get_timeout_returns_none():
    q = JobQueue(maxsize=10)
    assert q.get(timeout=0.05) is None


def test_job_queue_raises_full_when_over_capacity():
    q = JobQueue(maxsize=2)
    q.put_nowait("a")
    q.put_nowait("b")
    with pytest.raises(queue.Full):
        q.put_nowait("c")


def test_job_queue_default_maxsize_is_100():
    q = JobQueue()
    assert q.maxsize == 100


# ---------------------------------------------------------------------------
# Registries bundle
# ---------------------------------------------------------------------------


def test_registries_bundle_provides_fresh_instances():
    r = Registries()
    assert isinstance(r.jobs, JobRegistry)
    assert isinstance(r.batches, BatchRegistry)
    assert isinstance(r.queue, JobQueue)


# ---------------------------------------------------------------------------
# Worker behavior (process_job, finalize_batch, Worker class)
# ---------------------------------------------------------------------------


def _stage_result(content_id: str, success: bool = True, error: str | None = None) -> StageResult:
    return StageResult(outcomes=[FileOutcome(content_id=content_id, stage="x", success=success, error=error)])


def _build_pipeline_side_effect(
    content_id: str,
    *,
    sentiment_enabled: bool = False,
    fail_stage: str | None = None,
):
    """Return a callable suitable for `_run_pipeline=...` that simulates the
    real pipeline's side effects: writes per-stage JSONs into the configured
    directories and returns a PipelineResult reflecting success/failure."""

    def fake_run_pipeline(
        *, source_dir, demucs_output_dir, diarization_dir, transcription_dir, sentiment_dir, **kwargs
    ):
        # Write diarization JSON unless told to fail diarization.
        diarization_dir.mkdir(parents=True, exist_ok=True)
        transcription_dir.mkdir(parents=True, exist_ok=True)
        if sentiment_enabled:
            sentiment_dir.mkdir(parents=True, exist_ok=True)

        # Minimal valid JSON for each stage's Pydantic model.
        audio_blob = {
            "path": str(source_dir / f"audio_{content_id}.wav"),
            "sample_rate": 16000,
            "channels": 1,
            "duration_seconds": 1.0,
            "frames": 16000,
            "format_str": "WAV",
            "subtype": "PCM_16",
        }
        ts = "2026-05-19T22:00:00"
        diar_doc = {
            "input_file": audio_blob["path"],
            "input_info": audio_blob,
            "segments": [],
            "num_speakers": 0,
            "device": "cpu",
            "processing_time_seconds": 0.1,
            "started_at": ts,
            "completed_at": ts,
        }
        tx_doc = {
            "input_file": audio_blob["path"],
            "input_info": audio_blob,
            "language": "en",
            "segments": [],
            "device": "cpu",
            "compute_type": "float16",
            "batch_size": 16,
            "processing_time_seconds": 0.1,
            "started_at": ts,
            "completed_at": ts,
        }
        sent_doc = {
            "transcription_file": str(transcription_dir / f"transcription_{content_id}.json"),
            "segments": [],
            "device": "cpu",
            "processing_time_seconds": 0.05,
            "started_at": ts,
            "completed_at": ts,
        }

        import json

        if fail_stage != "diarization":
            (diarization_dir / f"diarization_{content_id}.json").write_text(json.dumps(diar_doc))
        if fail_stage != "transcription":
            (transcription_dir / f"transcription_{content_id}.json").write_text(json.dumps(tx_doc))
        if sentiment_enabled and fail_stage != "sentiment":
            (sentiment_dir / f"sentiment_{content_id}.json").write_text(json.dumps(sent_doc))

        sep_outcomes = [FileOutcome(content_id=content_id, stage="separate", success=True)]
        diar_outcomes = [
            FileOutcome(
                content_id=content_id,
                stage="diarize",
                success=fail_stage != "diarization",
                error="boom" if fail_stage == "diarization" else None,
            )
        ]
        tx_outcomes = [
            FileOutcome(
                content_id=content_id,
                stage="transcribe",
                success=fail_stage not in ("diarization", "transcription"),
                error="boom" if fail_stage == "transcription" else None,
            )
        ]
        sent_outcomes = []
        if sentiment_enabled:
            sent_outcomes = [
                FileOutcome(
                    content_id=content_id,
                    stage="sentiment",
                    success=fail_stage != "sentiment",
                    error="boom" if fail_stage == "sentiment" else None,
                )
            ]

        return PipelineResult(
            total_discovered=1,
            separation=StageResult(outcomes=sep_outcomes),
            diarization=StageResult(outcomes=diar_outcomes),
            transcription=StageResult(outcomes=tx_outcomes),
            sentiment=StageResult(outcomes=sent_outcomes),
        )

    return fake_run_pipeline


def _register_job_and_batch(
    registries: Registries,
    *,
    job_id: str = "rfj_abc123def4567890",
    batch_id: str = "btc_xyz",
    summary_uri: str = "file:///tmp/summary.json",
    input_uri: str = "file:///inbox/x.wav",
    output_uri: str = "file:///outbox/x.json",
) -> tuple[Job, Batch]:
    job = Job(
        job_id=job_id,
        batch_id=batch_id,
        input_uri=input_uri,
        output_uri=output_uri,
    )
    registries.jobs.add(job)
    batch = Batch(
        batch_id=batch_id,
        summary_uri=summary_uri,
        job_ids=[job_id],
    )
    registries.batches.add(batch)
    return job, batch


@pytest.fixture
def handles() -> PipelineHandles:
    return PipelineHandles(diarization=MagicMock(), whisperx=MagicMock(), sentiment=None)


@pytest.fixture
def config() -> ServiceConfig:
    return ServiceConfig(device="cpu", sentiment_enabled=False)


def test_process_job_happy_path_writes_transcript_and_marks_completed(
    handles: PipelineHandles, config: ServiceConfig, tmp_path: Path
) -> None:
    registries = Registries()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    job, _ = _register_job_and_batch(
        registries,
        input_uri=f"file://{audio}",
        output_uri=f"file://{tmp_path / 'out.json'}",
    )
    content_id = job.job_id.removeprefix("rfj_")

    with (
        patch("src.service.jobs.run_pipeline", new=_build_pipeline_side_effect(content_id)),
        patch("src.service.jobs.upload") as upload_mock,
        patch("src.service.jobs.notify_job_failed") as notify,
    ):
        process_job(job, handles, config, registries)

    fetched = registries.jobs.get(job.job_id)
    assert fetched is not None
    assert fetched.status == "completed"
    assert fetched.duration_seconds is not None
    assert fetched.completed_at is not None
    assert fetched.stage is None
    assert fetched.error is None

    upload_mock.assert_called_once()
    notify.assert_not_called()


def test_process_job_download_failure_marks_failed_and_notifies(
    handles: PipelineHandles, config: ServiceConfig, tmp_path: Path
) -> None:
    from src.service.uri_io import FetchError

    registries = Registries()
    job, _ = _register_job_and_batch(registries, input_uri="file:///nonexistent/x.wav")

    with (
        patch("src.service.jobs.fetch_input", side_effect=FetchError("not found")),
        patch("src.service.jobs.run_pipeline") as run_mock,
        patch("src.service.jobs.upload") as upload_mock,
        patch("src.service.jobs.notify_job_failed") as notify,
    ):
        process_job(job, handles, config, registries)

    fetched = registries.jobs.get(job.job_id)
    assert fetched is not None
    assert fetched.status == "failed"
    assert fetched.stage == "download"
    assert fetched.retryable is True
    run_mock.assert_not_called()
    upload_mock.assert_not_called()
    notify.assert_called_once()
    assert notify.call_args.args[1] == "download"


def test_process_job_transcribe_failure_marks_failed_no_upload(
    handles: PipelineHandles, config: ServiceConfig, tmp_path: Path
) -> None:
    registries = Registries()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    job, _ = _register_job_and_batch(registries, input_uri=f"file://{audio}")
    content_id = job.job_id.removeprefix("rfj_")

    with (
        patch(
            "src.service.jobs.run_pipeline",
            new=_build_pipeline_side_effect(content_id, fail_stage="transcription"),
        ),
        patch("src.service.jobs.upload") as upload_mock,
        patch("src.service.jobs.notify_job_failed") as notify,
    ):
        process_job(job, handles, config, registries)

    fetched = registries.jobs.get(job.job_id)
    assert fetched is not None
    assert fetched.status == "failed"
    assert fetched.stage == "transcribe"
    upload_mock.assert_not_called()
    notify.assert_called_once()


def test_process_job_upload_failure_marks_failed(
    handles: PipelineHandles, config: ServiceConfig, tmp_path: Path
) -> None:
    from src.service.uri_io import UploadError

    registries = Registries()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    job, _ = _register_job_and_batch(registries, input_uri=f"file://{audio}")
    content_id = job.job_id.removeprefix("rfj_")

    with (
        patch("src.service.jobs.run_pipeline", new=_build_pipeline_side_effect(content_id)),
        patch("src.service.jobs.upload", side_effect=UploadError("403")),
        patch("src.service.jobs.notify_job_failed") as notify,
    ):
        process_job(job, handles, config, registries)

    fetched = registries.jobs.get(job.job_id)
    assert fetched is not None
    assert fetched.status == "failed"
    assert fetched.stage == "upload"
    assert fetched.retryable is True
    notify.assert_called_once()


def test_process_job_uncaught_exception_marks_failed(
    handles: PipelineHandles, config: ServiceConfig, tmp_path: Path
) -> None:
    registries = Registries()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    job, _ = _register_job_and_batch(registries, input_uri=f"file://{audio}")

    with (
        patch("src.service.jobs.run_pipeline", side_effect=RuntimeError("pipeline crashed")),
        patch("src.service.jobs.notify_job_failed") as notify,
    ):
        process_job(job, handles, config, registries)

    fetched = registries.jobs.get(job.job_id)
    assert fetched is not None
    assert fetched.status == "failed"
    assert fetched.stage == "transcribe"
    assert "pipeline crashed" in (fetched.error or "")
    notify.assert_called_once()


def test_process_job_persists_intermediates_when_configured(handles: PipelineHandles, tmp_path: Path) -> None:
    intermediate_root = tmp_path / "intermediates"
    config = ServiceConfig(device="cpu", intermediate_dir=intermediate_root)
    registries = Registries()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    job, _ = _register_job_and_batch(registries, input_uri=f"file://{audio}")
    content_id = job.job_id.removeprefix("rfj_")

    with (
        patch("src.service.jobs.run_pipeline", new=_build_pipeline_side_effect(content_id)),
        patch("src.service.jobs.upload"),
        patch("src.service.jobs.notify_job_failed"),
    ):
        process_job(job, handles, config, registries)

    persisted_dir = intermediate_root / job.job_id
    assert persisted_dir.is_dir()
    assert (persisted_dir / f"diarization_{content_id}.json").is_file()
    assert (persisted_dir / f"transcription_{content_id}.json").is_file()


def test_finalize_batch_uploads_summary_with_all_terminal_jobs(tmp_path: Path) -> None:
    registries = Registries()
    summary_path = tmp_path / "summary.json"
    batch = Batch(
        batch_id="btc_x",
        summary_uri=f"file://{summary_path}",
        job_ids=["rfj_a", "rfj_b"],
    )
    registries.batches.add(batch)

    # One success, one failure.
    completed = Job(
        job_id="rfj_a",
        batch_id="btc_x",
        input_uri="file:///x.wav",
        output_uri="file:///x.json",
        status="completed",
        started_at=datetime(2026, 5, 19),
        completed_at=datetime(2026, 5, 19, 0, 1),
        duration_seconds=60.0,
    )
    failed = Job(
        job_id="rfj_b",
        batch_id="btc_x",
        input_uri="file:///y.wav",
        output_uri="file:///y.json",
        status="failed",
        started_at=datetime(2026, 5, 19),
        failed_at=datetime(2026, 5, 19, 0, 1),
        stage="transcribe",
        error="boom",
        retryable=False,
    )
    registries.jobs.add(completed)
    registries.jobs.add(failed)

    with patch("src.service.jobs.upload") as upload_mock:
        finalize_batch("btc_x", registries)

    upload_mock.assert_called_once()
    args, _ = upload_mock.call_args
    assert args[0] == f"file://{summary_path}"
    payload = args[1]
    assert payload["batch_id"] == "btc_x"
    assert payload["totals"] == {"submitted": 2, "completed": 1, "failed": 1}
    assert {j["job_id"] for j in payload["jobs"]} == {"rfj_a", "rfj_b"}


def test_finalize_batch_upload_failure_does_not_raise() -> None:
    from src.service.uri_io import UploadError

    registries = Registries()
    registries.batches.add(Batch(batch_id="btc_x", summary_uri="https://x/s.json", job_ids=[]))

    # Must not raise.
    with patch("src.service.jobs.upload", side_effect=UploadError("403")):
        finalize_batch("btc_x", registries)


def test_worker_processes_queued_job_and_finalizes_batch(
    handles: PipelineHandles, config: ServiceConfig, tmp_path: Path
) -> None:
    registries = Registries()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    summary_path = tmp_path / "summary.json"
    job, batch = _register_job_and_batch(
        registries,
        input_uri=f"file://{audio}",
        output_uri=f"file://{tmp_path / 'out.json'}",
        summary_uri=f"file://{summary_path}",
    )
    content_id = job.job_id.removeprefix("rfj_")
    registries.queue.put_nowait(job.job_id)

    fake_run = _build_pipeline_side_effect(content_id)
    with (
        patch("src.service.jobs.run_pipeline", new=fake_run),
        patch("src.service.jobs.notify_job_failed"),
    ):
        worker = Worker(registries, handles, config, get_timeout=0.05)
        worker.start()
        # Wait up to 2 seconds for completion.
        for _ in range(40):
            if registries.batches.get("btc_xyz").completed_at is not None:  # type: ignore[union-attr]
                break
            time.sleep(0.05)
        worker.stop(timeout=2.0)

    fetched_job = registries.jobs.get(job.job_id)
    assert fetched_job is not None
    assert fetched_job.status == "completed"

    # Summary was written.
    assert summary_path.is_file()


def test_worker_continues_after_one_job_fails(handles: PipelineHandles, config: ServiceConfig, tmp_path: Path) -> None:
    registries = Registries()

    # Two jobs in one batch: first fails (bad file://), second succeeds.
    good_audio = tmp_path / "good.wav"
    good_audio.write_bytes(b"x")
    summary_path = tmp_path / "summary.json"

    bad = Job(
        job_id="rfj_aaaaaaaaaaaaaaaa",
        batch_id="btc_dual",
        input_uri="file:///nonexistent/x.wav",
        output_uri=f"file://{tmp_path / 'a.json'}",
    )
    good = Job(
        job_id="rfj_bbbbbbbbbbbbbbbb",
        batch_id="btc_dual",
        input_uri=f"file://{good_audio}",
        output_uri=f"file://{tmp_path / 'b.json'}",
    )
    registries.jobs.add(bad)
    registries.jobs.add(good)
    registries.batches.add(
        Batch(
            batch_id="btc_dual",
            summary_uri=f"file://{summary_path}",
            job_ids=[bad.job_id, good.job_id],
        )
    )
    registries.queue.put_nowait(bad.job_id)
    registries.queue.put_nowait(good.job_id)

    fake_run = _build_pipeline_side_effect(good.job_id.removeprefix("rfj_"))
    with (
        patch("src.service.jobs.run_pipeline", new=fake_run),
        patch("src.service.jobs.notify_job_failed"),
    ):
        worker = Worker(registries, handles, config, get_timeout=0.05)
        worker.start()
        for _ in range(40):
            b = registries.batches.get("btc_dual")
            if b is not None and b.completed_at is not None:
                break
            time.sleep(0.05)
        worker.stop(timeout=2.0)

    assert registries.jobs.get(bad.job_id).status == "failed"  # type: ignore[union-attr]
    assert registries.jobs.get(good.job_id).status == "completed"  # type: ignore[union-attr]
    assert summary_path.is_file()


def test_worker_stop_terminates_thread() -> None:
    registries = Registries()
    handles = PipelineHandles(diarization=MagicMock(), whisperx=MagicMock())
    config = ServiceConfig(device="cpu")
    worker = Worker(registries, handles, config, get_timeout=0.05)
    worker.start()
    worker.stop(timeout=1.0)
    assert worker._thread is None or not worker._thread.is_alive()


# ---------------------------------------------------------------------------
# RetentionSweeper
# ---------------------------------------------------------------------------


from datetime import timedelta  # noqa: E402


def _terminal_completed_job(job_id: str, completed_at: datetime) -> Job:
    return Job(
        job_id=job_id,
        batch_id="btc_x",
        input_uri="file:///x.wav",
        output_uri="file:///x.json",
        status="completed",
        started_at=completed_at,
        completed_at=completed_at,
        duration_seconds=60.0,
    )


def _terminal_failed_job(job_id: str, failed_at: datetime) -> Job:
    return Job(
        job_id=job_id,
        batch_id="btc_x",
        input_uri="file:///x.wav",
        output_uri="file:///x.json",
        status="failed",
        started_at=failed_at,
        failed_at=failed_at,
        stage="transcribe",
        error="boom",
        retryable=False,
    )


def test_retention_sweep_evicts_terminal_jobs_older_than_window():
    registries = Registries()
    config = ServiceConfig(job_retention_seconds=3600)

    now = datetime.now()
    old_completed = _terminal_completed_job("rfj_old1", now - timedelta(hours=2))
    old_failed = _terminal_failed_job("rfj_old2", now - timedelta(hours=2))
    fresh = _terminal_completed_job("rfj_fresh", now - timedelta(minutes=5))
    pending = Job(
        job_id="rfj_pending",
        batch_id="btc_x",
        input_uri="file:///x",
        output_uri="file:///y",
        status="processing",
        started_at=now,  # no terminal timestamp
    )
    for j in (old_completed, old_failed, fresh, pending):
        registries.jobs.add(j)

    sweeper = RetentionSweeper(registries, config)
    evicted_jobs, evicted_batches = sweeper.sweep_once()

    assert evicted_jobs == 2
    assert evicted_batches == 0
    assert registries.jobs.get("rfj_old1") is None
    assert registries.jobs.get("rfj_old2") is None
    assert registries.jobs.get("rfj_fresh") is not None
    assert registries.jobs.get("rfj_pending") is not None


def test_retention_sweep_evicts_terminal_batches_older_than_window():
    registries = Registries()
    config = ServiceConfig(job_retention_seconds=3600)

    now = datetime.now()
    old_batch = Batch(batch_id="btc_old", summary_uri="x", job_ids=[])
    old_batch.completed_at = now - timedelta(hours=2)
    fresh_batch = Batch(batch_id="btc_fresh", summary_uri="x", job_ids=[])
    fresh_batch.completed_at = now - timedelta(minutes=5)
    in_flight = Batch(batch_id="btc_inflight", summary_uri="x", job_ids=[])
    # no completed_at — still in-flight
    for b in (old_batch, fresh_batch, in_flight):
        registries.batches.add(b)

    sweeper = RetentionSweeper(registries, config)
    evicted_jobs, evicted_batches = sweeper.sweep_once()

    # Only the old batch (>1h ago) is evicted. The fresh one (5 min ago) is
    # newer than the cutoff and stays. The in-flight one has no completed_at
    # so it's never eligible.
    assert evicted_jobs == 0
    assert evicted_batches == 1
    assert registries.batches.get("btc_old") is None
    assert registries.batches.get("btc_fresh") is not None
    assert registries.batches.get("btc_inflight") is not None


def test_retention_sweep_handles_empty_registries():
    registries = Registries()
    config = ServiceConfig(job_retention_seconds=3600)
    sweeper = RetentionSweeper(registries, config)
    assert sweeper.sweep_once() == (0, 0)


def test_retention_sweeper_thread_lifecycle():
    registries = Registries()
    config = ServiceConfig(job_retention_seconds=3600)
    sweeper = RetentionSweeper(registries, config, tick_seconds=0.05)
    sweeper.start()
    time.sleep(0.02)
    assert sweeper._thread is not None
    assert sweeper._thread.is_alive()
    sweeper.stop(timeout=1.0)
    assert not sweeper._thread.is_alive()


def test_retention_sweeper_runs_sweep_once_per_tick():
    """Plant an old job, start the sweeper with a fast tick, and verify the job
    is evicted within a couple of tick periods."""
    registries = Registries()
    config = ServiceConfig(job_retention_seconds=1)  # 1 second retention

    # Job is already 5 seconds old → eligible for eviction.
    registries.jobs.add(_terminal_completed_job("rfj_old", datetime.now() - timedelta(seconds=5)))

    sweeper = RetentionSweeper(registries, config, tick_seconds=0.05)
    sweeper.start()
    # Initial wait is one tick, then sweep_once runs.
    for _ in range(40):
        if registries.jobs.get("rfj_old") is None:
            break
        time.sleep(0.05)
    sweeper.stop(timeout=1.0)

    assert registries.jobs.get("rfj_old") is None


def test_retention_sweeper_continues_after_sweep_error():
    """If sweep_once raises, the daemon thread must keep ticking."""
    registries = Registries()
    config = ServiceConfig(job_retention_seconds=1)
    sweeper = RetentionSweeper(registries, config, tick_seconds=0.02)

    call_count = {"n": 0}
    original = sweeper.sweep_once

    def flaky() -> tuple[int, int]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("first sweep blew up")
        return original()

    sweeper.sweep_once = flaky  # type: ignore[method-assign]
    sweeper.start()
    # Wait long enough for several ticks (each ~20ms).
    time.sleep(0.2)
    sweeper.stop(timeout=1.0)

    assert call_count["n"] >= 2, f"sweeper died after first error; only {call_count['n']} calls"
