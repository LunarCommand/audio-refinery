"""Service-mode end-to-end tests through the full FastAPI + Worker stack.

The pipeline itself is mocked (no GPU required) but every other layer runs
real: the HTTP endpoint accepts the request, the request goes through auth +
URI validation + queue, the Worker thread pulls and processes it, the URI I/O
layer writes the transcript to ``output_uri``, and the batch summary lands at
``summary_uri`` after the batch settles. The mock-pipeline ``side_effect``
populates the per-stage JSON files the worker expects on disk.

These complement ``test_app.py`` (HTTP boundary in isolation) and ``test_jobs.py``
(Worker in isolation) by exercising the full chain together. Real-pipeline
GPU-required tests live in ``test_integration.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from src.pipeline import FileOutcome, PipelineResult, StageResult
from src.service.app import create_app
from src.service.config import PipelineHandles, ServiceConfig
from src.service.jobs import JobQueue, Registries

_AUTH = {"Authorization": "Bearer e2e-key"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(tmp_path: Path) -> ServiceConfig:
    return ServiceConfig(
        device="cpu",
        max_queue_size=10,
        max_batch_size=10,
        # Retention long enough that jobs never get swept during the test.
        job_retention_seconds=3600,
    )


def _build_app(tmp_path: Path) -> tuple[TestClient, Registries]:
    handles = PipelineHandles(diarization=MagicMock(name="diar"), whisperx=MagicMock(name="wx"))
    registries = Registries(queue=JobQueue(maxsize=10))
    app = create_app(
        _config(tmp_path),
        registries=registries,
        api_keys={"e2e-key"},
        handles=handles,
    )
    return TestClient(app), registries


def _pipeline_side_effect(content_id: str):
    """Return a side_effect callable that simulates ``run_pipeline`` by writing
    minimal valid per-stage JSON files into the configured directories and
    returning a success-only ``PipelineResult``."""

    def fake_run_pipeline(
        *, source_dir, demucs_output_dir, diarization_dir, transcription_dir, sentiment_dir, **_kwargs
    ):
        diarization_dir.mkdir(parents=True, exist_ok=True)
        transcription_dir.mkdir(parents=True, exist_ok=True)
        audio_blob = {
            "path": str(source_dir / f"{content_id}.wav"),
            "sample_rate": 16000,
            "channels": 1,
            "duration_seconds": 1.0,
            "frames": 16000,
            "format_str": "WAV",
            "subtype": "PCM_16",
        }
        ts = "2026-05-20T22:00:00"
        (diarization_dir / f"diarization_{content_id}.json").write_text(
            json.dumps(
                {
                    "input_file": audio_blob["path"],
                    "input_info": audio_blob,
                    "segments": [],
                    "num_speakers": 0,
                    "device": "cpu",
                    "processing_time_seconds": 0.1,
                    "started_at": ts,
                    "completed_at": ts,
                }
            )
        )
        (transcription_dir / f"transcription_{content_id}.json").write_text(
            json.dumps(
                {
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
            )
        )
        return PipelineResult(
            total_discovered=1,
            separation=StageResult(outcomes=[FileOutcome(content_id=content_id, stage="separate", success=True)]),
            diarization=StageResult(outcomes=[FileOutcome(content_id=content_id, stage="diarize", success=True)]),
            transcription=StageResult(outcomes=[FileOutcome(content_id=content_id, stage="transcribe", success=True)]),
        )

    return fake_run_pipeline


def _wait_for_summary(summary_path: Path, *, timeout: float = 5.0) -> None:
    """Block until the summary JSON appears on disk, or fail."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if summary_path.is_file():
            return
        time.sleep(0.02)
    raise AssertionError(f"summary never appeared at {summary_path} within {timeout}s")


def _wait_for_batch_completion(registries: Registries, batch_id: str, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        batch = registries.batches.get(batch_id)
        if batch is not None and batch.completed_at is not None:
            return
        time.sleep(0.02)
    raise AssertionError(f"batch {batch_id} never settled within {timeout}s")


# ---------------------------------------------------------------------------
# file:// end-to-end
# ---------------------------------------------------------------------------


def test_e2e_file_uri_single_job_writes_transcript_and_summary(tmp_path: Path) -> None:
    """POST a single file:// job and verify the transcript lands at output_uri
    and the summary lands at summary_uri after the batch settles."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 64)
    output_path = tmp_path / "out" / "transcript.json"
    summary_path = tmp_path / "summaries" / "summary.json"

    client, registries = _build_app(tmp_path)
    body = {
        "summary_uri": f"file://{summary_path}",
        "jobs": [
            {
                "input_uri": f"file://{audio}",
                "output_uri": f"file://{output_path}",
            }
        ],
    }

    # Dynamic side_effect derives content_id from the source_dir at call time
    # so the patch can sit in the outer `with` block. This closes the race
    # where the worker could pull the job and call the real run_pipeline
    # before a post-POST patch took effect.
    def dynamic_run_pipeline(*, source_dir, **kwargs):
        matches = list(source_dir.glob("*.wav"))
        assert len(matches) == 1, f"expected exactly one audio file, got {matches}"
        content_id = matches[0].stem
        return _pipeline_side_effect(content_id)(source_dir=source_dir, **kwargs)

    with (
        client,  # context manager runs lifespan so the worker starts
        patch("src.service.jobs.run_pipeline", new=dynamic_run_pipeline),
        patch("src.service.jobs.notify_job_failed"),
    ):
        resp = client.post("/transcribe", json=body, headers=_AUTH)
        assert resp.status_code == 202
        payload = resp.json()
        batch_id = payload["batch_id"]
        job_id_placeholder = payload["job_ids"][0]
        _wait_for_batch_completion(registries, batch_id, timeout=5.0)

    # Transcript landed at the per-job output_uri.
    assert output_path.is_file()
    transcript = json.loads(output_path.read_text())
    assert transcript["schema_version"] == "1.0.0"
    assert transcript["diarization"]["num_speakers"] == 0
    assert transcript["transcription"]["language"] == "en"

    # Summary landed at the per-batch summary_uri.
    _wait_for_summary(summary_path)
    summary = json.loads(summary_path.read_text())
    assert summary["batch_id"] == batch_id
    assert summary["totals"] == {"submitted": 1, "completed": 1, "failed": 0}
    assert len(summary["jobs"]) == 1
    assert summary["jobs"][0]["status"] == "completed"
    assert summary["jobs"][0]["job_id"] == job_id_placeholder


def test_e2e_file_uri_failed_input_writes_summary_only(tmp_path: Path) -> None:
    """Bad file:// input → worker fails the job during download. No transcript
    lands at output_uri. Batch summary captures the failure."""
    output_path = tmp_path / "out" / "should_not_exist.json"
    summary_path = tmp_path / "summaries" / "summary.json"

    client, registries = _build_app(tmp_path)
    body = {
        "summary_uri": f"file://{summary_path}",
        "jobs": [
            {
                "input_uri": "file:///nonexistent/input.wav",
                "output_uri": f"file://{output_path}",
            }
        ],
    }

    with client, patch("src.service.jobs.notify_job_failed"):
        resp = client.post("/transcribe", json=body, headers=_AUTH)
        assert resp.status_code == 202
        batch_id = resp.json()["batch_id"]
        _wait_for_batch_completion(registries, batch_id, timeout=5.0)

    # No transcript was written.
    assert not output_path.exists()

    # Summary captures the failure.
    _wait_for_summary(summary_path)
    summary = json.loads(summary_path.read_text())
    assert summary["totals"] == {"submitted": 1, "completed": 0, "failed": 1}
    entry = summary["jobs"][0]
    assert entry["status"] == "failed"
    assert entry["stage"] == "download"
    assert entry["retryable"] is True


def test_e2e_multi_job_batch_processes_in_submission_order(tmp_path: Path) -> None:
    """POST a 3-job batch. Worker processes them serially in submission order.
    Summary captures all three in order."""
    audios = []
    outputs = []
    for i in range(3):
        a = tmp_path / f"audio_{i}.wav"
        a.write_bytes(b"\x00" * 64)
        audios.append(a)
        outputs.append(tmp_path / "out" / f"transcript_{i}.json")
    summary_path = tmp_path / "summaries" / "summary.json"

    client, registries = _build_app(tmp_path)
    body = {
        "summary_uri": f"file://{summary_path}",
        "jobs": [
            {"input_uri": f"file://{a}", "output_uri": f"file://{o}"} for a, o in zip(audios, outputs, strict=True)
        ],
    }

    with client, patch("src.service.jobs.notify_job_failed"):
        # Dynamic side_effect: each call writes the JSON files keyed off the
        # actual content_id (the worker derives it from the job_id).
        def dynamic_side_effect(*, source_dir, **kwargs):
            # Find the audio file the worker placed in source_dir.
            matches = list(source_dir.glob("*.wav"))
            assert len(matches) == 1
            content_id = matches[0].stem
            return _pipeline_side_effect(content_id)(source_dir=source_dir, **kwargs)

        with patch("src.service.jobs.run_pipeline", new=dynamic_side_effect):
            resp = client.post("/transcribe", json=body, headers=_AUTH)
            assert resp.status_code == 202
            payload = resp.json()
            batch_id = payload["batch_id"]
            submitted_job_ids = payload["job_ids"]

            _wait_for_batch_completion(registries, batch_id, timeout=10.0)

    # All three transcripts written.
    for o in outputs:
        assert o.is_file(), f"expected transcript at {o}"

    # Summary order matches submission order.
    _wait_for_summary(summary_path)
    summary = json.loads(summary_path.read_text())
    assert summary["totals"] == {"submitted": 3, "completed": 3, "failed": 0}
    summary_job_ids = [j["job_id"] for j in summary["jobs"]]
    assert summary_job_ids == submitted_job_ids


# ---------------------------------------------------------------------------
# https:// end-to-end via httpx MockTransport
# ---------------------------------------------------------------------------


def test_e2e_https_uri_routes_fetch_and_upload_through_httpx(tmp_path: Path) -> None:
    """POST an HTTPS job; verify the worker hits the input URL with GET and
    uploads to the output URL with PUT. summary upload PUTs the batch envelope."""

    # Captured by the mock transport.
    captured: dict[str, Any] = {"requests": []}

    fake_audio = b"\x00" * 2048  # bytes the worker sees as the downloaded audio

    def handler(request: httpx.Request) -> httpx.Response:
        captured["requests"].append(
            {
                "method": request.method,
                "url": str(request.url),
                "body": request.read(),
                "content_type": request.headers.get("Content-Type"),
            }
        )
        if request.method == "GET":
            return httpx.Response(200, content=fake_audio)
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)

    def patched_stream(method, url, **kwargs):
        return httpx.Client(transport=transport).stream(method, url, **kwargs)

    def patched_put(url, **kwargs):
        with httpx.Client(transport=transport) as c:
            return c.put(url, **kwargs)

    # Dynamic side_effect — derives content_id from the source_dir the worker
    # creates per-job, so the patch can be applied BEFORE the POST. This
    # eliminates a race where the worker thread could pull the job and call
    # the real run_pipeline before a post-POST patch takes effect.
    def dynamic_run_pipeline(*, source_dir, **kwargs):
        matches = list(source_dir.glob("*.wav"))
        assert len(matches) == 1, f"expected exactly one audio file, got {matches}"
        content_id = matches[0].stem
        return _pipeline_side_effect(content_id)(source_dir=source_dir, **kwargs)

    client, registries = _build_app(tmp_path)
    input_uri = "https://bucket.example.com/audio/x.wav?X-Amz-Signature=abc"
    output_uri = "https://bucket.example.com/transcript/x.json?X-Amz-Signature=def"
    summary_uri = "https://bucket.example.com/summary/btc_x.json?X-Amz-Signature=ghi"
    body = {
        "summary_uri": summary_uri,
        "jobs": [{"input_uri": input_uri, "output_uri": output_uri}],
    }

    with (
        client,
        patch("src.service.uri_io.httpx.stream", new=patched_stream),
        patch("src.service.uri_io.httpx.put", new=patched_put),
        patch("src.service.jobs.run_pipeline", new=dynamic_run_pipeline),
        patch("src.service.jobs.notify_job_failed"),
    ):
        resp = client.post("/transcribe", json=body, headers=_AUTH)
        assert resp.status_code == 202
        batch_id = resp.json()["batch_id"]
        _wait_for_batch_completion(registries, batch_id, timeout=5.0)

    # Three HTTP calls: GET input, PUT transcript, PUT summary.
    methods_and_urls = [(r["method"], r["url"].split("?")[0]) for r in captured["requests"]]
    assert ("GET", "https://bucket.example.com/audio/x.wav") in methods_and_urls
    assert ("PUT", "https://bucket.example.com/transcript/x.json") in methods_and_urls
    assert ("PUT", "https://bucket.example.com/summary/btc_x.json") in methods_and_urls

    # Both PUT bodies are valid JSON with the expected schema versions.
    put_requests = [r for r in captured["requests"] if r["method"] == "PUT"]
    for r in put_requests:
        assert r["content_type"] == "application/json"
        payload = json.loads(r["body"])
        assert payload["schema_version"] == "1.0.0"
