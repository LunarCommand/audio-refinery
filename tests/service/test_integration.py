"""Service-mode integration tests against the real pipeline.

These exercise the full service stack — FastAPI app, lifespan warmup,
Worker, real pyannote + WhisperX + (optional) sentiment models — against
real WAV files. Slow and resource-heavy: a real warmup runs ~10s, a real
per-job pass runs another 10–60s depending on clip length.

Mock-pipeline coverage of the same paths lives in ``test_e2e.py`` and runs
in CI; this file is operator-triggered via ``make test-integration``.

Audio fixtures come from the shared ``integration_audio_files`` fixture in
``tests/conftest.py``. Point at your own WAV directory via::

    export REFINERY_TEST_AUDIO_DIR=/path/to/wavs
    export HF_TOKEN=hf_xxx
    make test-integration

Or drop ``.wav`` files into ``tests/_audio_fixtures/`` (gitignored).

Tests with the ``integration_audio_files`` fixture skip cleanly when no
files are present or HF_TOKEN is unset.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.service.app import create_app
from src.service.config import ServiceConfig
from src.service.jobs import JobQueue, Registries

pytestmark = pytest.mark.integration


def _make_config(*, device: str = "cuda:0") -> ServiceConfig:
    return ServiceConfig(
        device=device,
        hf_token=os.getenv("HF_TOKEN", ""),
        # Disable retention sweeping during tests — we want to read final state.
        job_retention_seconds=3600,
        # Disable thermal guard so the test doesn't get killed if a CI box runs hot.
        gpu_temp_limit_celsius=0,
    )


def _wait(predicate, *, timeout: float, poll: float = 0.5, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(poll)
    raise AssertionError(f"{what} did not occur within {timeout}s")


def _require_hf_token() -> None:
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN is required for pyannote model download")


def _run_batch(
    *,
    audio_files: list[Path],
    output_dir: Path,
    summary_path: Path,
    warmup_timeout: float = 120.0,
    batch_timeout: float = 600.0,
) -> tuple[str, list[str]]:
    """Spin up the real service, POST a batch with one job per audio file,
    wait for completion, return ``(batch_id, job_ids)``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    config = _make_config()
    registries = Registries(queue=JobQueue(maxsize=max(10, len(audio_files) + 2)))
    app = create_app(
        config,
        registries=registries,
        api_keys={"integration-key"},
        enable_lifespan_warmup=True,
    )

    body = {
        "summary_uri": f"file://{summary_path}",
        "jobs": [
            {
                "input_uri": f"file://{audio}",
                "output_uri": f"file://{output_dir / f'transcript_{i}.json'}",
            }
            for i, audio in enumerate(audio_files)
        ],
    }

    with TestClient(app) as client:
        _wait(
            lambda: client.get("/health").status_code == 200,
            timeout=warmup_timeout,
            poll=1.0,
            what="warmup completion (/health 200)",
        )

        resp = client.post(
            "/transcribe",
            json=body,
            headers={"Authorization": "Bearer integration-key"},
        )
        assert resp.status_code == 202, resp.text
        payload = resp.json()
        batch_id = payload["batch_id"]
        job_ids = payload["job_ids"]

        def _batch_settled() -> bool:
            batch = registries.batches.get(batch_id)
            return batch is not None and batch.completed_at is not None

        _wait(_batch_settled, timeout=batch_timeout, poll=2.0, what=f"batch {batch_id} completion")

    return batch_id, job_ids


def test_real_pipeline_file_uri_end_to_end(integration_audio: Path, tmp_path: Path) -> None:
    """One real job via file://: assert transcript at output_uri + summary
    at summary_uri are both valid documents."""
    _require_hf_token()
    output_dir = tmp_path / "out"
    summary_path = tmp_path / "summary" / "summary.json"

    batch_id, _ = _run_batch(
        audio_files=[integration_audio],
        output_dir=output_dir,
        summary_path=summary_path,
    )

    # Transcript landed and is well-formed.
    transcript_path = output_dir / "transcript_0.json"
    assert transcript_path.is_file(), f"transcript missing at {transcript_path}"
    transcript = json.loads(transcript_path.read_text())
    assert transcript["schema_version"] == "1.0.0"
    assert transcript["audio"]["duration_seconds"] > 0
    assert transcript["diarization"]["num_speakers"] >= 1
    assert transcript["transcription"]["language"]
    assert transcript["transcription"]["segments"], "expected at least one transcript segment"
    assert transcript["model_versions"]["diarization"]
    assert transcript["model_versions"]["transcription"]

    # Batch summary landed and captures the success.
    assert summary_path.is_file(), f"summary missing at {summary_path}"
    summary = json.loads(summary_path.read_text())
    assert summary["batch_id"] == batch_id
    assert summary["totals"] == {"submitted": 1, "completed": 1, "failed": 0}
    assert summary["jobs"][0]["status"] == "completed"
    assert summary["jobs"][0]["duration_seconds"] > 0


def test_real_pipeline_multi_job_batch(integration_audio_files: list[Path], tmp_path: Path) -> None:
    """Run a batch with every available WAV. Verifies the batch-summary path
    against the real pipeline: each successful job writes its own transcript,
    and one summary at the end captures every job in submission order."""
    _require_hf_token()
    if len(integration_audio_files) < 2:
        pytest.skip(f"multi-job batch test requires at least 2 WAVs; only {len(integration_audio_files)} present")

    output_dir = tmp_path / "out"
    summary_path = tmp_path / "summary" / "summary.json"

    # Batch processing is serial, so scale the timeout with the input count.
    per_job_budget_seconds = 300.0
    batch_timeout = per_job_budget_seconds * len(integration_audio_files)

    batch_id, submitted_job_ids = _run_batch(
        audio_files=integration_audio_files,
        output_dir=output_dir,
        summary_path=summary_path,
        batch_timeout=batch_timeout,
    )

    # Every job's transcript landed.
    for i in range(len(integration_audio_files)):
        path = output_dir / f"transcript_{i}.json"
        assert path.is_file(), f"transcript {i} missing at {path}"
        transcript = json.loads(path.read_text())
        assert transcript["schema_version"] == "1.0.0"
        assert transcript["transcription"]["segments"], f"transcript {i} has no segments"

    # Summary captures every job in submission order with totals matching.
    assert summary_path.is_file(), f"summary missing at {summary_path}"
    summary = json.loads(summary_path.read_text())
    assert summary["batch_id"] == batch_id
    assert summary["totals"]["submitted"] == len(integration_audio_files)
    assert summary["totals"]["completed"] == len(integration_audio_files)
    assert summary["totals"]["failed"] == 0
    summary_job_ids = [j["job_id"] for j in summary["jobs"]]
    assert summary_job_ids == submitted_job_ids, "summary order should match POST order"
    # Each entry has a real duration recorded.
    for entry in summary["jobs"]:
        assert entry["status"] == "completed"
        assert entry["duration_seconds"] > 0
