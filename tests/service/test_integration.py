"""Service-mode integration tests against the real pipeline.

These exercise the full service stack — FastAPI app, lifespan warmup,
Worker, real pyannote + WhisperX + (optional) sentiment models — against
a known test WAV. Required to verify a built image works end to end, but
slow and resource-heavy: a real warmup runs ~10s, a real per-job pass
runs another 10–60s depending on clip length.

Mock-pipeline coverage of the same paths lives in ``test_e2e.py`` and
runs in CI; this file is operator-triggered via ``make test-integration``.

Requirements:
    - A GPU (or CPU with patience)
    - A test audio file at the path below
    - HF_TOKEN set in env (for pyannote)
    - WhisperX installed (see Makefile install-whisperx)
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

TEST_AUDIO_PATH = Path("/mnt/fast_scratch/test_fixtures/test_audio.wav")

pytestmark = pytest.mark.integration


@pytest.fixture
def integration_audio() -> Path:
    if not TEST_AUDIO_PATH.exists():
        pytest.skip(f"Test audio not found: {TEST_AUDIO_PATH}")
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN is required for pyannote model download")
    return TEST_AUDIO_PATH


def _make_config(*, device: str = "cuda:0") -> ServiceConfig:
    return ServiceConfig(
        device=device,
        # Defaults for the rest. HF_TOKEN is picked up from env by warm_up.
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


def test_real_pipeline_file_uri_end_to_end(integration_audio: Path, tmp_path: Path) -> None:
    """Spin up the real service, POST a job pointing at the test WAV via
    file://, wait for the batch to settle, and verify the transcript JSON
    is a valid CombinedTranscript with non-empty content."""
    output_path = tmp_path / "out" / "transcript.json"
    summary_path = tmp_path / "summary" / "summary.json"

    config = _make_config()
    registries = Registries(queue=JobQueue(maxsize=10))
    app = create_app(
        config,
        registries=registries,
        api_keys={"integration-key"},
        enable_lifespan_warmup=True,  # real warmup
    )

    body = {
        "summary_uri": f"file://{summary_path}",
        "jobs": [
            {
                "input_uri": f"file://{integration_audio}",
                "output_uri": f"file://{output_path}",
            }
        ],
    }

    with TestClient(app) as client:
        # /health should be 503 until warmup completes; allow up to 90s for
        # cold HF Hub downloads on first run, well under HEALTHCHECK 60s
        # start-period + retries.
        _wait(
            lambda: client.get("/health").status_code == 200,
            timeout=120.0,
            poll=1.0,
            what="warmup completion (/health 200)",
        )

        resp = client.post(
            "/transcribe",
            json=body,
            headers={"Authorization": "Bearer integration-key"},
        )
        assert resp.status_code == 202, resp.text
        batch_id = resp.json()["batch_id"]

        # Real pipeline run; allow a few minutes for slow CPUs.
        def _batch_settled() -> bool:
            batch = registries.batches.get(batch_id)
            return batch is not None and batch.completed_at is not None

        _wait(_batch_settled, timeout=300.0, poll=2.0, what=f"batch {batch_id} completion")

    # Transcript landed at the per-job output_uri.
    assert output_path.is_file(), f"transcript missing at {output_path}"
    transcript = json.loads(output_path.read_text())
    assert transcript["schema_version"] == "1.0.0"
    assert transcript["audio"]["duration_seconds"] > 0
    assert transcript["diarization"]["num_speakers"] >= 1
    assert transcript["transcription"]["language"]
    assert transcript["transcription"]["segments"], "expected at least one transcript segment"
    # Model-versions block records what produced the transcript.
    assert transcript["model_versions"]["diarization"]
    assert transcript["model_versions"]["transcription"]

    # Batch summary landed and captures the success.
    assert summary_path.is_file(), f"summary missing at {summary_path}"
    summary = json.loads(summary_path.read_text())
    assert summary["batch_id"] == batch_id
    assert summary["totals"] == {"submitted": 1, "completed": 1, "failed": 0}
    assert summary["jobs"][0]["status"] == "completed"
    assert summary["jobs"][0]["duration_seconds"] > 0
