"""Tests for `src.service.app` — the FastAPI HTTP surface.

Uses FastAPI's TestClient with `enable_lifespan_warmup=False` so the real
warm_up() never runs. Tests inject pre-built handles where the worker needs
to actually consume a job; the rest just exercise the HTTP envelope.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.service.app import create_app
from src.service.config import PipelineHandles, ServiceConfig
from src.service.jobs import Job, Registries
from src.service.lifecycle import ServiceReadiness

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _config() -> ServiceConfig:
    return ServiceConfig(
        device="cpu",
        max_queue_size=10,
        max_batch_size=5,
    )


def _ready_readiness() -> ServiceReadiness:
    r = ServiceReadiness()
    r.mark_ready()
    return r


def _client(
    *,
    api_keys: set[str] | None = None,
    readiness: ServiceReadiness | None = None,
    registries: Registries | None = None,
    handles: PipelineHandles | None = None,
) -> TestClient:
    # Default to a ready readiness object so the happy-path /transcribe and
    # /jobs tests don't have to opt in. Tests that exercise the readiness
    # gate explicitly pass a loading/failed readiness.
    app = create_app(
        _config(),
        registries=registries,
        readiness=readiness if readiness is not None else _ready_readiness(),
        api_keys=api_keys if api_keys is not None else {"test-key"},
        handles=handles,
        enable_lifespan_warmup=False,
    )
    return TestClient(app)


_AUTH = {"Authorization": "Bearer test-key"}


def _body(
    *,
    n_jobs: int = 1,
    summary_uri: str = "file:///tmp/summary.json",
) -> dict:
    return {
        "summary_uri": summary_uri,
        "jobs": [
            {
                "input_uri": f"file:///inbox/in_{i}.wav",
                "output_uri": f"file:///outbox/out_{i}.json",
            }
            for i in range(n_jobs)
        ],
    }


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_503_while_loading():
    readiness = ServiceReadiness()  # default state: loading
    resp = _client(readiness=readiness).get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "loading"


def test_health_returns_200_after_ready():
    readiness = ServiceReadiness()
    readiness.mark_ready()
    resp = _client(readiness=readiness).get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_health_returns_503_with_stage_on_failed():
    readiness = ServiceReadiness()
    readiness.mark_failed("transcription", "CUDA OOM")
    resp = _client(readiness=readiness).get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "failed"
    assert body["stage"] == "transcription"
    assert "CUDA OOM" in body["detail"]


def test_health_does_not_require_auth():
    resp = _client().get("/health")
    # No Authorization header. Should not 401 even though /health returns 503.
    assert resp.status_code in (200, 503)


# ---------------------------------------------------------------------------
# POST /transcribe — happy path
# ---------------------------------------------------------------------------


def test_transcribe_single_job_returns_202_with_batch_and_job_ids():
    client = _client()
    resp = client.post("/transcribe", json=_body(n_jobs=1), headers=_AUTH)
    assert resp.status_code == 202
    body = resp.json()
    assert body["batch_id"].startswith("btc_")
    assert isinstance(body["job_ids"], list)
    assert len(body["job_ids"]) == 1
    assert body["job_ids"][0].startswith("rfj_")


def test_transcribe_multi_job_returns_index_aligned_job_ids():
    client = _client()
    resp = client.post("/transcribe", json=_body(n_jobs=3), headers=_AUTH)
    assert resp.status_code == 202
    body = resp.json()
    assert len(body["job_ids"]) == 3
    assert all(jid.startswith("rfj_") for jid in body["job_ids"])
    assert len(set(body["job_ids"])) == 3  # all unique


def test_transcribe_registers_jobs_and_batch_and_enqueues():
    registries = Registries()
    # Need a queue that matches the config's max_queue_size for the size check.
    from src.service.jobs import JobQueue

    registries = Registries(queue=JobQueue(maxsize=10))
    client = _client(registries=registries)

    resp = client.post("/transcribe", json=_body(n_jobs=2), headers=_AUTH)
    assert resp.status_code == 202
    body = resp.json()

    # Jobs registered.
    for jid in body["job_ids"]:
        assert registries.jobs.get(jid) is not None
    # Batch registered with correct pending count.
    batch = registries.batches.get(body["batch_id"])
    assert batch is not None
    assert batch.pending_count == 2
    # Queue depth reflects the two enqueued ids.
    assert registries.queue.qsize() == 2


# ---------------------------------------------------------------------------
# POST /transcribe — validation errors
# ---------------------------------------------------------------------------


def test_transcribe_missing_summary_uri_returns_400():
    client = _client()
    body = _body(n_jobs=1)
    body.pop("summary_uri")
    resp = client.post("/transcribe", json=body, headers=_AUTH)
    assert resp.status_code in (400, 422)


def test_transcribe_empty_jobs_returns_400():
    client = _client()
    body = _body(n_jobs=0)
    resp = client.post("/transcribe", json=body, headers=_AUTH)
    assert resp.status_code in (400, 422)


def test_transcribe_bad_input_scheme_returns_400():
    client = _client()
    body = _body(n_jobs=1)
    body["jobs"][0]["input_uri"] = "ftp://nope/x.wav"
    resp = client.post("/transcribe", json=body, headers=_AUTH)
    assert resp.status_code in (400, 422)
    assert "scheme" in resp.text.lower() or "ftp" in resp.text.lower()


def test_transcribe_bad_output_scheme_returns_400():
    client = _client()
    body = _body(n_jobs=1)
    body["jobs"][0]["output_uri"] = "s3://nope/x.json"
    resp = client.post("/transcribe", json=body, headers=_AUTH)
    assert resp.status_code in (400, 422)


def test_transcribe_bad_summary_scheme_returns_400():
    client = _client()
    body = _body(n_jobs=1, summary_uri="redis://localhost/summary")
    resp = client.post("/transcribe", json=body, headers=_AUTH)
    assert resp.status_code in (400, 422)


def test_transcribe_batch_exceeding_max_returns_400():
    client = _client()
    # _config sets max_batch_size=5
    body = _body(n_jobs=6)
    resp = client.post("/transcribe", json=body, headers=_AUTH)
    assert resp.status_code == 400
    assert "batch_too_large" in resp.text


# ---------------------------------------------------------------------------
# POST /transcribe — auth + queue capacity
# ---------------------------------------------------------------------------


def test_transcribe_returns_503_during_warmup():
    """While ServiceReadiness is in the `loading` state (warmup in progress),
    POST /transcribe must NOT accept work — same readiness contract as /health."""
    loading = ServiceReadiness()  # default state: loading
    client = _client(readiness=loading)
    resp = client.post("/transcribe", json=_body(), headers=_AUTH)
    assert resp.status_code == 503
    body = resp.json()
    assert body["detail"]["error"] == "service_not_ready"
    assert body["detail"]["state"] == "loading"
    assert resp.headers.get("retry-after") == "5"


def test_transcribe_returns_503_when_warmup_failed():
    """When warm_up raised and ServiceReadiness is `failed`, POST /transcribe
    returns 503. Crucially this prevents zombie jobs from queueing in a
    container that's about to be restarted by the orchestrator."""
    failed = ServiceReadiness()
    failed.mark_failed("transcription", "CUDA OOM")
    client = _client(readiness=failed)
    resp = client.post("/transcribe", json=_body(), headers=_AUTH)
    assert resp.status_code == 503
    body = resp.json()
    assert body["detail"]["error"] == "service_not_ready"
    assert body["detail"]["state"] == "failed"
    assert body["detail"]["stage"] == "transcription"


def test_transcribe_without_auth_returns_401():
    client = _client()
    resp = client.post("/transcribe", json=_body())
    assert resp.status_code == 401


def test_transcribe_with_wrong_bearer_returns_401():
    client = _client()
    resp = client.post(
        "/transcribe",
        json=_body(),
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401


def test_transcribe_queue_full_returns_429():
    from src.service.jobs import JobQueue

    # Tiny queue: 2 slots.
    registries = Registries(queue=JobQueue(maxsize=2))
    # Pre-fill the queue.
    registries.queue.put_nowait("preexisting_1")
    registries.queue.put_nowait("preexisting_2")

    client = _client(registries=registries)
    resp = client.post("/transcribe", json=_body(n_jobs=1), headers=_AUTH)
    assert resp.status_code == 429
    assert "queue_full" in resp.text


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------


def test_get_job_returns_status_for_known_job():
    registries = Registries()
    job = Job(
        job_id="rfj_abc",
        batch_id="btc_xyz",
        input_uri="file:///in.wav",
        output_uri="file:///out.json",
        status="processing",
        started_at=datetime(2026, 5, 19, 22, 0, 0),
    )
    registries.jobs.add(job)

    resp = _client(registries=registries).get("/jobs/rfj_abc", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "rfj_abc"
    assert body["batch_id"] == "btc_xyz"
    assert body["status"] == "processing"
    assert body["started_at"] is not None
    assert body["completed_at"] is None
    assert body["failed_at"] is None


def test_get_job_includes_failure_detail_for_failed_job():
    registries = Registries()
    job = Job(
        job_id="rfj_fail",
        batch_id="btc_x",
        input_uri="file:///in.wav",
        output_uri="file:///out.json",
        status="failed",
        started_at=datetime(2026, 5, 19),
        failed_at=datetime(2026, 5, 19, 0, 1),
        stage="transcribe",
        error="ValueError: bad audio",
        retryable=False,
    )
    registries.jobs.add(job)

    resp = _client(registries=registries).get("/jobs/rfj_fail", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["stage"] == "transcribe"
    assert body["error"] == "ValueError: bad audio"
    assert body["retryable"] is False


def test_get_job_unknown_returns_404():
    resp = _client().get("/jobs/rfj_nonexistent", headers=_AUTH)
    assert resp.status_code == 404
    assert "job_not_found" in resp.text


def test_get_job_requires_auth():
    resp = _client().get("/jobs/rfj_abc")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Lifespan: worker / sweeper starts when handles provided
# ---------------------------------------------------------------------------


def test_resolve_scratch_location_uses_config_when_set(tmp_path: Path):
    from src.service.app import _resolve_scratch_location

    config = ServiceConfig(scratch_dir=tmp_path / "rfj_scratch")
    resolved, _fstype = _resolve_scratch_location(config)
    assert resolved == tmp_path / "rfj_scratch"


def test_resolve_scratch_location_falls_back_to_tempfile_default():
    import tempfile

    from src.service.app import _resolve_scratch_location

    config = ServiceConfig()  # scratch_dir defaults to None
    resolved, _fstype = _resolve_scratch_location(config)
    assert resolved == Path(tempfile.gettempdir())


def test_lifespan_starts_worker_and_sweeper_when_handles_supplied():
    handles = PipelineHandles(diarization=MagicMock(), whisperx=MagicMock())
    app = create_app(
        _config(),
        api_keys={"k"},
        handles=handles,
        enable_lifespan_warmup=False,
    )
    # Using TestClient as a context manager runs the lifespan handler.
    with TestClient(app) as client:
        # Trigger an endpoint so we know the app is up.
        resp = client.get("/health")
        assert resp.status_code in (200, 503)
        assert app.state.worker is not None
        assert app.state.sweeper is not None
    # After context exit, both should be stopped.
    assert app.state.worker is not None  # reference retained
