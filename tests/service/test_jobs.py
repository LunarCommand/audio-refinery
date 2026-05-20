"""Tests for the in-memory data primitives in `src.service.jobs`.

Worker behavior tests land in a subsequent commit. This file covers the
dataclasses, registries, queue, and ID-generation helpers in isolation.
"""

from __future__ import annotations

import queue
import re
import threading
from datetime import datetime

import pytest

from src.service.jobs import (
    Batch,
    BatchRegistry,
    Job,
    JobQueue,
    JobRegistry,
    Registries,
    make_batch_id,
    make_job_id,
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
