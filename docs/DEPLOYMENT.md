# Deployment

This document covers running audio-refinery in production: sustained unattended batch runs,
integration with larger systems via an async worker pattern, containerization for cloud
deployments, and operational monitoring.

## Single-Machine Production Setup

For sustained batch processing on a local GPU workstation, a few practices make long-running
jobs reliable and recoverable.

### Environment and Startup

```bash
# Activate the virtualenv
source /path/to/audio-refinery/.venv/bin/activate

# Verify GPU is accessible and not occupied
audio-refinery pipeline --base-dir /data/audio/batch
```

The pipeline runs a GPU pre-flight check before loading any models. If active processes are
found on the target GPU, it prints the PID and VRAM footprint of each and asks for confirmation
before proceeding. This prevents inadvertently loading models onto a GPU already under load.

### RAM Disk

Mount a tmpfs before starting any batch run that uses the default Demucs output directory:

```bash
sudo mkdir -p /mnt/fast_scratch
sudo mount -t tmpfs -o size=32G,mode=1777 tmpfs /mnt/fast_scratch
```

To persist across reboots, add to `/etc/fstab`:

```
tmpfs /mnt/fast_scratch tmpfs defaults,size=32G,mode=1777 0 0
```

Size the RAM disk to at least 2× the largest expected file's stem output (~400 MB for a typical
5–8 minute file). 8–32 GB provides comfortable headroom for longer recordings.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the rationale behind the RAM disk strategy.

### Running a Batch

```bash
# Standard batch run
audio-refinery pipeline --base-dir /data/audio/batch --compute-type int8_float16

# Dual-GPU
audio-refinery pipeline-parallel \
  --base-dir /data/audio/batch \
  --compute-type int8_float16 \
  --power-limit 350

# Background with logging
nohup audio-refinery pipeline \
  --base-dir /data/audio/batch \
  --compute-type int8_float16 \
  > /data/audio/batch/logs/pipeline.log 2>&1 &
```

### Resume and Recovery

The pipeline skips files whose output already exists and is non-empty. An interrupted run —
due to power loss, thermal shutdown, or manual termination — can be restarted with the same
command. Completed files are not reprocessed.

```bash
# Resume an interrupted run (default behavior — no flags needed)
audio-refinery pipeline --base-dir /data/audio/batch

# Force full reprocessing
audio-refinery pipeline --base-dir /data/audio/batch --no-resume
```

---

## Async Worker Pattern

For applications that need to submit audio processing jobs programmatically and retrieve results
asynchronously, audio-refinery can be deployed as a headless worker backed by a job queue.

### Why Not a Synchronous API

Audio processing time is non-deterministic and can range from seconds to minutes per file
depending on duration and diarization complexity. Blocking an HTTP request for this duration
leads to timeouts, poor resource management, and fragile client logic.

The recommended pattern is a **database-as-queue**: a client inserts job records; the worker
polls, locks, and processes them; results are written back to the database.

### PostgreSQL Job Queue Schema

```sql
CREATE TABLE refinery_jobs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status      TEXT NOT NULL DEFAULT 'queued',   -- queued | processing | completed | failed
    input_path  TEXT NOT NULL,
    output_dir  TEXT NOT NULL,
    device      TEXT NOT NULL DEFAULT 'cuda',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at  TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    result_json JSONB,
    error_msg   TEXT
);

CREATE INDEX ON refinery_jobs (status, created_at);
```

### Worker Loop Pattern

```python
import time
import json
from pathlib import Path
from sqlalchemy import create_engine, text

from src.separator import separate
from src.diarizer import diarize
from src.transcriber import transcribe

engine = create_engine("postgresql://user:pass@localhost/mydb")

def claim_job(conn):
    """Atomically claim the oldest queued job."""
    row = conn.execute(text("""
        UPDATE refinery_jobs
        SET status = 'processing', started_at = now()
        WHERE id = (
            SELECT id FROM refinery_jobs
            WHERE status = 'queued'
            ORDER BY created_at
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, input_path, output_dir, device
    """)).fetchone()
    conn.commit()
    return row

def run_worker():
    while True:
        with engine.connect() as conn:
            job = claim_job(conn)
            if job is None:
                time.sleep(5)
                continue

            try:
                sep = separate(job.input_path, output_dir=job.output_dir, device=job.device)
                diar = diarize(sep.vocals_path, device=job.device)
                tx = transcribe(sep.vocals_path, diarization=diar, device=job.device)

                conn.execute(text("""
                    UPDATE refinery_jobs
                    SET status = 'completed', completed_at = now(), result_json = :result
                    WHERE id = :id
                """), {"id": job.id, "result": json.dumps(tx.model_dump())})
                conn.commit()

            except Exception as exc:
                conn.execute(text("""
                    UPDATE refinery_jobs
                    SET status = 'failed', completed_at = now(), error_msg = :err
                    WHERE id = :id
                """), {"id": job.id, "err": str(exc)})
                conn.commit()
```

The `FOR UPDATE SKIP LOCKED` clause ensures that multiple worker processes can pull from the
same queue without racing on the same job — a second worker process on a different GPU skips
any row that the first worker has already locked.

### Submitting Jobs

```python
with engine.connect() as conn:
    conn.execute(text("""
        INSERT INTO refinery_jobs (input_path, output_dir, device)
        VALUES (:input, :output, :device)
    """), {"input": "/data/audio/file.wav", "output": "/data/results", "device": "cuda:0"})
    conn.commit()
```

---

## Docker Containerization

Containerizing audio-refinery ensures that the PyTorch 2.1.2 + CUDA 12.1 dependency stack is
portable and reproducible across machines, including cloud GPU instances.

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 refinery
WORKDIR /app
USER refinery

# Install uv
RUN pip install --user uv

# Install PyTorch first (CUDA 12.1 wheel)
RUN uv pip install torch==2.1.2 torchaudio==2.1.2 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install WhisperX and runtime deps (must be separate due to ctranslate2 constraints)
RUN uv pip install setuptools && \
    uv pip install --no-deps --no-build-isolation \
      "whisperx @ git+https://github.com/m-bain/whisperX.git@v3.1.1" && \
    uv pip install "ctranslate2>=4.0" "faster-whisper>=1.0.0" \
      "transformers>=4.35.0,<4.42.0" nltk

# Copy and install the package
COPY --chown=refinery:refinery . .
RUN uv pip install -e .

CMD ["audio-refinery", "--help"]
```

### docker-compose.yml

```yaml
services:
  refinery:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']       # Pin to specific GPU by PCI ID
              capabilities: [gpu]
    volumes:
      - /data/audio:/data             # Persistent audio storage
      - /mnt/fast_scratch:/mnt/fast_scratch  # RAM disk (mount on host first)
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL}
    command: >
      audio-refinery pipeline
        --base-dir /data/batch
        --compute-type int8_float16
```

### Running in the cloud

On a cloud GPU instance without a RAM disk, substitute a high-bandwidth NVMe instance volume
for `/mnt/fast_scratch`. Cloud GPU instances typically provide NVMe-backed instance storage at
2–4 GB/s write throughput — adequate as a scratch substitute at the cost of some SSD wear.

```bash
# Use a local NVMe volume as scratch instead of RAM disk
audio-refinery pipeline \
  --base-dir /data/batch \
  --compute-type int8_float16
  # The pipeline will prompt for confirmation before writing to local storage
  # if /mnt/fast_scratch is not mounted
```

---

## Monitoring

### GPU Temperature

The pipeline monitors GPU temperature automatically during batch runs. The progress bar
description shows the current temperature, color-coded relative to `--temp-limit` (default 80°C):

| Color  | Condition                             |
|--------|---------------------------------------|
| Green  | More than 10°C below limit (safe)     |
| Yellow | Within 10°C of limit (watch it)       |
| Red    | At or above limit (shutdown imminent) |

If the temperature exceeds the limit, the pipeline shuts down gracefully. Completed files are
preserved; the run can be resumed after the GPU cools.

```bash
# Tighten the thermal limit for a warm environment
audio-refinery pipeline --base-dir /data/batch --temp-limit 75

# Disable thermal monitoring entirely
audio-refinery pipeline --base-dir /data/batch --temp-limit 0
```

### Continuous nvidia-smi Logging

For sustained multi-day runs, log GPU utilization and temperature to a file:

```bash
# Log temperature + utilization every 10 seconds
nvidia-smi dmon -s tu -d 10 | tee /tmp/gpu_monitor.log &
MONITOR_PID=$!

audio-refinery pipeline --base-dir /data/batch

kill $MONITOR_PID
```

### Slack Notifications

Set `SLACK_WEBHOOK_URL` in `.env` to receive notifications when a pipeline run completes or
shuts down due to a thermal event:

```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
```

Notifications are fire-and-forget. A delivery failure never blocks or aborts the pipeline.

---

## VRAM Lifecycle Management

Models loaded into GPU VRAM can leave fragmented allocations between stages. For long-running
deployments (async workers, multi-day batch jobs), explicit VRAM cleanup between pipeline runs
prevents gradual memory pressure from accumulating.

If integrating audio-refinery stages directly into a Python process rather than through the CLI:

```python
import gc
import torch

# After each file is processed:
gc.collect()
torch.cuda.empty_cache()

# After unloading a model:
del model
gc.collect()
torch.cuda.empty_cache()
```

The CLI pipeline handles this automatically between files. It is only relevant when embedding
stage functions directly in a long-lived process.

### VRAM Footprint by Stage

These figures are approximate on a default `large-v3` configuration:

| Stage                            | Model             | Peak VRAM |
|----------------------------------|-------------------|:---------:|
| Vocal separation                 | Demucs htdemucs   |   ~4 GB   |
| Speaker diarization              | Pyannote 3.1      |   ~1 GB   |
| Transcription                    | WhisperX large-v3 |  ~10 GB   |
| All models loaded simultaneously | —                 |  ~15 GB   |

A 24 GB GPU comfortably holds all three models resident simultaneously. On a 10–12 GB GPU,
models must be loaded and unloaded between stages, adding ~10–30 seconds of overhead per file.
