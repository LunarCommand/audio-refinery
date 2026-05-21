# Service Guide

Audio Refinery's service mode is a long-lived HTTP service that wraps the same
core pipeline as the [CLI](cli.md). It accepts batches of jobs over a
bearer-authenticated REST API, fetches and uploads audio by URI, processes jobs
on a background worker, and writes one combined transcript per job plus one
summary per batch. Models load once at container startup and stay resident.

For one-off, interactive processing on a workstation, use the [CLI](cli.md)
instead.

## Contents

- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Endpoint reference](#endpoint-reference)
- [Output schemas](#output-schemas)
- [Environment variables](#environment-variables)
- [URI schemes](#uri-schemes)
- [Operations](#operations)
- [Local development loop](#local-development-loop)
- [Troubleshooting](#troubleshooting)

---

## Quickstart

```bash
docker run --gpus all -p 8000:8000 \
  -e REFINERY_API_KEYS=your-secret-key \
  -e HF_TOKEN=hf_your_token \
  lunarcommand/audio-refinery:latest
```

Wait for the service to finish warming up (models load at startup, ~10–30 s),
then submit a batch:

```bash
# Wait for readiness
until curl -sf http://localhost:8000/health > /dev/null; do sleep 2; done

# Submit a two-job batch with presigned URLs
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "summary_uri": "https://bucket.s3.amazonaws.com/batches/run-42/summary.json?X-Amz-...",
    "jobs": [
      {"input_uri": "https://bucket.s3.amazonaws.com/in/a.wav?X-Amz-...",
       "output_uri": "https://bucket.s3.amazonaws.com/out/a.json?X-Amz-..."}
    ]
  }'
# → 202 {"batch_id": "btc_...", "job_ids": ["rfj_..."]}
```

Poll a job until it reaches a terminal state:

```bash
curl -s -H "Authorization: Bearer your-secret-key" \
  http://localhost:8000/jobs/rfj_...
```

---

## How it works

1. `POST /transcribe` validates the request, registers the batch and its jobs,
   enqueues them, and returns `202` immediately with generated IDs.
2. A single background worker pulls jobs FIFO and runs them **serially** (the
   GPU is the bottleneck — one job at a time per container).
3. For each job the worker: downloads `input_uri` to a per-job scratch dir, runs
   the pipeline (separate → diarize → transcribe → optional sentiment), and
   uploads a [combined transcript](#combined-transcript) to `output_uri`.
4. After every job in a batch reaches a terminal state, the worker uploads one
   [batch summary](#batch-summary) to the batch's `summary_uri`.
5. Terminal jobs and batches are retained in memory for
   `REFINERY_JOB_RETENTION_SECONDS` (default 1 h), then swept.

Jobs live in memory only — a container restart loses in-flight and completed
job state. Scale by running more containers behind a queue, not by threading
inside one container.

---

## Endpoint reference

### `GET /health`

Liveness + readiness probe. **No authentication.**

| State | HTTP | Body |
|-------|------|------|
| Models still warming up | `503` | `{"status": "loading", "stage": "...", "detail": null}` |
| Ready to serve | `200` | `{"status": "ok", "stage": null, "detail": null}` |
| Warmup failed | `503` | `{"status": "failed", "stage": "...", "detail": "..."}` |

```bash
curl -i http://localhost:8000/health
```

### `POST /transcribe`

Submit a batch of jobs. **Requires `Authorization: Bearer <key>`.**

Request body:

```json
{
  "summary_uri": "https://.../summary.json",
  "jobs": [
    {"input_uri": "https://.../in.wav", "output_uri": "https://.../out.json"}
  ]
}
```

- `summary_uri` (required) — where the batch summary is uploaded once all jobs settle.
- `jobs` (required, ≥1) — each with `input_uri` (audio to fetch) and `output_uri` (combined transcript destination).

Responses:

| HTTP | Meaning | Body |
|------|---------|------|
| `202` | Accepted | `{"batch_id": "btc_...", "job_ids": ["rfj_...", ...]}` |
| `401` | Missing/invalid bearer token | — |
| `422` | Unsupported URI scheme (not `https://` or `file://`) | validation error |
| `400` | Batch exceeds `REFINERY_MAX_BATCH_SIZE` | `{"error": "batch_too_large", "max": N, "submitted": M}` |
| `429` | Queue would exceed `REFINERY_MAX_QUEUE_SIZE` | `{"error": "queue_full"}` |
| `503` | Service still warming up | `{"error": "service_not_ready", ...}` + `Retry-After: 5` |

The `202` response returns immediately; processing happens asynchronously.

### `GET /jobs/{job_id}`

Poll a single job. **Requires `Authorization: Bearer <key>`.**

```json
{
  "job_id": "rfj_...",
  "batch_id": "btc_...",
  "status": "completed",
  "input_uri": "https://.../in.wav",
  "output_uri": "https://.../out.json",
  "started_at": "2026-05-21T00:37:24.699Z",
  "completed_at": "2026-05-21T00:37:45.398Z",
  "failed_at": null,
  "stage": null,
  "error": null,
  "retryable": null,
  "duration_seconds": 20.7
}
```

`status` is one of `queued`, `processing`, `completed`, `failed`. On failure,
`failed_at`, `stage`, `error`, and `retryable` are populated instead of
`completed_at`/`duration_seconds`. Returns `404` `{"error": "job_not_found"}` for
an unknown or already-swept job ID.

---

## Output schemas

Both documents carry a `schema_version` (semver). Consumers **must** key off it —
the transcript shape will change when v0.3.0 alignment lands. Both start at
`1.0.0` in this release.

### Combined transcript

One document per **successful** job, uploaded to that job's `output_uri`. It
envelopes the per-stage pipeline outputs so a consumer reads a single document
instead of stitching diarization + transcription + sentiment together.

```jsonc
{
  "schema_version": "1.0.0",
  "audio_refinery_version": "0.2.0",
  "processed_at": "2026-05-21T00:37:45.398Z",
  "audio": { /* AudioFileInfo: sample_rate, channels, duration_seconds, ... */ },
  "diarization": { /* full DiarizationResult */ },
  "transcription": { /* full TranscriptionResult with word-level timestamps */ },
  "sentiment": null,            // SentimentResult, present only when enabled
  "model_versions": {           // per-stage model identifiers
    "diarization": "pyannote/speaker-diarization-3.1",
    "transcription": "large-v3"
  }
}
```

### Batch summary

One document per **accepted batch**, uploaded to the batch's `summary_uri` after
every job settles. This is the canonical surface for an orchestrator to discover
batch outcomes.

```jsonc
{
  "schema_version": "1.0.0",
  "batch_id": "btc_...",
  "submitted_at": "2026-05-21T00:37:24.699Z",
  "completed_at": "2026-05-21T00:38:01.094Z",
  "jobs": [
    {
      "job_id": "rfj_...",
      "input_uri": "https://.../a.wav",
      "output_uri": "https://.../a.json",
      "status": "completed",
      "started_at": "2026-05-21T00:37:24.699Z",
      "completed_at": "2026-05-21T00:37:45.398Z",
      "duration_seconds": 20.7
    },
    {
      "job_id": "rfj_...",
      "input_uri": "https://.../b.wav",
      "output_uri": "https://.../b.json",
      "status": "failed",
      "started_at": "2026-05-21T00:37:45.398Z",
      "failed_at": "2026-05-21T00:37:46.001Z",
      "stage": "download",
      "error": "404 fetching input_uri",
      "retryable": true
    }
  ],
  "totals": {"submitted": 2, "completed": 1, "failed": 1}
}
```

`stage` is one of `download`, `transcribe`, `upload`, `thermal_shutdown`.

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `REFINERY_API_KEYS` | **required** | Comma-separated allowlist of accepted bearer tokens. The service exits at startup if unset. |
| `HF_TOKEN` | **required for diarization** | HuggingFace token for the gated Pyannote models. |
| `REFINERY_DEVICE` | `cuda` | Compute device: `cpu`, `cuda`, or `cuda:N`. |
| `REFINERY_WHISPER_MODEL` | `large-v3` | WhisperX model. Variants: `distil-large-v3`, `medium.en`, `medium`. Fixed for the container's lifetime. |
| `REFINERY_COMPUTE_TYPE` | `float16` | CTranslate2 compute type: `float16`, `int8_float16`, `int8`, `float32`. |
| `REFINERY_DEFAULT_LANGUAGE` | `en` | Language code or `auto`. |
| `REFINERY_SENTIMENT_ENABLED` | `false` | Set `true` to run step 4 (sentiment) on every job. |
| `REFINERY_SCRATCH_DIR` | unset (`/tmp`) | Base dir for per-job scratch. Mount a tmpfs here for the RAM-disk throughput benefit. |
| `REFINERY_INTERMEDIATE_DIR` | unset | Debugging: copy per-stage JSONs to `<dir>/<job_id>/` after each success. Off by default. |
| `REFINERY_MAX_BATCH_SIZE` | `25` | Server-side cap on jobs per `POST /transcribe`. Over-limit → `400`. |
| `REFINERY_MAX_QUEUE_SIZE` | `100` | Job queue cap. Over-limit POSTs → `429`. |
| `REFINERY_JOB_RETENTION_SECONDS` | `3600` | How long terminal jobs/batches stay queryable before being swept. |
| `REFINERY_GPU_TEMP_LIMIT` | `0` (off) | °C threshold for the thermal guard. Any positive value spawns the guard daemon. |
| `REFINERY_GPU_TEMP_POLL_SECONDS` | `5.0` | How often the thermal guard polls `nvidia-smi`. |
| `REFINERY_PORT` | `8000` | Port the service binds (all interfaces). |
| `REFINERY_LOG_FORMAT` | `json` | `json` for structured logs (recommended in production) or `console` for human-readable. |
| `SLACK_WEBHOOK_URL` | unset | Optional. Fire-and-forget failure notifications. |

One container = one model configuration. To serve a different Whisper variant or
device, run another container.

---

## URI schemes

Both `input_uri`/`output_uri`/`summary_uri` accept:

- **`https://`** — for input, a GET-able URL (e.g. an S3 presigned GET). For
  output/summary, a PUT-able URL (e.g. an S3 presigned PUT). This is the
  production path; the service holds no cloud credentials of its own.
- **`file://`** — a path on a filesystem visible to the container. Intended for
  local development with bind mounts (see below) and for shared-volume
  deployments.

Any other scheme is rejected with `422` at request validation.

---

## Operations

### Readiness probes

`/health` returns `503` until model warmup finishes, then `200`. Wire it to your
orchestrator's readiness probe so traffic only arrives once the container can
serve it.

Kubernetes:

```yaml
readinessProbe:
  httpGet: { path: /health, port: 8000 }
  initialDelaySeconds: 15
  periodSeconds: 10
  failureThreshold: 6
livenessProbe:
  httpGet: { path: /health, port: 8000 }
  periodSeconds: 30
```

The container `HEALTHCHECK` already probes `/health` with a 60 s start period.

### Scaling

One worker processes jobs serially per container. To increase throughput, run
more containers (one per GPU) behind your own queue/dispatcher. There is no
in-container parallelism to tune — the `pipeline-parallel` multi-GPU mode is a
CLI-only feature.

### Scratch and the tmpfs hint

Per-job scratch (input download + Demucs stems) defaults to `/tmp` inside the
container. On a RAM-rich host, mount a tmpfs and point `REFINERY_SCRATCH_DIR` at
it for a throughput boost:

```bash
docker run --gpus all -p 8000:8000 \
  --tmpfs /scratch \
  -e REFINERY_SCRATCH_DIR=/scratch \
  -e REFINERY_API_KEYS=your-secret-key -e HF_TOKEN=hf_... \
  lunarcommand/audio-refinery:latest
```

If `REFINERY_SCRATCH_DIR` points at a non-tmpfs path, the service logs a
`scratch.not_tmpfs` warning at startup — informational, not an error. Peak
transient scratch is roughly one job's worth (~3–5 GB).

### Model cache

Pyannote and WhisperX weights download on first use to
`~/.cache` inside the container (the `refinery` user's home). To avoid
re-downloading on every container start, mount a persistent volume at
`/home/refinery/.cache`.

### Logs

Logs are structured JSON by default (`REFINERY_LOG_FORMAT=json`), one object per
line — ready for ingestion by a log aggregator. Each request and job-lifecycle
event carries a `caller_fp` (SHA-256[:8] of the bearer token) for attribution
without logging the secret. Set `REFINERY_LOG_FORMAT=console` for readable local
output.

---

## Local development loop

Use `file://` URIs with bind mounts to exercise the full service without any
cloud setup. The quickest path is the Makefile target, which binds
`.docker-inbox` / `.docker-outbox` / `.docker-summaries` in the repo root:

```bash
REFINERY_API_KEYS=test-key HF_TOKEN=$HF_TOKEN make run-service-local
```

Or run it explicitly with your own mount points:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e REFINERY_API_KEYS=test-key \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$PWD/inputs:/inputs:ro" \
  -v "$PWD/outputs:/outputs" \
  -v "$PWD/summaries:/summaries" \
  lunarcommand/audio-refinery:latest
```

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "summary_uri": "file:///summaries/batch.json",
    "jobs": [
      {"input_uri": "file:///inputs/sample.wav",
       "output_uri": "file:///outputs/sample.json"}
    ]
  }'
```

The combined transcript lands in `./outputs/sample.json` and the batch summary
in `./summaries/batch.json` on the host. Run the container's uid 1000 against
host-owned dirs (or `chmod` the output dirs) so the `refinery` user can write.

You can also run the service without Docker for fast iteration:

```bash
REFINERY_API_KEYS=test-key audio-refinery serve
# equivalently: audio-refinery-service
```

---

## Troubleshooting

- **Container exits immediately at startup** — `REFINERY_API_KEYS` is unset. The service refuses to start without an allowlist (a service no caller can reach is worse than a crash loop). Set it.
- **`/health` stuck at `503 loading`** — Warmup is still running, or it failed. Check the logs for `service.ready` (success) or a warmup error. Cold model downloads on first run can take a while; mount a cache volume.
- **All jobs fail at `stage: download`** — The `input_uri` isn't reachable from the container, or a presigned URL expired. For `file://`, confirm the bind mount and path.
- **All jobs fail at `stage: upload`** — The `output_uri`/`summary_uri` isn't PUT-able (wrong presign method, expired URL), or for `file://` the `refinery` user can't write the mounted dir.
- **`401` on every request** — Missing `Authorization: Bearer <key>` header, or the key isn't in `REFINERY_API_KEYS`.
- **Diarization fails for every job** — `HF_TOKEN` missing/invalid, or the gated model licenses weren't accepted. See [the README prerequisites](../README.md#prerequisites).
- **`scratch.not_tmpfs` warning** — Informational. Mount a tmpfs at the scratch path for better batch throughput, or ignore it on RAM-tight hosts.
