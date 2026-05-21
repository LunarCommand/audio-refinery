# Deployment

Audio Refinery deploys in two shapes:

- **CLI deployment** — sustained, unattended batch runs over a directory of WAV
  files on a GPU workstation or a single cloud instance. Best when you control
  the input directory and want to process a corpus.
- **Service deployment** — the containerized HTTP service for programmatic,
  asynchronous job submission (URI-in / URI-out). Best when jobs arrive from
  other systems or you need to scale behind an orchestrator.

Both run the same core pipeline. This guide covers operating each in production;
for the service's API surface and environment-variable reference, see the
[Service Guide](service.md).

---

## CLI deployment (workstation batch)

For sustained batch processing on a local GPU workstation, a few practices make
long-running jobs reliable and recoverable.

### Environment and startup

```bash
# Activate the virtualenv
source /path/to/audio-refinery/.venv/bin/activate

# Verify GPU is accessible and not occupied
audio-refinery pipeline --base-dir /data/audio/batch
```

The pipeline runs a GPU pre-flight check before loading any models. If active
processes are found on the target GPU, it prints the PID and VRAM footprint of
each and asks for confirmation before proceeding. This prevents inadvertently
loading models onto a GPU already under load.

### Scratch directory (RAM disk)

The Demucs scratch directory resolves to `$REFINERY_SCRATCH_DIR/demucs` when set,
or `tempfile.gettempdir()/audio-refinery-demucs` (typically `/tmp`) otherwise.
For heavy batch runs, point it at a tmpfs RAM disk to avoid SSD write
amplification:

```bash
sudo mkdir -p /mnt/fast_scratch
sudo mount -t tmpfs -o size=32G,mode=1777 tmpfs /mnt/fast_scratch
export REFINERY_SCRATCH_DIR=/mnt/fast_scratch
```

To persist across reboots, add to `/etc/fstab`:

```
tmpfs /mnt/fast_scratch tmpfs defaults,size=32G,mode=1777 0 0
```

Size the RAM disk to at least 2× the largest expected file's stem output (~400 MB
for a typical 5–8 minute file). 8–32 GB provides comfortable headroom for longer
recordings.

See [architecture.md](architecture.md) for the rationale behind the RAM disk
strategy, and [cli.md](cli.md#scratch-directory) for per-invocation overrides.

### Running a batch

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

### Resume and recovery

The pipeline skips files whose output already exists and is non-empty. An
interrupted run — due to power loss, thermal shutdown, or manual termination —
can be restarted with the same command. Completed files are not reprocessed.

```bash
# Resume an interrupted run (default behavior — no flags needed)
audio-refinery pipeline --base-dir /data/audio/batch

# Force full reprocessing
audio-refinery pipeline --base-dir /data/audio/batch --no-resume
```

---

## Service deployment (HTTP API)

The HTTP service is the supported path for programmatic, asynchronous job
submission. It provides a job queue, a background worker, readiness probes, and
URI-driven I/O out of the box — so you no longer need to build your own
database-as-queue worker around the CLI. See the [Service Guide](service.md) for
the full endpoint reference, output schemas, and environment-variable table.

### Container prerequisites

#### HuggingFace token

Pyannote speaker diarization uses gated models that require HuggingFace
authentication:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the model terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Generate a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Pass the token via `-e HF_TOKEN` or the compose `environment:` block. Without it
the diarization stage fails with a 401 the first time it tries to download the
model.

#### NVIDIA driver version

The `nvidia/cuda:12.1.1` base image requires **NVIDIA driver ≥ 525.85.12** on the
host. Check before pulling:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

If the driver is older than 525, update it before proceeding.

#### NVIDIA Container Toolkit

Docker cannot access the GPU without the NVIDIA Container Toolkit installed and
configured on the host. This applies to both local workstations and cloud
instances — most cloud GPU images include NVIDIA drivers but not the container
toolkit.

```bash
# Install the toolkit (Ubuntu / Debian)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access inside a container before building:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

This should print the same `nvidia-smi` output as the host. If the toolkit was
configured via CDI and the spec is stale after a driver upgrade, regenerate it
with `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`.

### Building the image

The repository ships a multi-stage [`Dockerfile`](../Dockerfile) (CUDA runtime
base, non-root `refinery` user, all WhisperX pins baked in). Build it with the
Makefile target:

```bash
make build-image          # builds lunarcommand/audio-refinery:latest
```

The build clones WhisperX from a pinned commit and downloads PyTorch CUDA
wheels, so it requires internet access and takes 10–20 minutes on first run.
Subsequent builds reuse layer caching. Released images are published to Docker
Hub as `lunarcommand/audio-refinery:<version>`.

### Running the service container

```bash
docker run --gpus all -p 8000:8000 \
  -e REFINERY_API_KEYS=your-secret-key \
  -e HF_TOKEN="${HF_TOKEN}" \
  --mount type=tmpfs,dst=/scratch,tmpfs-mode=1777 -e REFINERY_SCRATCH_DIR=/scratch \
  lunarcommand/audio-refinery:latest
```

The default `CMD` is `audio-refinery-service`. `/health` returns `503` until
warmup finishes, then `200`. See the [Service Guide quickstart](service.md#quickstart)
for submitting jobs and the [environment-variable table](service.md#environment-variables)
for all `REFINERY_*` knobs.

### Orchestrators (Kubernetes / ECS)

Wire `/health` to a readiness probe so traffic only arrives once the container
can serve it. Run one container per GPU and scale horizontally — there is no
in-container parallelism to tune. Mount a persistent volume at
`/home/refinery/.cache` to avoid re-downloading model weights on every start.
Probe config snippets are in the [Service Guide operations section](service.md#operations).

### Cloud registry workflow

Build and push from a local machine or CI runner, then pull on the cloud
instance:

```bash
# Build and push
docker build -t your-registry/audio-refinery:latest .
docker push your-registry/audio-refinery:latest

# Pull and run (cloud instance — after completing prerequisites above)
docker pull your-registry/audio-refinery:latest
docker run --gpus all -p 8000:8000 \
  -e REFINERY_API_KEYS=your-secret-key \
  -e HF_TOKEN="${HF_TOKEN}" \
  your-registry/audio-refinery:latest
```

The recommended minimum is a **24 GB GPU** to hold all models resident
simultaneously (see [VRAM footprint by stage](#vram-footprint-by-stage)). Common
instance types: NVIDIA A10G (AWS g5), L4 (GCP g2), RTX 3090 / 4090 (bare metal).

### Running the CLI in a container

The same image can run a one-shot CLI batch instead of the service by overriding
the command — useful for containerized batch jobs over a bind-mounted directory:

```bash
docker run --rm --gpus '"device=0"' \
  -v /data/audio:/data \
  --mount type=tmpfs,dst=/scratch,tmpfs-mode=1777 -e REFINERY_SCRATCH_DIR=/scratch \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL}" \
  lunarcommand/audio-refinery:latest \
  audio-refinery pipeline --base-dir /data/batch --compute-type int8_float16
```

`--gpus '"device=0"'` pins to a specific GPU by index; use `--gpus all` for all GPUs.

---

## Monitoring

### GPU temperature

The pipeline monitors GPU temperature automatically during batch runs. The
progress bar description shows the current temperature, color-coded relative to
`--temp-limit` (default 80°C):

| Color  | Condition                             |
|--------|---------------------------------------|
| Green  | More than 10°C below limit (safe)     |
| Yellow | Within 10°C of limit (watch it)       |
| Red    | At or above limit (shutdown imminent) |

If the temperature exceeds the limit, the pipeline shuts down gracefully.
Completed files are preserved; the run can be resumed after the GPU cools. In
service mode the same guard is opt-in via `REFINERY_GPU_TEMP_LIMIT` (see the
[Service Guide](service.md#environment-variables)).

```bash
# Tighten the thermal limit for a warm environment
audio-refinery pipeline --base-dir /data/batch --temp-limit 75

# Disable thermal monitoring entirely
audio-refinery pipeline --base-dir /data/batch --temp-limit 0
```

### Continuous nvidia-smi logging

For sustained multi-day runs, log GPU utilization and temperature to a file:

```bash
# Log temperature + utilization every 10 seconds
nvidia-smi dmon -s tu -d 10 | tee /tmp/gpu_monitor.log &
MONITOR_PID=$!

audio-refinery pipeline --base-dir /data/batch

kill $MONITOR_PID
```

### Slack notifications

Set `SLACK_WEBHOOK_URL` (in `.env` for the CLI, or `-e` for the container) to
receive notifications when a run completes or shuts down due to a thermal event:

```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
```

Notifications are fire-and-forget. A delivery failure never blocks or aborts
processing.

---

## VRAM lifecycle management

Models loaded into GPU VRAM can leave fragmented allocations between stages. The
CLI pipeline and the service worker both handle cleanup between files
automatically. Explicit cleanup is only relevant if you embed stage functions
directly in a long-lived Python process of your own:

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

### VRAM footprint by stage

These figures are approximate on a default `large-v3` configuration:

| Stage                            | Model             | Peak VRAM |
|----------------------------------|-------------------|:---------:|
| Vocal separation                 | Demucs htdemucs   |   ~4 GB   |
| Speaker diarization              | Pyannote 3.1      |   ~1 GB   |
| Transcription                    | WhisperX large-v3 |  ~10 GB   |
| All models loaded simultaneously | —                 |  ~15 GB   |

A 24 GB GPU comfortably holds all three models resident simultaneously. On a
10–12 GB GPU, models must be loaded and unloaded between stages, adding ~10–30
seconds of overhead per file.
