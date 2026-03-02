# Audio Refinery

GPU-accelerated audio processing pipeline: vocal separation (Demucs), speaker diarization (Pyannote), transcription (WhisperX), and text sentiment analysis. Its primary use case is building AI-ready audio databases — transforming raw recordings into structured, speaker-attributed JSON with word-level timestamps that feed directly into RAG pipelines, vector stores, and fine-tuning datasets. The pipeline uses a Ghost Track strategy: AI models run against a clean, music-free vocal stem to maximize accuracy, then the resulting metadata is applied back to the original audio, preserving its acoustic character. Designed to run on 24 GB consumer GPUs with all models resident in VRAM simultaneously, it processes large corpora in batch with no model reload overhead between files.

## Installation

```bash
# Create and activate a Python 3.11 virtualenv
uv venv --python 3.11.14
source .venv/bin/activate

# Install PyTorch first (CUDA 12.1 wheel — adjust for your CUDA version)
uv pip install torch==2.1.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Install whisperx in isolation to avoid ctranslate2/torch version conflicts
uv pip install "setuptools<74"
uv pip install --no-deps --no-build-isolation "whisperx @ git+https://github.com/m-bain/whisperX.git@v3.1.1"
# whisperx runtime deps (transformers must be <4.42 — newer versions require torch>=2.4)
uv pip install "ctranslate2>=4.0" "faster-whisper>=1.0.0" "transformers>=4.35.0,<4.42.0" nltk

# Install the package and remaining deps
uv pip install -e .
uv pip install pytest pytest-mock  # for development
```

> **NumPy constraint:** `numpy<2.0.0` is pinned in `pyproject.toml`. Do not upgrade it — WhisperX and some audio libraries break with NumPy 2.x.

---

## Prerequisites

### HuggingFace access token (required for diarization)

Pyannote speaker diarization models are gated on HuggingFace. Complete these steps once before running `audio-refinery diarize`:

1. Create a HuggingFace account at [huggingface.co](https://huggingface.co) if you don't have one.
2. Accept the license for each gated model (must be logged in):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Create a read-only access token: Profile → Settings → Access Tokens → New token.
4. Add it to your `.env` file (copy `.env.example` to `.env`):
   ```
   HF_TOKEN=hf_your_token_here
   ```
   Or export it in your shell: `export HF_TOKEN=hf_your_token_here`

The `.env` file is gitignored. The token is never embedded in code.

### RAM disk (strongly recommended)

The default Demucs output directory is `/mnt/fast_scratch/demucs_output`. Using a tmpfs RAM disk avoids SSD write amplification during heavy audio processing and significantly reduces per-file latency. If the RAM disk is not available, the pipeline will ask for confirmation before falling back to local storage.

```bash
sudo mkdir -p /mnt/fast_scratch
sudo mount -t tmpfs -o size=32G,mode=1777 tmpfs /mnt/fast_scratch
```

The `mode=1777` flag is required so non-root users can write to the mount.

To persist across reboots, add to `/etc/fstab`:

```
tmpfs /mnt/fast_scratch tmpfs defaults,size=32G,mode=1777 0 0
```

Or append it automatically:

```bash
echo "tmpfs /mnt/fast_scratch tmpfs defaults,size=32G,mode=1777 0 0" | sudo tee -a /etc/fstab
```

Check usage:

```bash
df -h /mnt/fast_scratch
```

If you prefer not to use a RAM disk, pass `-o /some/other/dir` to write output anywhere.

### Demucs model weights

The first run will auto-download the `htdemucs` model weights (~80 MB) to `~/.cache/torch/hub/checkpoints/`. No HuggingFace token or manual download is required.

---

## Step 1: Separate Vocals

```bash
audio-refinery separate /path/to/audio.wav
```

This runs Demucs `htdemucs` with `--two-stems=vocals` on the GPU. Output:
- A blue panel summarizing input/output/device
- An animated spinner while Demucs processes
- A green results table with file info, timing, and output paths
- A JSON dump of the full `SeparationResult` provenance record

Demucs writes two stems:
```
/mnt/fast_scratch/demucs_output/htdemucs/<track_name>/vocals.wav
/mnt/fast_scratch/demucs_output/htdemucs/<track_name>/no_vocals.wav
```

### CLI options

```
Usage: audio-refinery separate [OPTIONS] INPUT_FILE

Options:
  -o, --output-dir PATH        Output directory (default: /mnt/fast_scratch/demucs_output)
  -d, --device TEXT            Compute device: 'cpu', 'cuda', or 'cuda:N' (default: cuda)
  --segment INTEGER            Segment size in seconds for VRAM optimization
  --help                       Show help and exit
```

### Common variations

```bash
# Pin to a specific GPU
audio-refinery separate /path/to/audio.wav -d cuda:0

# Use CPU instead of GPU (slower)
audio-refinery separate /path/to/audio.wav -d cpu

# Write output to a custom directory
audio-refinery separate /path/to/audio.wav -o /tmp/my_output

# Reduce VRAM usage for long files (splits into 40-second chunks)
audio-refinery separate /path/to/audio.wav --segment 40
```

### Troubleshooting

- **`CUDA out of memory`** — Try `--segment 40` (or lower) to reduce peak VRAM usage.
- **`audio-refinery: command not found`** — Run `uv pip install -e .` with the venv activated.
- **`Demucs is not installed or not on PATH`** — Make sure the venv is activated.

---

## Step 2: Diarize

After step 1 produces `vocals.wav`, feed it to `audio-refinery diarize` to identify speaker turns.

```bash
audio-refinery diarize /mnt/fast_scratch/demucs_output/htdemucs/audio_track/vocals.wav
```

This runs Pyannote `speaker-diarization-3.1`. Output:
- A blue panel summarizing input and device
- An animated spinner while Pyannote processes
- A green summary table: speakers detected, total segments, total speech duration
- A per-speaker breakdown table
- A JSON dump of the full `DiarizationResult` provenance record

### CLI options

```
Usage: audio-refinery diarize [OPTIONS] INPUT_FILE

Options:
  -d, --device TEXT            Compute device: 'cpu', 'cuda', or 'cuda:N' (default: cuda)
  --min-speakers INTEGER       Minimum number of speakers (optional hint)
  --max-speakers INTEGER       Maximum number of speakers (optional hint)
  --hf-token TEXT              HuggingFace token (overrides HF_TOKEN env var)
  -o, --output-file PATH       Write DiarizationResult JSON to this file
  --help                       Show help and exit
```

### Common variations

```bash
# Hint that the audio has 2–4 speakers
audio-refinery diarize /path/to/vocals.wav -d cuda:0 --min-speakers 2 --max-speakers 4

# Save result JSON for downstream processing
audio-refinery diarize /path/to/vocals.wav -d cuda:0 -o /tmp/diarization.json
```

### Troubleshooting

- **`HuggingFace token not found`** — Follow the HF setup steps above.
- **`Failed to load Pyannote pipeline`** — Check that you accepted both model licenses on HuggingFace.

---

## Step 3: Transcribe

After step 2 produces a `DiarizationResult` JSON, feed the same `vocals.wav` to `audio-refinery transcribe` to generate word-level timestamps. Optionally pass the diarization JSON to get speaker labels on each segment.

```bash
# Pure transcription (no speaker labels)
audio-refinery transcribe /path/to/vocals.wav -d cuda:0 -o /tmp/transcription.json

# With speaker assignment from step 2
audio-refinery transcribe /path/to/vocals.wav \
  -d cuda:0 \
  --diarization-file /tmp/diarization.json \
  -o /tmp/transcription_with_speakers.json
```

This runs WhisperX (`large-v3`) with Wav2Vec2 forced alignment. Output:
- A blue panel summarizing input, device, compute type, and language
- An animated spinner while processing
- A green summary table: language, segments, words, processing time
- A transcript preview table (first 10 segments, with optional speaker column)
- A JSON dump of the full `TranscriptionResult` provenance record

### CLI options

```
Usage: audio-refinery transcribe [OPTIONS] INPUT_FILE

Options:
  -d, --device TEXT                Compute device: 'cpu', 'cuda', or 'cuda:N' (default: cuda)
  --compute-type [float16|int8_float16|int8|float32]
                                   CTranslate2 compute type (default: float16).
                                   int8_float16 gives ~1.5× throughput with negligible accuracy delta.
  --batch-size INTEGER             Batch size for transcription (default: 16)
  --language TEXT                  Language code, e.g. 'en', 'fr', or 'auto' (default: en)
  --diarization-file PATH          DiarizationResult JSON from step 2 for speaker assignment
  -o, --output-file PATH           Write TranscriptionResult JSON to this file
  --help                           Show help and exit
```

### Common variations

```bash
# Use int8_float16 for ~1.5× throughput (recommended)
audio-refinery transcribe /path/to/vocals.wav --compute-type int8_float16

# Auto-detect language
audio-refinery transcribe /path/to/vocals.wav --language auto

# Full pipeline output with speaker labels
audio-refinery transcribe /path/to/vocals.wav \
  -d cuda:0 \
  --diarization-file /tmp/diarization.json \
  -o /tmp/transcript.json
```

### Troubleshooting

- **`whisperx is not installed`** — Follow the install steps above. whisperx requires manual installation due to ctranslate2 version constraints.
- **`No such file or directory: 'ffmpeg'`** — Install the ffmpeg binary: `sudo apt install ffmpeg`
- **`CUDA out of memory`** — Try `--compute-type int8` to reduce VRAM usage, or lower `--batch-size`.
- **`alignment failed` warning** — No Wav2Vec2 alignment model for the detected language. Transcription falls back to Whisper's raw timestamps; segment text is still correct.

---

## Step 4: Text Sentiment Analysis

After step 3 produces a `TranscriptionResult` JSON, feed it to `audio-refinery sentiment` to score each segment as positive, neutral, or negative. **This step is text-only — no audio or GPU required.**

The analyzer uses [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) and writes results in two places:

1. A standalone `SentimentResult` JSON (optional, via `-o`)
2. In-place update of the source transcription JSON — each segment gains a `sentiment` field and `sentiment_applied` is set to `true`

```bash
# Analyse and update the transcription in place
audio-refinery sentiment /data/audio/transcription/transcription_abc123.json

# Also write a standalone SentimentResult JSON
audio-refinery sentiment /data/audio/transcription/transcription_abc123.json \
  -o /data/audio/sentiment/sentiment_abc123.json

# Use a different model
audio-refinery sentiment /path/to/transcription.json --model my-org/my-sentiment-model
```

### CLI options

```
Usage: audio-refinery sentiment [OPTIONS] TRANSCRIPTION_FILE

Arguments:
  TRANSCRIPTION_FILE     TranscriptionResult JSON written by step 3  [required]

Options:
  --model TEXT           HuggingFace model for text-classification
                         [default: cardiffnlp/twitter-roberta-base-sentiment-latest]
  -d, --device TEXT      Compute device: 'cpu', 'cuda', or 'cuda:N'  [default: cpu]
  -o, --output-file PATH Write SentimentResult JSON to this file
  --help                 Show this message and exit.
```

---

## Batch Pipeline

`audio-refinery pipeline` runs all active stages (separation → diarization → transcription, and optionally sentiment) across an entire directory of WAV files. Processing is interleaved per-file: each file completes all active stages before the next begins, so ghost-track stems never accumulate beyond one file's worth of data at a time. All models are loaded once before the loop starts.

### Directory layout

```
/mnt/fast_scratch/              ← RAM disk (preferred scratch, strongly recommended)
  demucs/                       ← Demucs stems (created and deleted per-file by default)

<base>/
  extracted/                    ← input WAV files (must exist)
    audio_<id>.wav
  diarization/                  ← intermediate Pyannote JSON (created automatically)
  transcription/                ← final WhisperX JSON output (created automatically)
  summary/                      ← run summary JSON (created automatically)
    pipeline_summary.json
```

If `<base>/extracted` does not exist the pipeline exits immediately with an error. All other subdirectories are created on demand.

**Demucs scratch (priority order):**
1. `/mnt/fast_scratch/demucs` — used automatically if `/mnt/fast_scratch` is mounted (preferred)
2. `<base>/demucs` — fallback to local storage if the RAM disk is unavailable; the pipeline asks for confirmation before proceeding

### Running the pipeline

```bash
# Minimal invocation (steps 1–3: separate, diarize, transcribe)
audio-refinery pipeline --base-dir /data/audio/batch

# Pin to a specific GPU
audio-refinery pipeline --base-dir /data/audio/batch -d cuda:0

# Reduce VRAM pressure during Demucs
audio-refinery pipeline --base-dir /data/audio/batch --segment 40

# Re-run everything from scratch
audio-refinery pipeline --base-dir /data/audio/batch --no-resume

# Keep all ghost-track stems for inspection
audio-refinery pipeline --base-dir /data/audio/batch --keep-scratch

# Enable step 4 (text sentiment analysis)
audio-refinery pipeline --base-dir /data/audio/batch --sentiment

# Use int8_float16 for ~1.5× throughput with negligible accuracy delta
audio-refinery pipeline --base-dir /data/audio/batch --compute-type int8_float16

# Tighten the thermal ceiling (default is 80°C)
audio-refinery pipeline --base-dir /data/audio/batch --temp-limit 75

# Disable temperature monitoring entirely
audio-refinery pipeline --base-dir /data/audio/batch --temp-limit 0
```

### CLI options

```
Usage: audio-refinery pipeline [OPTIONS]

Options:
  --base-dir PATH                    Base directory (extracted/ must exist inside)  [required]
  -d, --device TEXT                  Compute device: 'cpu', 'cuda', or 'cuda:N'  [default: cuda]
  --segment INTEGER                  Demucs segment size in seconds for VRAM optimisation
  --compute-type [float16|int8_float16|int8|float32]
                                     WhisperX compute type  [default: float16]
  --batch-size INTEGER               WhisperX batch size  [default: 16]
  --language TEXT                    Language code (e.g. 'en') or 'auto'  [default: en]
  --whisper-model TEXT               WhisperX model name  [default: large-v3]
                                     Variants: large-v3 (highest accuracy), distil-large-v3
                                     (~2× faster), medium.en (~3.2× faster, English-only),
                                     medium (~3× faster, multilingual)
  --hf-token TEXT                    HuggingFace token for Pyannote (overrides HF_TOKEN env var)
  --no-resume                        Re-process all files, ignoring existing outputs
  --keep-scratch                     Keep all ghost-track stems after processing
  --temp-limit INTEGER               GPU temperature limit in °C.  [default: 80]
                                     A background thread checks every 5 s and shuts the pipeline
                                     down if the limit is exceeded. Set to 0 to disable.
  --sentiment                        Enable step 4: Text Sentiment Analysis
  --help                             Show this message and exit.
```

### Scratch-disk management

By default, ghost-track stems are cleaned up per-file as soon as they are no longer needed:

- `no_vocals.wav` — deleted immediately after separation
- `vocals.wav` — deleted after transcription

This bounds scratch usage to roughly one file's worth of stems (~400 MB) at any point in the run. Use `--keep-scratch` to retain all stems.

### Resume behaviour

By default, the pipeline skips any file whose transcription output already exists and is non-empty. An interrupted run can be safely restarted. Use `--no-resume` to force full reprocessing.

### Summary file

At the end of every run, `<base>/summary/pipeline_summary.json` is written with per-stage counts (processed / skipped / failed), timing, active steps, and a list of any failures.

### Troubleshooting

- **`Source directory not found`** — Create `<base>/extracted/` and place `audio_<id>.wav` files inside.
- **`No audio_*.wav files found`** — Source files must match the pattern `audio_<id>.wav`.
- **Partial failures** — Check the failure table printed at the end of the run and `summary/pipeline_summary.json`.
- **`Thermal shutdown: cuda:N at XX°C`** — The GPU exceeded `--temp-limit`. Wait for it to cool, then resume (completed files are skipped automatically).

---

## GPU Identification and Device Targeting

CUDA and `nvidia-smi` use independent GPU numbering schemes by default, and they don't always
agree. Without corrective configuration, `cuda:0` in PyTorch may refer to a different physical
card than index `0` in `nvidia-smi`, making it impossible to reliably pin workloads to specific
GPUs.

Audio-refinery sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` at startup, before any CUDA context is
created. This forces both CUDA and `nvidia-smi` to enumerate GPUs in PCI bus order, so their
indices always match.

### Identifying your GPUs

```bash
# List all GPUs with their nvidia-smi index, name, and VRAM
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

Example output on a dual-GPU system:

```
index, name, memory.total [MiB]
0, NVIDIA GeForce RTX 3090 Ti, 24576 MiB
1, NVIDIA GeForce RTX 4090, 24576 MiB
```

With `CUDA_DEVICE_ORDER=PCI_BUS_ID` set, `cuda:0` in any audio-refinery command maps to
index 0 in this output, and `cuda:1` maps to index 1.

### Device string conventions

The `--device` flag uses PyTorch device syntax:

| String   | Meaning                                                                  |
|----------|--------------------------------------------------------------------------|
| `cuda`   | Default GPU (typically `cuda:0` in PCI bus order)                        |
| `cuda:0` | GPU at PCI bus index 0                                                   |
| `cuda:1` | GPU at PCI bus index 1                                                   |
| `cpu`    | CPU only — significantly slower; useful for testing or GPU-free machines |

### Verifying device assignment

Before a long batch run, confirm that `cuda:N` refers to the GPU you intend:

```bash
python -c "
import os; os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch
for i in range(torch.cuda.device_count()):
    print(f'cuda:{i} ->', torch.cuda.get_device_name(i))
"
```

The output should match the `nvidia-smi` index order:

```
cuda:0 -> NVIDIA GeForce RTX 3090 Ti
cuda:1 -> NVIDIA GeForce RTX 4090
```

### Automatic GPU ranking (pipeline-parallel)

`audio-refinery pipeline-parallel` uses `detect_gpu_order()` to rank all available GPUs
before assigning workers. The ranking uses FP16 TFLOPS from `src/gpu_tflops.toml` as the
primary sort key, so the fastest GPU gets the first worker slot. GPUs not in the table fall
back to a `(rounded VRAM GB, max SM clock)` heuristic — reliable within a generation but not
across them.

To check whether your GPU is in the table:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader
```

Compare the output against the entries in `src/gpu_tflops.toml`. If your GPU is listed, it
ranks above any unlisted GPU. If it is not, the heuristic applies — which may or may not
produce the ordering you want.

### Adding a GPU to the TFLOPS table

If your GPU is not in `src/gpu_tflops.toml`, add it for correct ranking in multi-GPU setups:

1. Get the exact name as `nvidia-smi` reports it:
   ```bash
   nvidia-smi --query-gpu=name --format=csv,noheader
   ```

2. Find the FP16 TFLOPS figure at [TechPowerUp GPU Specs](https://www.techpowerup.com/gpu-specs/).
   Use the **Shader Performance** row for consumer GPUs, or **FP16 (half)** for data center GPUs.
   See the comments in `src/gpu_tflops.toml` for guidance on which value to use.

3. Add an entry to `src/gpu_tflops.toml`:
   ```toml
   "NVIDIA GeForce RTX 5080" = 86.4
   ```

The GPU name must match `nvidia-smi` output exactly. The lookup code automatically handles
the presence or absence of the `NVIDIA ` prefix across driver versions.

---

## GPU Health Monitoring

### Pre-flight availability check

Before starting any model work, every GPU command queries `nvidia-smi` to check whether the targeted device is already occupied. If active processes are found, a yellow panel lists each PID and its VRAM footprint and asks for confirmation before proceeding.

### Live temperature display

During `audio-refinery pipeline` runs the current GPU temperature is shown in the progress bar description, colour-coded relative to the configured limit:

| Colour | Meaning                                        |
|--------|------------------------------------------------|
| green  | More than 10°C below the limit (safe)          |
| yellow | Within 10°C of the limit (warm — watch it)     |
| red    | At or above the limit (shutdown imminent)      |

Temperature is sampled at most once every 5 seconds to avoid hammering `nvidia-smi`.

### External monitoring

```bash
# Print temperature + GPU utilisation every 10 s, log to file
nvidia-smi dmon -s tu -d 10 | tee /tmp/gpu_temps.log
```

---

## Parallel Pipeline (Multi-GPU)

`audio-refinery pipeline-parallel` runs N simultaneous pipeline workers — one per `--device` flag — for significantly higher throughput when multiple GPUs are available.

### How it works

The launcher:
1. Discovers all WAV files in `<base>/extracted/`
2. Splits them into N interleaved partitions
3. Writes per-worker manifests to `<base>/manifests/`
4. Spawns one `audio-refinery pipeline` child process per `--device` flag, labelled W0, W1, …
5. Redirects each worker's output to `<base>/logs/worker_N.log`
6. Waits for all workers to complete, then prints a combined summary and writes `<base>/summary/combined_report.json`

### Directory layout

```
<base>/
  extracted/                    ← input WAV files (must exist)
  manifests/                    ← auto-generated per-worker file lists
    manifest_0.txt
    manifest_1.txt
  logs/                         ← per-worker output logs
    worker_0.log
    worker_1.log
  diarization/                  ← shared output (workers write non-overlapping files)
  transcription/                ← shared output (workers write non-overlapping files)
  summary/
    worker_0.json
    worker_1.json
    combined_report.json        ← always written; aggregates all workers and failures
```

### Running the parallel pipeline

```bash
# Basic dual-GPU run (default: cuda:0 and cuda:1)
audio-refinery pipeline-parallel --base-dir /data/audio/batch

# Recommended: int8_float16 for ~1.5× per-GPU speedup
audio-refinery pipeline-parallel --base-dir /data/audio/batch \
  --compute-type int8_float16

# With power limits (recommended for extended runs)
audio-refinery pipeline-parallel --base-dir /data/audio/batch \
  --compute-type int8_float16 \
  --power-limit 350

# Three-GPU run
audio-refinery pipeline-parallel --base-dir /data/audio/batch \
  --device cuda:0 --device cuda:1 --device cuda:2

# Monitor workers in separate terminals
tail -f /data/audio/batch/logs/worker_0.log
tail -f /data/audio/batch/logs/worker_1.log
```

### CLI options

```
Usage: audio-refinery pipeline-parallel [OPTIONS]

Options:
  --base-dir PATH                    Base directory (extracted/ must exist inside)  [required]
  --device TEXT                      GPU device for a worker. Repeat for each worker.
                                     [default: cuda:0, cuda:1]
  --segment INTEGER                  Demucs segment size in seconds for VRAM optimisation
  --compute-type [float16|int8_float16|int8|float32]
                                     WhisperX compute type  [default: float16]
  --batch-size INTEGER               WhisperX batch size  [default: 16]
  --language TEXT                    Language code (e.g. 'en') or 'auto'  [default: en]
  --whisper-model TEXT               WhisperX model name  [default: large-v3]
  --hf-token TEXT                    HuggingFace token for Pyannote
  --sentiment                        Enable step 4: Text Sentiment Analysis
  --no-resume                        Re-process all files, ignoring existing outputs
  --keep-scratch                     Keep all ghost-track stems after processing
  --temp-limit INTEGER               GPU temperature limit in °C.  [default: 80]
                                     Each worker shuts down if its GPU exceeds this threshold.
                                     Set to 0 to disable.
  --power-limit INTEGER              Set nvidia-smi power limit (watts) on each GPU before launch.
                                     Requires passwordless sudo for nvidia-smi.
  --help                             Show this message and exit.
```

### Power limit / sudoers

`--power-limit` invokes `sudo nvidia-smi -pl <watts>`. To allow this without a password prompt:

```bash
echo 'YOUR_USERNAME ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi' | sudo tee /etc/sudoers.d/nvidia-smi
```

### Troubleshooting

- **A worker exits non-zero** — Inspect `<base>/logs/worker_N.log` and `<base>/summary/combined_report.json`.
- **`nvidia-smi power limit failed: Insufficient Permissions`** — Add the sudoers rule above, or set limits manually with `sudo nvidia-smi -i 0 -pl 350` before running.
- **`Thermal shutdown: Worker WN (cuda:N) at XX°C`** — A worker's GPU exceeded `--temp-limit`. The offending worker is terminated; the others continue. Resume with `audio-refinery pipeline-parallel --base-dir ...` — completed files are skipped automatically.

---

## Slack Notifications

Set `SLACK_WEBHOOK_URL` in your `.env` file (or environment) to receive notifications when a pipeline run completes or shuts down due to overheating:

```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
```

Notifications are fire-and-forget — a failure to deliver never blocks or aborts the pipeline.

---

## Documentation

| Document                             | Description                                                        |
|--------------------------------------|--------------------------------------------------------------------|
| [Architecture](docs/ARCHITECTURE.md) | Ghost Track pipeline design, model selection rationale, data model |
| [Use Cases](docs/USE_CASES.md)       | Who uses this and for what                                         |
| [Performance](docs/PERFORMANCE.md)   | Throughput benchmarks, scaling options, optimization guide         |
| [Deployment](docs/DEPLOYMENT.md)     | Production patterns, async workers, Docker, monitoring             |
| [Development](docs/DEVELOPMENT.md)   | Dev setup, testing, contributing, release process                  |

---

## Development

```bash
uv venv --python 3.11.14
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run unit tests (no GPU required)
pytest tests/ -m "not integration" -v

# Run integration tests (requires GPU, HF_TOKEN, and test audio)
pytest tests/ -m integration -v

# Lint and format
ruff check src/ tests/
ruff format src/ tests/
```

---

## License & Dependencies

**audio-refinery** is released under the [MIT License](LICENSE).

**Dependency note:** The Pyannote model weights (`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`) are gated on HuggingFace under separate terms. If you run this tool in a commercial data product, verify that your HuggingFace account's accepted terms cover your use case. The MIT license on this software does not extend to the model weights — those are governed by their respective HuggingFace model cards.
