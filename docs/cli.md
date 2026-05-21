# CLI Reference

The `audio-refinery` command-line interface runs the pipeline as one-shot
operations on a workstation — ideal for interactive use, ad-hoc processing, and
batch runs over a local directory of WAV files. For the long-lived HTTP service
(URI-in / URI-out, async jobs, containerized deployment), see
[service.md](service.md).

This page assumes you have already installed the package — see
[Installation](../README.md#installation) in the README and the
[Hugging Face token setup](../README.md#prerequisites), which diarization
requires in either mode.

## Contents

- [Scratch directory](#scratch-directory)
- [Demucs model weights](#demucs-model-weights)
- [Step 1: Separate vocals](#step-1-separate-vocals)
- [Step 2: Diarize](#step-2-diarize)
- [Step 3: Transcribe](#step-3-transcribe)
- [Step 4: Text sentiment analysis](#step-4-text-sentiment-analysis)
- [Batch pipeline](#batch-pipeline)
- [GPU identification and device targeting](#gpu-identification-and-device-targeting)
- [GPU health monitoring](#gpu-health-monitoring)
- [Parallel pipeline (multi-GPU)](#parallel-pipeline-multi-gpu)
- [Slack notifications](#slack-notifications)

---

## Scratch directory

Demucs writes its separated stems to a scratch directory. Resolution order:

1. `$REFINERY_SCRATCH_DIR/demucs` when `REFINERY_SCRATCH_DIR` is set.
2. `tempfile.gettempdir()/audio-refinery-demucs` otherwise — i.e.
   `/tmp/audio-refinery-demucs` on most systems (honors `$TMPDIR`).

Single-stage commands accept `-o/--output-dir` and the batch pipeline accepts
`--demucs-output-dir` to override per invocation.

**RAM disk (recommended for large batches).** Pointing the scratch directory at
a tmpfs mount avoids SSD write amplification during heavy processing and
reduces per-file latency. On a RAM-rich host:

```bash
sudo mkdir -p /mnt/fast_scratch
sudo mount -t tmpfs -o size=32G,mode=1777 tmpfs /mnt/fast_scratch
export REFINERY_SCRATCH_DIR=/mnt/fast_scratch
```

The `mode=1777` flag lets non-root users write to the mount. To persist across
reboots, add to `/etc/fstab`:

```
tmpfs /mnt/fast_scratch tmpfs defaults,size=32G,mode=1777 0 0
```

A RAM disk is optional — the default `/tmp` location works everywhere, and on
RAM-tight machines a fast SSD is a fine target.

## Demucs model weights

The first run auto-downloads the `htdemucs` weights (~80 MB) to
`~/.cache/torch/hub/checkpoints/`. No Hugging Face token or manual download is
required for separation.

---

## Step 1: Separate vocals

```bash
audio-refinery separate /path/to/audio.wav
```

Runs Demucs `htdemucs` with `--two-stems=vocals` on the GPU. Output:

- A blue panel summarizing input/output/device
- An animated spinner while Demucs processes
- A green results table with file info, timing, and output paths
- A JSON dump of the full `SeparationResult` provenance record

Demucs writes two stems under the scratch directory:

```
<scratch>/htdemucs/<track_name>/vocals.wav
<scratch>/htdemucs/<track_name>/no_vocals.wav
```

### Options

```
Usage: audio-refinery separate [OPTIONS] INPUT_FILE

Options:
  -o, --output-dir DIRECTORY  Output directory for separated stems.
                              [default: /tmp/audio-refinery-demucs]
  -d, --device TEXT           Compute device: 'cpu', 'cuda', or 'cuda:N'
                              (e.g. 'cuda:0').  [default: cuda]
  --segment INTEGER           Segment size in seconds for VRAM optimization.
  --help                      Show this message and exit.
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

After step 1 produces `vocals.wav`, feed it to `audio-refinery diarize` to
identify speaker turns.

```bash
audio-refinery diarize /tmp/audio-refinery-demucs/htdemucs/audio_track/vocals.wav
```

Runs Pyannote `speaker-diarization-3.1`. Output:

- A blue panel summarizing input and device
- An animated spinner while Pyannote processes
- A green summary table: speakers detected, total segments, total speech duration
- A per-speaker breakdown table
- A JSON dump of the full `DiarizationResult` provenance record

### Options

```
Usage: audio-refinery diarize [OPTIONS] INPUT_FILE

Options:
  -d, --device TEXT       Compute device: 'cpu', 'cuda', or 'cuda:N'
                          (e.g. 'cuda:0').  [default: cuda]
  --min-speakers INTEGER  Minimum number of speakers (optional hint for Pyannote).
  --max-speakers INTEGER  Maximum number of speakers (optional hint for Pyannote).
  --hf-token TEXT         Hugging Face token (overrides HF_TOKEN env var).
  -o, --output-file FILE  Write DiarizationResult JSON to this file.
  --help                  Show this message and exit.
```

### Common variations

```bash
# Hint that the audio has 2–4 speakers
audio-refinery diarize /path/to/vocals.wav -d cuda:0 --min-speakers 2 --max-speakers 4

# Save result JSON for downstream processing
audio-refinery diarize /path/to/vocals.wav -d cuda:0 -o /tmp/diarization.json
```

### Troubleshooting

- **`Hugging Face token not found`** — Follow the HF setup steps in the README.
- **`Failed to load Pyannote pipeline`** — Check that you accepted both model licenses on Hugging Face.

---

## Step 3: Transcribe

After step 2 produces a `DiarizationResult` JSON, feed the same `vocals.wav` to
`audio-refinery transcribe` to generate word-level timestamps. Optionally pass
the diarization JSON to get speaker labels on each segment.

```bash
# Pure transcription (no speaker labels)
audio-refinery transcribe /path/to/vocals.wav -d cuda:0 -o /tmp/transcription.json

# With speaker assignment from step 2
audio-refinery transcribe /path/to/vocals.wav \
  -d cuda:0 \
  --diarization-file /tmp/diarization.json \
  -o /tmp/transcription_with_speakers.json
```

Runs WhisperX (`large-v3`) with Wav2Vec2 forced alignment. Output:

- A blue panel summarizing input, device, compute type, and language
- An animated spinner while processing
- A green summary table: language, segments, words, processing time
- A transcript preview table (first 10 segments, with optional speaker column)
- A JSON dump of the full `TranscriptionResult` provenance record

### Options

```
Usage: audio-refinery transcribe [OPTIONS] INPUT_FILE

Options:
  -d, --device TEXT               Compute device: 'cpu', 'cuda', or 'cuda:N'
                                  (e.g. 'cuda:0').  [default: cuda]
  --compute-type [float16|int8_float16|int8|float32]
                                  CTranslate2 compute type.  [default: float16]
  --batch-size INTEGER            Batch size for transcription.  [default: 16]
  --language TEXT                 Language code (e.g. 'en', 'fr') or 'auto'
                                  for detection.  [default: en]
  --diarization-file FILE         DiarizationResult JSON from step 2 for
                                  speaker assignment.
  -o, --output-file FILE          Write TranscriptionResult JSON to this file.
  --help                          Show this message and exit.
```

`int8_float16` gives roughly 1.5× throughput with a negligible accuracy delta.

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

- **`whisperx is not installed`** — Follow the install steps. whisperx requires manual installation due to ctranslate2 version constraints.
- **`No such file or directory: 'ffmpeg'`** — Install the ffmpeg binary: `sudo apt install ffmpeg`
- **`CUDA out of memory`** — Try `--compute-type int8` to reduce VRAM usage, or lower `--batch-size`.
- **`alignment failed` warning** — No Wav2Vec2 alignment model for the detected language. Transcription falls back to Whisper's raw timestamps; segment text is still correct.

---

## Step 4: Text sentiment analysis

After step 3 produces a `TranscriptionResult` JSON, feed it to
`audio-refinery sentiment` to score each segment as positive, neutral, or
negative. **This step is text-only — no audio file is required.**

The analyzer uses
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
and writes results in two places:

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

### Options

```
Usage: audio-refinery sentiment [OPTIONS] TRANSCRIPTION_FILE

Arguments:
  TRANSCRIPTION_FILE     TranscriptionResult JSON written by step 3  [required]

Options:
  --model TEXT           Hugging Face text-classification model name.
                         [default: cardiffnlp/twitter-roberta-base-sentiment-latest]
  -d, --device TEXT      Compute device: 'cpu', 'cuda', or 'cuda:N'.  [default: cpu]
  -o, --output-file FILE Write SentimentResult JSON to this file.
  --help                 Show this message and exit.
```

Segments with no transcribed speech (e.g. silent audio) are skipped rather than
treated as failures — the run completes and the affected segments simply carry
no sentiment.

---

## Batch pipeline

`audio-refinery pipeline` runs all active stages (separation → diarization →
transcription, and optionally sentiment) across an entire directory of WAV
files. Processing is interleaved per file: each file completes all active stages
before the next begins, so ghost-track stems never accumulate beyond one file's
worth of data. All models are loaded once before the loop starts.

### Directory layout

```
<base>/
  extracted/                    ← input WAV files (must exist)
  diarization/                  ← intermediate Pyannote JSON (created automatically)
  transcription/                ← final WhisperX JSON output (created automatically)
  sentiment/                    ← sentiment JSON (created when --sentiment is set)
  summary/                      ← run summary JSON (created automatically)
    pipeline_summary.json
```

If `<base>/extracted` does not exist the pipeline exits immediately with an
error. All other subdirectories are created on demand.

**Input files.** Any `*.wav` file in `extracted/` is processed. The content ID
is the filename stem with an optional `audio_` prefix stripped — i.e.
`audio_xyz.wav` and `xyz.wav` both produce content ID `xyz`.

**Demucs scratch** resolves to `$REFINERY_SCRATCH_DIR/demucs` when set, or
`tempfile.gettempdir()/audio-refinery-demucs` otherwise. Point the env var at a
tmpfs mount for the RAM-backed throughput benefit (see
[Scratch directory](#scratch-directory)), or override with
`--demucs-output-dir`.

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

### Options

```
Usage: audio-refinery pipeline [OPTIONS]

Options:
  --base-dir DIRECTORY               Base directory (extracted/ must exist inside)  [required]
  -d, --device TEXT                  Compute device: 'cpu', 'cuda', or 'cuda:N'  [default: cuda]
  --segment INTEGER                  Demucs segment size in seconds for VRAM optimisation
  --compute-type [float16|int8_float16|int8|float32]
                                     WhisperX CTranslate2 compute type  [default: float16]
  --batch-size INTEGER               WhisperX batch size  [default: 16]
  --language TEXT                    Language code (e.g. 'en') or 'auto'  [default: en]
  --hf-token TEXT                    Hugging Face token for Pyannote (overrides HF_TOKEN env var)
  --no-resume                        Re-process all files, ignoring existing outputs
  --keep-scratch                     Keep Demucs stems on the scratch disk after the run
  --whisper-model TEXT               WhisperX model name  [default: large-v3]
                                     Variants: large-v3 (highest accuracy), distil-large-v3
                                     (~2× faster), medium.en (~3.2× faster, English-only),
                                     medium (~3× faster, multilingual)
  --temp-limit INTEGER               GPU temperature limit in °C.  [default: 80]
                                     A background thread checks every 5 s and shuts the
                                     pipeline down if the limit is exceeded. Set to 0 to disable.
  --sentiment                        Enable step 4: Text Sentiment Analysis
  --emotion                          Enable step 5: Speech Emotion Recognition [not yet implemented]
  --events                           Enable step 6: Audio Event Detection via CLAP
                                     [not yet implemented]
  --help                             Show this message and exit.
```

Steps 5 (`--emotion`) and 6 (`--events`) are scaffolded but not yet
implemented; the flags are reserved and currently no-ops.

### Scratch-disk management

By default, ghost-track stems are cleaned up per file as soon as they are no
longer needed:

- `no_vocals.wav` — deleted immediately after separation
- `vocals.wav` — deleted after transcription

This bounds scratch usage to roughly one file's worth of stems (~400 MB) at any
point. Use `--keep-scratch` to retain all stems.

### Resume behaviour

By default, the pipeline skips any file whose transcription output already
exists and is non-empty. An interrupted run can be safely restarted. Use
`--no-resume` to force full reprocessing.

### Summary file

At the end of every run, `<base>/summary/pipeline_summary.json` is written with
per-stage counts (processed / skipped / failed), timing, active steps, and a
list of any failures.

### Troubleshooting

- **`Source directory not found`** — Create `<base>/extracted/` and place `.wav` files inside.
- **No files processed** — Confirm `extracted/` contains `.wav` files.
- **Partial failures** — Check the failure table printed at the end of the run and `summary/pipeline_summary.json`.
- **`Thermal shutdown: cuda:N at XX°C`** — The GPU exceeded `--temp-limit`. Wait for it to cool, then resume (completed files are skipped automatically).

---

## GPU identification and device targeting

CUDA and `nvidia-smi` use independent GPU numbering schemes by default, and they
don't always agree. Without corrective configuration, `cuda:0` in PyTorch may
refer to a different physical card than index `0` in `nvidia-smi`, making it
impossible to reliably pin workloads to specific GPUs.

Audio-refinery sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` at startup, before any CUDA
context is created. This forces both CUDA and `nvidia-smi` to enumerate GPUs in
PCI bus order, so their indices always match.

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

With `CUDA_DEVICE_ORDER=PCI_BUS_ID` set, `cuda:0` in any audio-refinery command
maps to index 0 in this output, and `cuda:1` maps to index 1.

### Device string conventions

The `--device` flag uses PyTorch device syntax:

| String   | Meaning                                                                  |
|----------|--------------------------------------------------------------------------|
| `cuda`   | Default GPU (typically `cuda:0` in PCI bus order)                        |
| `cuda:0` | GPU at PCI bus index 0                                                   |
| `cuda:1` | GPU at PCI bus index 1                                                   |
| `cpu`    | CPU only — significantly slower; useful for testing or GPU-free machines |

The same `cuda:N` targeting works in service mode via the `REFINERY_DEVICE`
environment variable — see [service.md](service.md).

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

`audio-refinery pipeline-parallel` uses `detect_gpu_order()` to rank all
available GPUs before assigning workers. The ranking uses FP16 TFLOPS from
`src/gpu_tflops.toml` as the primary sort key, so the fastest GPU gets the first
worker slot. GPUs not in the table fall back to a `(rounded VRAM GB, max SM
clock)` heuristic — reliable within a generation but not across them.

To check whether your GPU is in the table:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader
```

Compare the output against the entries in `src/gpu_tflops.toml`. If your GPU is
listed, it ranks above any unlisted GPU. If not, the heuristic applies.

### Adding a GPU to the TFLOPS table

If your GPU is not in `src/gpu_tflops.toml`, add it for correct ranking in
multi-GPU setups:

1. Get the exact name as `nvidia-smi` reports it:
   ```bash
   nvidia-smi --query-gpu=name --format=csv,noheader
   ```

2. Find the FP16 TFLOPS figure at [TechPowerUp GPU Specs](https://www.techpowerup.com/gpu-specs/).
   Use the **Shader Performance** row for consumer GPUs, or **FP16 (half)** for
   data center GPUs. See the comments in `src/gpu_tflops.toml` for guidance.

3. Add an entry to `src/gpu_tflops.toml`:
   ```toml
   "NVIDIA GeForce RTX 5080" = 86.4
   ```

The GPU name must match `nvidia-smi` output exactly. The lookup code handles the
presence or absence of the `NVIDIA ` prefix across driver versions.

---

## GPU health monitoring

### Pre-flight availability check

Before starting any model work, every GPU command queries `nvidia-smi` to check
whether the targeted device is already occupied. If active processes are found,
a yellow panel lists each PID and its VRAM footprint and asks for confirmation
before proceeding.

### Live temperature display

During `audio-refinery pipeline` runs the current GPU temperature is shown in
the progress bar description, colour-coded relative to the configured limit:

| Colour | Meaning                                        |
|--------|------------------------------------------------|
| green  | More than 10°C below the limit (safe)          |
| yellow | Within 10°C of the limit (warm — watch it)     |
| red    | At or above the limit (shutdown imminent)      |

Temperature is sampled at most once every 5 seconds to avoid hammering
`nvidia-smi`.

### External monitoring

```bash
# Print temperature + GPU utilisation every 10 s, log to file
nvidia-smi dmon -s tu -d 10 | tee /tmp/gpu_temps.log
```

---

## Parallel pipeline (multi-GPU)

`audio-refinery pipeline-parallel` runs N simultaneous pipeline workers — one
per `--device` flag — for significantly higher throughput when multiple GPUs are
available.

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
# Use all detected GPUs (ranked best-first)
audio-refinery pipeline-parallel --base-dir /data/audio/batch

# Recommended: int8_float16 for ~1.5× per-GPU speedup
audio-refinery pipeline-parallel --base-dir /data/audio/batch \
  --compute-type int8_float16

# With power limits (recommended for extended runs)
audio-refinery pipeline-parallel --base-dir /data/audio/batch \
  --compute-type int8_float16 \
  --power-limit 350

# Explicit three-GPU run
audio-refinery pipeline-parallel --base-dir /data/audio/batch \
  --device cuda:0 --device cuda:1 --device cuda:2

# Monitor workers in separate terminals
tail -f /data/audio/batch/logs/worker_0.log
tail -f /data/audio/batch/logs/worker_1.log
```

### Options

```
Usage: audio-refinery pipeline-parallel [OPTIONS]

Options:
  --base-dir DIRECTORY               Base directory (extracted/ must exist inside)  [required]
  --device TEXT                      GPU device for a worker. Repeat for each worker:
                                     --device cuda:0 --device cuda:1. Order determines
                                     partition priority — first device gets the largest
                                     files. Defaults to all GPUs detected by nvidia-smi,
                                     ranked best-first.
  --segment INTEGER                  Demucs segment size in seconds for VRAM optimisation
  --compute-type [float16|int8_float16|int8|float32]
                                     WhisperX CTranslate2 compute type  [default: float16]
  --batch-size INTEGER               WhisperX batch size  [default: 16]
  --language TEXT                    Language code (e.g. 'en') or 'auto'  [default: en]
  --hf-token TEXT                    Hugging Face token for Pyannote (overrides HF_TOKEN env var)
  --no-resume                        Re-process all files, ignoring existing outputs
  --keep-scratch                     Keep Demucs stems on the scratch disk after the run
  --whisper-model TEXT               WhisperX model name  [default: large-v3]
  --temp-limit INTEGER               GPU temperature limit in °C.  [default: 80]
                                     Each worker monitors its own device every 5 s; the
                                     coordinator terminates a worker that exceeds the limit.
                                     Set to 0 to disable.
  --power-limit INTEGER              Set nvidia-smi power limit (watts) on each GPU before
                                     launch. Recommended: 350. Requires passwordless sudo.
  --sentiment                        Enable step 4: Text Sentiment Analysis (forwarded to each worker)
  --help                             Show this message and exit.
```

### Combined report fields

`combined_report.json` is always written after all workers finish. It contains
aggregate metrics across all workers:

| Field | Type | Description |
|---|---|---|
| `run_at` | string | ISO 8601 timestamp of run start (UTC) |
| `total_discovered` | int | Total WAV files found in `extracted/` |
| `total_time_seconds` | float | Wall-clock seconds from first worker start to last finish |
| `total_audio_hours` | float | Total audio duration processed across all workers |
| `source_audio_bytes` | int | Combined size of all input WAV files |
| `total_words` | int | Total words transcribed across all files |
| `total_segments` | int | Total transcript segments across all files |
| `avg_time_per_file_seconds` | float | `total_time / total_discovered` — average wall-clock cost per file |
| `avg_time_per_mb_seconds` | float | `total_time / source_MB` — processing seconds per MB of source audio |
| `processing_speed_ratio` | float | `audio_seconds / wall_seconds` — real-time factor (e.g. `3.7` = 3.7× faster than playback) |
| `words_per_audio_hour` | float | Transcription density — useful for detecting sparse/silent audio or diarization misses |
| `gpu_temp_celsius` | object | Per-device temperature summary: `peak_celsius`, `avg_celsius`, `sample_count` |
| `workers` | array | Per-worker label, device, exit code, and individual summary |
| `combined_failures` | array | Aggregated failure records from all workers |

`null` is written for derived metrics when the divisor is zero (e.g.
`avg_time_per_file_seconds` is `null` if no files were discovered).

### Power limit / sudoers

`--power-limit` invokes `sudo nvidia-smi -pl <watts>`. To allow this without a
password prompt:

```bash
echo 'YOUR_USERNAME ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi' | sudo tee /etc/sudoers.d/nvidia-smi
```

### Troubleshooting

- **A worker exits non-zero** — Inspect `<base>/logs/worker_N.log` and `<base>/summary/combined_report.json`.
- **`nvidia-smi power limit failed: Insufficient Permissions`** — Add the sudoers rule above, or set limits manually with `sudo nvidia-smi -i 0 -pl 350` before running.
- **`Thermal shutdown: Worker WN (cuda:N) at XX°C`** — A worker's GPU exceeded `--temp-limit`. The offending worker is terminated; the others continue. Resume with the same command — completed files are skipped automatically.

---

## Slack notifications

Set `SLACK_WEBHOOK_URL` in your `.env` file (or environment) to receive a
notification when a pipeline run completes or shuts down due to overheating:

```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
```

Notifications are fire-and-forget — a failure to deliver never blocks or aborts
the pipeline. The same variable is honored in service mode.
