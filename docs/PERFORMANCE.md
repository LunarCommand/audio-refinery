# Performance

This document covers measured throughput figures, stage-level time distribution, software
optimizations, and scaling strategies for processing large audio corpora.

## Baseline Throughput

The following figures are measured on an **NVIDIA RTX 4090 FE** (Ada Lovelace, 24 GB GDDR6X)
with the default pipeline configuration: `large-v3` transcription model, `float16` compute type,
batch size 16, interleaved per-file processing.

| Metric                           | Value                |
|:---------------------------------|:---------------------|
| Average processing time per file | ~85 seconds (1m 25s) |
| Files per hour                   | ~42                  |
| Files per 24-hour day            | ~1,017               |

These figures assume audio files of roughly 5–8 minutes average duration. Actual throughput
scales with audio duration — the relevant performance metric for planning is the **real-time
factor (RTF)** per stage (see below).

## Stage-Level Time Distribution

The 85-second average is a composite of three independent model workloads:

| Stage               | Model                              | Est. % of Total | Est. Time / File |
|:--------------------|:-----------------------------------|:---------------:|:----------------:|
| Vocal separation    | Demucs `htdemucs`                  |      ~15%       |       ~13s       |
| Speaker diarization | Pyannote `speaker-diarization-3.1` |      ~25%       |       ~21s       |
| Transcription       | WhisperX `large-v3` + Wav2Vec2     |      ~60%       |       ~51s       |

**Transcription dominates** because WhisperX `large-v3` is a 1.55B-parameter encoder-decoder
model, and the forced alignment pass (Wav2Vec2) runs a second full-model inference over the same
audio. This makes transcription the primary target for both software optimization and hardware
acceleration.

## Real-Time Factor (RTF)

RTF measures how long processing takes relative to the audio duration. An RTF of 0.044 means
one second of audio takes 0.044 seconds to process.

| Stage                             | Approximate RTF  |
|:----------------------------------|:----------------:|
| Vocal separation (Demucs)         |      ~0.012      |
| Speaker diarization (Pyannote)    |      ~0.021      |
| Transcription (WhisperX large-v3) |      ~0.044      |
| **Total pipeline**                | **~0.075–0.095** |

RTF scales linearly with audio duration, making it useful for projecting processing time over
large corpora: `corpus_audio_hours × stage_RTF = stage_GPU_hours`.

## Software Optimizations

These settings require no hardware changes and can be applied immediately.

### INT8 Quantization (`--compute-type int8_float16`)

WhisperX uses CTranslate2 as its inference backend, which supports INT8 quantization natively.
The default is `float16`.

| Compute type        |  VRAM  | Throughput |       Accuracy        |
|:--------------------|:------:|:----------:|:---------------------:|
| `float32`           |  High  |    Slow    |       Reference       |
| `float16` (default) | Medium |     1×     | ~identical to float32 |
| `int8_float16`      |  Low   |   ~1.5×    |   negligible delta    |
| `int8`              | Lowest | ~1.7–2.0×  |   negligible delta    |

`int8_float16` performs computation in INT8 but accumulates in FP16, preserving numerical
stability while gaining roughly 1.5× throughput. On an RTX 4090 this reduces transcription
time from ~51s to ~28–34s per file.

```bash
audio-refinery pipeline --base-dir /data/audio --compute-type int8_float16
```

### WhisperX Model Variants (`--whisper-model`)

`large-v3` is the default because it provides the highest accuracy. For corpora where some
transcription error is acceptable, smaller models offer significant throughput gains:

| Model                | Parameters | Speed | WER (English) | Notes                             |
|:---------------------|:----------:|:-----:|:-------------:|:----------------------------------|
| `large-v3` (default) |   1.55B    |  1×   |     ~2.7%     | Highest accuracy                  |
| `distil-large-v3`    |    756M    |  ~2×  |     ~3.0%     | Best accuracy/speed tradeoff      |
| `medium`             |    307M    |  ~3×  |     ~3.4%     | Good for multilingual             |
| `medium.en`          |    307M    | ~3.2× |     ~3.1%     | English-only, slightly better WER |

`distil-large-v3` is a knowledge-distilled variant that retains ~99% of `large-v3`'s accuracy
with approximately 2× higher throughput. For most real-world audio it is the best first
optimization to try.

```bash
audio-refinery pipeline --base-dir /data/audio --whisper-model distil-large-v3
```

### Combined Quick-Win Configuration

Applying both `int8_float16` and `distil-large-v3` reduces per-file processing time by
approximately 35–50% with negligible accuracy impact:

```bash
audio-refinery pipeline \
  --base-dir /data/audio \
  --compute-type int8_float16 \
  --whisper-model distil-large-v3
```

Estimated throughput improvement: ~1,017 → ~1,700+ files/day on a single RTX 4090.

## Multi-GPU Parallelism

`audio-refinery pipeline-parallel` spawns one independent pipeline worker per `--device` flag,
partitioning the input file list across workers before launch. Workers share input and output
directories but never process the same file.

### Dual-GPU Example (RTX 4090 + RTX 3090 Ti)

Adding a second GPU — even an older generation card — provides near-linear throughput scaling.
A 3090 Ti running in parallel alongside the 4090 adds approximately 635 files/day of additional
capacity at no recurring cost:

| GPU                      | Architecture | FP16 TFLOPS | Est. Time / File | Files / Day |
|:-------------------------|:-------------|:-----------:|:----------------:|:-----------:|
| RTX 4090 FE              | Ada Lovelace |    ~330     |       ~85s       |   ~1,017    |
| RTX 3090 Ti              | Ampere       |    ~160     |      ~136s       |    ~635     |
| **Both (data parallel)** | —            |      —      |        —         | **~1,652**  |

```bash
# Default: cuda:0 and cuda:1
audio-refinery pipeline-parallel --base-dir /data/audio

# Explicit device assignment
audio-refinery pipeline-parallel \
  --base-dir /data/audio \
  --device cuda:0 \
  --device cuda:1 \
  --compute-type int8_float16
```

Combined with software optimizations, dual-GPU throughput reaches approximately 2,800 files/day.

### Three or more GPUs

Add additional `--device` flags for each additional GPU:

```bash
audio-refinery pipeline-parallel \
  --base-dir /data/audio \
  --device cuda:0 \
  --device cuda:1 \
  --device cuda:2
```

Throughput scales approximately linearly with the number of workers.

## Cloud Scaling

For burst workloads or corpora that exceed local hardware capacity, cloud GPU instances provide
an elastic ceiling.

### H100 Comparison

The NVIDIA H100 represents a qualitative step beyond consumer GPUs for transformer inference
workloads:

| Specification    | RTX 4090 FE  | H100 SXM5 80GB |
|:-----------------|:------------:|:--------------:|
| Architecture     | Ada Lovelace |     Hopper     |
| FP16 TFLOPS      |     ~330     |     ~1,979     |
| INT8 TOPS        |     ~660     |     ~3,958     |
| VRAM             |    24 GB     |     80 GB      |
| Memory bandwidth |  1,008 GB/s  |   3,350 GB/s   |

The H100's advantages compound for audio pipeline workloads:

- **3.3× higher memory bandwidth** accelerates the attention mechanisms in WhisperX and Pyannote,
  both of which are memory-bandwidth-bound during inference.
- **80 GB VRAM** allows holding all three models (Demucs ~4 GB, Pyannote ~1 GB, WhisperX
  large-v3 ~10 GB = ~15 GB total) simultaneously with room for much larger batch sizes.
- **Transformer Engine (FP8)** provides hardware-accelerated 8-bit inference with roughly 2×
  throughput over FP16.

Real-world end-to-end speedup for mixed audio pipelines on H100 vs. RTX 4090 is typically
**4–6×**, accounting for stages that are I/O-bound rather than compute-bound.

### H100 Throughput Estimates

| Scenario                         | Time / File | Files / Day  |
|:---------------------------------|:-----------:|:------------:|
| Single RTX 4090 FE (baseline)    |    ~85s     |    ~1,017    |
| Single H100 SXM5 (4× speedup)    |    ~21s     |    ~4,114    |
| Single H100 + INT8 + large batch |   ~12–15s   | ~5,760–7,200 |

### Cloud Provider Pricing (early 2026)

| Provider      | GPU            | On-Demand  | Spot / Interruptible |
|:--------------|:---------------|:----------:|:--------------------:|
| CoreWeave     | H100 SXM5 80GB | ~$4.25/hr  |      ~$2.50/hr       |
| Lambda Labs   | H100 SXM5 80GB | ~$3.99/hr  |         N/A          |
| RunPod        | H100 SXM5 80GB | ~$4.69/hr  |      ~$2.99/hr       |
| AWS p5.xlarge | H100 SXM5 80GB | ~$17.98/hr |   ~$5.40/hr (Spot)   |

**Cost example — 5,000 files on a single CoreWeave H100 spot instance (~0.5 days):**
0.5 days × 24 hr × $2.50/hr ≈ **~$30**

For cloud deployment, the pipeline's PyTorch 2.1.2 + CUDA 12.1 dependency stack should be
containerized. See [DEPLOYMENT.md](DEPLOYMENT.md) for containerization guidance.

## Combined Scenario Matrix

| Scenario                   | Hardware          | Software Opts          | Files/Day | Days for 5k |
|:---------------------------|:------------------|:-----------------------|:---------:|:-----------:|
| Baseline                   | RTX 4090 FE       | None                   |  ~1,017   |    ~4.9     |
| Quick win                  | RTX 4090 FE       | INT8 + distil-large-v3 |  ~1,700+  |    ~2.9     |
| Local dual-GPU             | 4090 FE + 3090 Ti | None                   |  ~1,652   |    ~3.0     |
| Local dual-GPU + optimized | 4090 FE + 3090 Ti | INT8 + distil-large-v3 |  ~2,800   |    ~1.8     |
| Cloud H100                 | 1× H100           | None                   |  ~4,114   |    ~1.2     |
| Cloud H100 + optimized     | 1× H100           | INT8 + batch 64        |  ~6,000+  |    ~0.8     |

## Recommendations

**For a single GPU:** Apply `--compute-type int8_float16` first. It is a flag change with ~1.5×
throughput gain and no accuracy trade-off. If accuracy allows, additionally switch to
`--whisper-model distil-large-v3` for a further ~2× improvement on the transcription stage.

**For multiple GPUs:** Use `pipeline-parallel`. Throughput scales approximately linearly with
GPU count. Apply software optimizations on each worker for maximum throughput.

**For large one-time backlogs:** Cloud H100 spot instances are cost-effective for burst
processing. A 5,000-file backlog clears in under a day for roughly $30–80 depending on provider
and instance configuration.

**For sustained high volume:** Local dual-GPU with software optimizations handles approximately
2,800 files/day with no recurring infrastructure cost. Cloudburst provides an elastic ceiling
for volume spikes beyond this.
