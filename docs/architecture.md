# Architecture

This document explains the design rationale behind audio-refinery: why the pipeline is structured
the way it is, how each model was chosen, and the engineering decisions that make high-accuracy
processing of noisy, music-heavy audio possible.

## The Problem

Raw audio recordings — vinyl records, podcast episodes, recorded meetings, broadcast captures —
rarely contain isolated speech. Dialogue is mixed with background music, ambient noise, and
overlapping voices. AI models for transcription and speaker identification were largely trained on
clean speech and perform significantly worse on mixed audio. The naive approach of feeding a
recording directly into a speech recognition model produces poor results: missed words,
hallucinated text, incorrect speaker boundaries, and timestamps that drift by seconds.

Three specific problems must be solved to produce accurate, speaker-attributed transcripts from
real-world audio:

**The Mix Problem** — Dialogue is mixed with music, sound effects, or environmental noise.
Transcription models trained on clean speech lose accuracy when competing signals are present.

**The Identity Problem** — Audio contains one or more people speaking, but no labels indicate
who is who. Speaker diarization models cluster voices by acoustic similarity; they perform best
when given clean vocal content, not audio where a trumpet blast might register as a human voice.

**The Boundary Problem** — Precise start and end timestamps for each utterance are required for
downstream use (search, retrieval, editing). Standard ASR models produce timestamps at the
segment level, which can drift by seconds. Word-level timestamp accuracy requires a separate
forced alignment pass.

## The Ghost Track Solution

audio-refinery solves these problems with what can be called the **Ghost Track** strategy: run
all AI models against a temporary, cleaned version of the audio — the ghost track — then apply
the resulting metadata back to the original.

``` text
Original audio  ──┬──────────────────────────────────────────────► playback / output
                  │
                  └─► Demucs: separate vocals ──► vocals.wav (ghost track)
                                                        │
                                                        ├─► Pyannote: who spoke when?
                                                        │         └─► DiarizationResult
                                                        │
                                                        ├─► WhisperX: what did they say?
                                                        │         └─► TranscriptionResult
                                                        │
                                                        └─► [sentiment, SER, etc.]
                                                                  └─► SentimentResult

Ghost track (vocals.wav) ──► deleted after transcription
```

The ghost track is used exclusively for analysis. It is never the output. This preserves the
original audio's character — including noise, room tone, and musical accompaniment — while
ensuring the AI models have the best possible signal to work with.

## Pipeline Stages

### Stage 1: Vocal Separation (Demucs)

**Model:** `htdemucs` (Hybrid Transformer Demucs)

Demucs applies a U-Net hybrid architecture — operating in both the waveform and spectrogram
domains — to decompose a mixed audio signal into stems. Audio-refinery runs it with
`--two-stems=vocals`, which produces:

- `vocals.wav` — the isolated vocal track (the ghost track)
- `no_vocals.wav` — all other content (music, effects, background)

The `no_vocals.wav` stem is discarded by default. The `vocals.wav` is retained only until
transcription completes.

**Why Demucs specifically?** It is the current state-of-the-art for open-source source
separation, particularly for speech embedded in music. Its hybrid spectrogram+waveform approach
handles both tonal (music) and transient (percussion, effects) interference better than
spectrogram-only or waveform-only architectures.

> **Maintenance status:** The original Meta Research Demucs repository is no longer actively
> maintained. The author has left Meta and created a personal fork at
> [github.com/adefossez/demucs](https://github.com/adefossez/demucs), where only important
> bug fixes will be processed. New feature development has stopped and no further model
> improvements are planned upstream. This is worth tracking if a superior open-source
> separator emerges — the Ghost Track pipeline is model-agnostic at the separation stage and
> could adopt a replacement without changes to any downstream stage.

### Stage 2: Speaker Diarization (Pyannote)

**Model:** `pyannote/speaker-diarization-3.1`

Pyannote answers the question "who spoke when?" via three sequential operations:

1. **Voice Activity Detection (VAD):** Identifies segments containing human speech, filtering out
   silence and non-speech audio.
2. **Speaker Embedding:** Extracts a high-dimensional acoustic fingerprint for each speech
   segment. These embeddings capture the unique characteristics of each voice — timbre, pitch,
   cadence — in a way that makes voices from the same speaker cluster together.
3. **Agglomerative Hierarchical Clustering:** Groups the embeddings into clusters, each
   representing a unique speaker. The output is a timeline of `SPEAKER_N: start → end` entries.

Running diarization on the ghost track (vocals-only) rather than the original mixed audio is
essential. On mixed audio, sustained musical tones can register as voice activity; percussion can
fragment speaker boundaries; reverb from instruments can corrupt speaker embeddings. The ghost
track eliminates all of this, giving Pyannote a clean signal to cluster against.

### Stage 3: Transcription with Forced Alignment (WhisperX)

**Model:** `large-v3` (default) via WhisperX

Standard Whisper produces transcripts in 30-second windows, which introduces two problems: the
model may hallucinate content at window boundaries, and timestamps can drift by seconds from their
true position. Word-level timestamp precision — required for any downstream use that involves
cutting, searching, or retrieving specific utterances — demands a second inference pass.

WhisperX augments Whisper with a forced alignment stage:

1. **Transcription:** `faster-whisper` (CTranslate2 backend) generates the initial text.
2. **Forced Alignment:** A phoneme recognition model (Wav2Vec2) aligns each word in the
   transcript to the audio waveform at the phoneme level. This produces word-level timestamps
   accurate to tens of milliseconds.
3. **Speaker Assignment:** If a `DiarizationResult` is provided, each segment is assigned to the
   speaker whose diarization interval it falls within.

The combination of these three outputs — text, word-level timestamps, and speaker labels — is
what makes the `TranscriptionResult` JSON useful for downstream search, editing, and analysis.

### Stage 4: Sentiment Analysis (HuggingFace Transformers)

**Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (default; configurable)

The sentiment stage is text-only: it consumes the `TranscriptionResult` JSON produced by stage 3
and classifies each segment as positive, neutral, or negative. No audio or GPU access is required.

Results are written in two places: a standalone `SentimentResult` JSON, and as an in-place update
to the `TranscriptionResult` — each segment gains a `sentiment` field. This allows the
transcription file to serve as a self-contained record of both what was said and how it was said.

## Data Model and Provenance

Every pipeline stage produces a Pydantic model serialized to JSON. Each output captures full
provenance: input file path, model used, compute device, processing timestamps, wall-clock
processing time, and the full result.

```
SeparationResult     → input_file, output_dir, model, device, vocals_path, ...
DiarizationResult    → input_file, model, device, num_speakers, segments: [SpeakerSegment]
TranscriptionResult  → input_file, model, device, language, segments: [TranscriptSegment]
                       ↳ TranscriptSegment → text, start, end, speaker, words: [WordSegment]
SentimentResult      → source_file, model, segments: [SegmentSentiment]
```

Downstream stages consume the JSON output of upstream stages — the `TranscriptionResult.json`
path is fed to `sentiment`, the `DiarizationResult.json` path is fed to `transcribe`. This
makes each stage independently testable and restartable: an interrupted run can resume from
the last completed stage.

## Batch Processing Design

### Interleaved vs. Staged Processing

An early architectural decision was whether to run all files through stage 1 before starting
stage 2 (staged) or to carry each file through all stages before moving to the next (interleaved).

Staged processing was abandoned due to a critical scratch-disk problem: accumulating all Demucs
stems before starting diarization fills the scratch disk proportionally to the corpus size. A
batch of 200 files (each producing a ~200 MB `vocals.wav`) requires ~40 GB of scratch space before
a single diarization job runs. On a RAM disk, this exhausts the available tmpfs.

**Interleaved processing** bounds scratch-disk usage to a single file's worth of stems
(~400 MB) at any point. The pipeline processes each file through all active stages, then deletes
the ghost-track stems before moving to the next file. Models are loaded once at startup and
remain in VRAM for the duration of the run — there is no model-reload overhead per file.

### Resident Model Strategy

All three GPU-resident models fit simultaneously in approximately 15 GB of VRAM:

| Model                              | Stage            | Peak VRAM  |
|:-----------------------------------|:-----------------|:-----------|
| Demucs `htdemucs`                  | Vocal separation | ~4 GB      |
| Pyannote `speaker-diarization-3.1` | Diarization      | ~1 GB      |
| WhisperX `large-v3`                | Transcription    | ~10 GB     |
| **Total**                          | —                | **~15 GB** |

This has two significant consequences.

**No load/unload overhead.** All models are loaded once before the file loop begins and stay
resident for the entire run. There is no stage-transition cost — moving from separation to
diarization to transcription per file involves no model I/O, only inference. On a 100-file
batch, this eliminates what would otherwise be 300 model load/unload cycles.

**24 GB consumer GPUs are the right hardware tier.** The ~15 GB combined footprint fits
comfortably on any 24 GB card (RTX 3090, 3090 Ti, 4090, RTX 5090, A5000) with 9 GB of
headroom for activations, batch data, and Demucs segment buffers. This is the sweet spot:
24 GB cards are widely available, relatively affordable, and provide enough VRAM to run the
full pipeline without compromise. Data center GPUs (A100, H100) are not required and offer no
architectural advantage for this workload beyond raw throughput — the same pipeline code runs
unchanged.

### Ghost Track Cleanup

Stem cleanup is per-file and staged:

- `no_vocals.wav` — deleted immediately after separation completes
- `vocals.wav` — deleted after transcription completes

Both stems are kept if `--keep-scratch` is set.

### Resume Behavior

By default, the pipeline skips any file whose output for that stage already exists and is
non-empty. This makes interrupted runs safely restartable: restarting the pipeline continues
from the first unprocessed file. `--no-resume` forces full reprocessing.

## Storage Strategy

### RAM Disk (tmpfs)

The Demucs separation stage is the most I/O-intensive operation in the pipeline. Writing stems to
persistent storage causes two problems:

1. **Write amplification:** Separating a file produces two WAV stems, effectively doubling the
   write volume. Processing thousands of files on an NVMe SSD induces significant write
   amplification and measurable drive wear over time.
2. **I/O latency:** Even fast NVMe drives (3–7 GB/s) are an order of magnitude slower than
   system RAM for the sustained sequential writes that Demucs produces.

A RAM disk (`tmpfs`) eliminates both issues. The ghost-track stems are volatile by nature — they
are needed only for the duration of processing and discarded immediately after. Writing them to
RAM rather than persistent storage aligns the storage tier with the data lifecycle.

Audio-refinery uses `/mnt/fast_scratch` as its default RAM disk location. If the mount is not
present, the pipeline falls back to local storage after asking for confirmation.

### Output JSON

All final outputs go to persistent storage under the `--base-dir` directory tree.
The scratch directory holds only the transient ghost-track stems.
