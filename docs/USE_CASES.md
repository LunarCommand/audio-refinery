# Use Cases

Audio-refinery is a general-purpose GPU-accelerated pipeline for extracting structured, searchable
metadata from audio recordings. It is particularly well-suited to any workload where the audio
is long-form, contains multiple speakers, and is mixed with background music or environmental noise.

## Building AI-Ready Audio Databases

### Semantic audio indexing

The `TranscriptionResult` JSON produced by audio-refinery is designed to be consumed by
downstream AI systems. Each segment has:

- Exact start/end timestamps (word-level)
- Speaker label
- The transcribed text
- Optional sentiment classification

This structure maps directly to the vector store ingestion patterns used by RAG (Retrieval
Augmented Generation) systems. Each segment can be embedded and stored with its timestamps as
metadata, enabling semantic search over audio content: "find the segment where someone explains
the refund policy" retrieves the exact timestamp from any file in the corpus.

### Integration with downstream inference pipelines

Audio-refinery is designed as a processing stage, not a complete system. Its JSON outputs are
the input interface for whatever comes next — a semantic search index, a fine-tuning dataset,
a content management system, or an agentic workflow. The Pydantic models are stable and
versioned; downstream consumers can deserialize them directly.

The pipeline's resume behavior (skipping already-processed files) makes it straightforward to
run audio-refinery incrementally as new audio arrives, keeping a downstream database up to date
without reprocessing the entire corpus.

### Training data preparation

Labeled, speaker-attributed transcripts are a core input for fine-tuning speech and language
models. Audio-refinery produces training-ready output: each segment has text, audio timestamps,
speaker identity, and (optionally) sentiment labels. Combined with the original audio file and
the word-level alignment data, the output contains everything needed to construct ASR, diarization,
or spoken language understanding training samples.

## Archival Audio Processing

### Vinyl and analog media digitization

Vinyl records, cassette tapes, and other analog media contain spoken content — audiobooks,
children's records, radio broadcasts, oral histories — that is valuable but unsearchable. The
content is often mixed with orchestral scores or sound effects, making naive transcription
unreliable.

The Ghost Track pipeline was designed precisely for this use case. Demucs separates the vocal
content from the musical accompaniment; Pyannote identifies speaker turns even when multiple
narrators or characters are present; WhisperX produces word-level transcripts that can be used
to build a searchable index of the entire archive.

**What you get out:** A JSON record for every recording with exact timestamps for every utterance,
speaker labels, and optionally sentiment scores — making a physical archive fully searchable
and queryable without manual transcription.

### Broadcast archives

Radio shows, podcasts, documentary recordings, and news broadcasts share the same challenge:
long files, multiple speakers, and background audio (theme music, jingles, interview room noise).
Audio-refinery processes these files in batch with a consistent output format, allowing an archive
of hundreds or thousands of recordings to be indexed in a single pipeline run.

**Example workflow:**
```
<archive>/
  extracted/                 ← source WAV files (named audio_<id>.wav)
  diarization/               ← per-episode speaker timelines
  transcription/             ← per-episode transcripts with word-level timestamps
  summary/pipeline_summary.json
```

A complete season of weekly 45-minute podcast episodes (52 files) processes on a single
consumer GPU in a few hours using the default configuration.

## Content Creation and Production

### Creator content transcription at scale

Content creators, video producers, and post-production teams often have large backlogs of
recorded audio (interviews, commentary, voiceovers) that need to be transcribed for captions,
show notes, or clip finding. Processing 250–500 files through a service API is expensive and
creates a dependency on network availability. Audio-refinery runs locally on any CUDA-capable
GPU and processes the same volume in hours.

Because each `TranscriptionResult` captures full provenance (source file, timestamps,
speaker labels), the output integrates directly into editorial workflows — NLEs, caption
editors, and search tools can consume the JSON directly.

### Clip retrieval and highlight extraction

Word-level timestamps make it possible to programmatically locate any utterance in a corpus
by searching the JSON output. A simple grep for a topic keyword returns the exact file, start
time, and speaker for every matching segment across hundreds of recordings — enabling fast
highlight extraction without manually scrubbing through audio.

Combined with sentiment scores from stage 4, this enables queries like "find segments in the
'excited' emotional register where the topic is X" — which is not possible from audio alone.

## Research and Analysis

### Academic corpora

Researchers in linguistics, phonetics, conversation analysis, and communication studies often
work with a large corpora of recorded speech. Audio-refinery produces structured JSON with
speaker-attributed, word-level transcripts that map directly to standard corpus annotation
formats. The full provenance model (source file, model used, processing timestamp) supports
reproducible research: the exact conditions under which each transcript was produced are
recorded in the output.

### Interview and qualitative research data

Qualitative researchers conducting interviews typically need transcripts with speaker labels and
timestamps. Audio-refinery automates this from a directory of WAV files. The `DiarizationResult`
identifies and labels speaker turns consistently across all files in a batch; the
`TranscriptionResult` associates each utterance with a speaker label and exact timestamps.

For small batches (single interviews), the individual CLI commands (`separate`, `diarize`,
`transcribe`) allow step-by-step processing with inspection between stages.

### Meeting and conversation analysis

Organizations and researchers analyzing meeting recordings, customer calls, or focus group
sessions can use audio-refinery to build structured datasets from raw recordings. Each output
file contains the full speaker turn structure, making it straightforward to compute per-speaker
word counts, turn-taking patterns, sentiment distributions, and other conversational metrics.

## Hardware Considerations by Use Case

| Use case                       | Files      | Recommended setup                            |
|:-------------------------------|:-----------|:---------------------------------------------|
| Single interview or recording  | 1–10       | Any CUDA GPU, 10GB+ VRAM                     |
| Weekly podcast archive         | 50–200     | Single GPU, 24GB VRAM                        |
| Large content backlog          | 500–5,000+ | Multi-GPU (pipeline-parallel) or cloud burst |
| Ongoing incremental processing | continuous | Single GPU + resume behavior                 |

See [PERFORMANCE.md](PERFORMANCE.md) for detailed throughput figures and scaling options.
