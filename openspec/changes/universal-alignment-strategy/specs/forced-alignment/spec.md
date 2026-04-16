## ADDED Requirements

### Requirement: Forced alignment stage produces word-level timestamps from text and audio

The system SHALL provide a forced-alignment stage that accepts a sanitized text string, an audio waveform, and a language code, and produces a list of aligned words with start time, end time, and a confidence score. The stage SHALL use `torchaudio.pipelines.MMS_FA` as its default acoustic model.

#### Scenario: Aligning a short English utterance

- **WHEN** the stage is called with a ~5 second audio clip, the sanitized text "hello world", and language code "en"
- **THEN** it returns two aligned words whose start/end times fall within the clip bounds, whose ordering matches the text, and whose scores are floats in `[0.0, 1.0]`

#### Scenario: Aligning a non-English utterance

- **WHEN** the stage is called with audio, sanitized text, and a non-English language code that MMS-FA supports
- **THEN** it loads the MMS-FA bundle once, produces aligned words for the input, and does not raise

#### Scenario: Aligner receives an unsupported language code

- **WHEN** the stage is called with a language code MMS-FA does not support
- **THEN** it raises a clear error identifying the unsupported language, and it does not silently fall back to a different language model

### Requirement: Forced alignment chunks audio before running the trellis

The system SHALL chunk input audio into windows of at most 30 seconds before running the forced-alignment trellis, using upstream ASR segment boundaries as the primary chunk delimiter. The system SHALL merge per-chunk results into a single global timeline using each chunk's start offset.

#### Scenario: Ten-minute input with multiple ASR segments

- **WHEN** the stage is called with a 10-minute audio file and 40 ASR segments
- **THEN** it processes the file as multiple chunks each ≤30 seconds long, each chunk's word timings are offset back into the global timeline, and the final output covers the full duration with monotonically increasing start times

#### Scenario: Single ASR segment exceeds the chunk cap

- **WHEN** an ASR segment's audio window is longer than 30 seconds
- **THEN** the stage splits the segment on a silence-based fallback (or reports a clear `fallback_reason` if no silence is found) and continues processing the rest of the file

### Requirement: Forced alignment stage owns its model load and unload lifecycle

The forced-alignment stage SHALL expose explicit `load_model` and `unload_model` entry points. After `unload_model` returns, the MMS-FA acoustic model SHALL no longer hold GPU memory, as observed by `torch.cuda.memory_allocated` returning to its pre-load baseline on the target device.

#### Scenario: Unloading releases VRAM

- **WHEN** the stage loads the MMS-FA model, runs one alignment, and calls `unload_model`
- **THEN** `torch.cuda.memory_allocated(device)` after `unload_model` is within 50 MB of its value before `load_model` was called

### Requirement: Forced alignment output includes provenance

The stage SHALL emit an `AlignmentResult` containing the input audio path, a pointer to the transcription JSON that sourced the text, the language code used, the acoustic model identifier, the device, the number of chunks processed, any `fallback_reason`, processing time, and timestamps.

#### Scenario: Serializing an AlignmentResult

- **WHEN** the stage completes alignment and serializes its result to JSON
- **THEN** the JSON contains `input_file`, `transcription_file`, `language`, `acoustic_model`, `device`, `chunks_processed`, `processing_time_seconds`, `started_at`, `completed_at`, and `aligned_words`

### Requirement: Aligned words link back to transcript segments

Each aligned word SHALL carry an index identifying which transcript segment it belongs to, so downstream consumers can reconstruct segment-to-word membership without relying on timestamp overlap heuristics.

#### Scenario: Reconstructing segment membership

- **WHEN** a consumer reads an `AlignmentResult` and a `TranscriptionResult` from the same pipeline run
- **THEN** every aligned word's `segment_index` value is a valid index into `TranscriptionResult.segments`, and iterating aligned words grouped by `segment_index` reproduces the word order within each segment
