## ADDED Requirements

### Requirement: Transcription stage is ASR-only and backend-agnostic

The transcription stage SHALL accept an audio file and return a `TranscriptionResult` containing rough segments (`text`, `start`, `end`), a detected language code, and processing provenance. The stage SHALL NOT perform forced alignment or word-level timestamp refinement. The stage's public interface SHALL be independent of any particular ASR library so that backends can be swapped without changing the stage's contract.

#### Scenario: Transcribing a short clip with the default backend

- **WHEN** the stage is called with a 10-second WAV file on `cuda:0`
- **THEN** it returns a `TranscriptionResult` with one or more segments, a non-empty `language` field, and a non-zero `processing_time_seconds`

#### Scenario: Transcription result has no per-word timestamps

- **WHEN** a caller serializes a `TranscriptionResult` to JSON
- **THEN** `TranscriptSegment` entries do not contain a `words` list (that data lives in `AlignmentResult`)

### Requirement: Transcription result includes detected language code

The stage SHALL emit a `language` field on `TranscriptionResult` containing either the language code passed in by the caller or the code detected by the ASR backend when the caller requested auto-detection. The language code SHALL be in a form the forced-alignment stage can consume directly (e.g., BCP-47 or the MMS-FA supported set).

#### Scenario: Auto-detecting language

- **WHEN** the stage is called with `language="auto"` on an English clip
- **THEN** the resulting `TranscriptionResult.language` is `"en"`

#### Scenario: Passing through an explicit language

- **WHEN** the stage is called with `language="es"`
- **THEN** the resulting `TranscriptionResult.language` is `"es"`

### Requirement: Transcription stage owns its model load and unload lifecycle

The transcription stage SHALL expose `load_model` and `unload_model` entry points. After `unload_model`, the ASR model SHALL no longer hold GPU memory, matching the VRAM baseline observed before `load_model`.

#### Scenario: Unload releases VRAM

- **WHEN** the stage loads the ASR model, runs one transcription, and unloads the model
- **THEN** `torch.cuda.memory_allocated(device)` returns within 50 MB of the pre-load baseline

### Requirement: Transcription stage is pluggable via a documented backend seam

The transcription module SHALL expose a single function-level seam (initially wrapping faster-whisper) that future ASR backends can replace. The seam's inputs and outputs SHALL match the stage contract (audio path in, rough segments + language out) so that swapping backends does not require touching the pipeline, aligner, or CLI.

#### Scenario: Replacing the default backend

- **WHEN** a second backend is introduced in a follow-up change
- **THEN** the change consists of a new backend module implementing the seam and a configuration flag; no edits to `pipeline.py`, `aligner.py`, or the public `TranscriptionResult` shape are required
