## Why

Audio Refinery is coupled to WhisperX, which bundles transcription and Wav2Vec2 alignment behind a single API. This prevents us from swapping in better ASR backends (Cohere Transcribe, cloud APIs, newer open-source models) without losing the millisecond-level word timestamps that downstream consumers (RAG databases, semantic video editing, diarization merging) depend on. Decoupling the transcription engine from the word-level aligner turns Audio Refinery into a model-agnostic audio processing factory where ASR backends are hot-swappable while alignment quality stays constant.

## What Changes

- Add a new **forced alignment** stage built on `torchaudio.pipelines.MMS_FA` (Wav2Vec2 acoustic model) that accepts a sanitized text string plus audio and emits precise word-level start/end times.
- Add a new **text normalization** step that sanitizes ASR output before alignment: expand numbers and currency ("$100" → "one hundred dollars"), strip punctuation and symbols that crash phoneme models, and lowercase.
- Refactor the **transcription** stage into an ASR-only, model-agnostic interface. The stage produces rough segments `[{start, end, text}]` and a detected language code; it no longer emits word-level timestamps directly.
- Introduce a **chunked alignment** strategy. Because trellis pathfinding memory scales quadratically with audio length, the aligner must chunk audio to ≤30 s windows based on ASR segment boundaries (or VAD) before running MMS-FA.
- Introduce a **hybrid model lifecycle** with two VRAM strategies: **co-resident** (all in-process models loaded simultaneously — fastest, requires sufficient VRAM) and **sequential** (load/unload each stage one at a time — fits any GPU). Each stage exposes `load_model()` / `unload_model()` with `gc.collect()` + `torch.cuda.empty_cache()`. Default strategy is co-resident; the system automatically falls back to sequential if the projected VRAM budget exceeds available capacity.
- Add a **VRAM budget preflight** that runs before the pipeline starts: queries available GPU memory, looks up projected VRAM per model from a new `model_budgets.toml` registry, and validates that the selected strategy fits. Blocks the run with an actionable error if it doesn't, suggesting alternatives (smaller model, different strategy, free GPU memory).
- Add an **interactive run planner** via a new `audio-refinery plan` CLI command. Discovers files, queries GPUs, renders the pipeline configuration with a VRAM budget bar chart, and lets the user adjust stages/models/strategy before committing. Optionally saves validated configurations as reusable **run profiles** (TOML) for non-interactive replay via `pipeline --plan profile.toml`.
- Plumb the ASR-detected `language_code` through to the aligner so future language-specific acoustic models can be swapped in dynamically.
- **BREAKING**: `TranscriptionResult` and the transcription stage contract change shape — word-level timestamps move out of transcription output into a new `AlignmentResult` produced by the forced-alignment stage. Downstream consumers (diarization merge, sentiment, CLI reports) read from the alignment output instead.
- **BREAKING**: WhisperX is removed from the critical path. The `transcriber` module either wraps faster-whisper directly (ASR-only) or exposes a pluggable backend interface; the `whisperx.align` call is deleted.

## Capabilities

### New Capabilities

- `forced-alignment`: Chunked word-level forced alignment of arbitrary text against audio using torchaudio MMS-FA (Wav2Vec2). Accepts a language code, produces per-word start/end/score arrays, and owns its own model load/unload lifecycle.
- `text-normalization`: Deterministic text sanitization that prepares ASR output for phoneme-based alignment (number/currency expansion, symbol stripping, casing).
- `transcription`: Model-agnostic ASR stage that emits rough segments and a detected language code. Establishes the contract that lets transcription backends be swapped without touching alignment.
- `model-lifecycle`: Hybrid VRAM strategy (co-resident vs. sequential) with per-stage `load_model`/`unload_model` hooks, a `model_budgets.toml` VRAM registry, an upfront preflight validation that blocks runs projected to OOM, and an interactive `audio-refinery plan` command for configuring and validating pipeline runs before execution.

### Modified Capabilities

<!-- No existing specs in openspec/specs/ — the four capabilities above are all net-new. -->

## Impact

- **Code**: `src/transcriber.py` (refactored into ASR-only wrapper), `src/pipeline.py` (new stage order + hybrid VRAM lifecycle), `src/cli.py` (new `align` and `plan` subcommands, updated `transcribe` and `pipeline` commands), `src/models/transcription.py` (shape change), new `src/aligner.py`, `src/text_normalizer.py`, `src/vram_preflight.py` modules, new `src/models/alignment.py` Pydantic model, new `src/model_budgets.toml` registry.
- **APIs**: `TranscriptionResult.segments[].words` removed; new `AlignmentResult` model with `aligned_words: list[AlignedWord]` becomes the source of truth for word timing. Sentiment and diarization merging read from the alignment output. New `VramBudget` dataclass and `validate_vram_budget()` function for preflight checks.
- **Dependencies**: `torchaudio>=2.1.2` (already pinned via torch 2.1.2) gains the MMS-FA bundle; HuggingFace model weights for `torchaudio.pipelines.MMS_FA` download on first use. WhisperX drops from the critical path but stays available as an optional legacy backend behind a feature flag for one release cycle to ease migration. `faster-whisper` becomes a direct dependency (previously pulled transitively by whisperx).
- **Tests**: New unit tests for `text_normalizer`, `aligner` (with mocked torchaudio bundle), VRAM preflight validation, and pipeline reordering; new integration test that runs the full decoupled pipeline end-to-end on a short clip with GPU marker.
- **Performance**: Co-resident mode preserves current batch throughput (models loaded once). Sequential mode trades load/unload overhead for reduced VRAM footprint. Preflight validation prevents OOM failures before work begins.
- **Docs**: `README.md`, `CLAUDE.md`, and `docs/DEVELOPMENT.md` updated to describe the new stage order, model-agnostic transcription contract, VRAM strategies, and the `plan` command.
