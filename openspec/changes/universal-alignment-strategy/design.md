## Context

Audio Refinery's transcription stage is implemented in `src/transcriber.py` as a thin wrapper around WhisperX. A single `transcribe()` call does three things at once: (1) runs Whisper large-v3 via faster-whisper/ctranslate2 to get rough segments, (2) loads a per-language Wav2Vec2 model via `whisperx.load_align_model()` and runs `whisperx.align()` to add word-level timestamps, and (3) optionally calls `whisperx.assign_word_speakers()` to merge pyannote diarization labels onto words. The pipeline pre-loads the WhisperX model once per run in `pipeline.py::_load_whisperx_model()` and passes it through.

This design is constrained by WhisperX's release cadence and API. The project is pinned to a specific commit (`741ab9a2a8a1076c171e785363b23c55a91ceff1`) because upstream tags either lacked `device_index` or broke against our pinned `torch==2.1.2+cu121`. Upgrading ASR backends (e.g., to Cohere Transcribe or a newer Whisper variant) means giving up Wav2Vec2 alignment, because alignment is bolted onto WhisperX's internal data shapes. The result: the project cannot benefit from ASR improvements without losing downstream word-timestamp precision that RAG indexing, video editing, and word-level diarization merge depend on.

Separately, VRAM pressure is already real on 24 GB consumer GPUs. `pipeline.py` currently keeps WhisperX + align model resident for the whole loop, and pyannote is loaded per file. Adding another acoustic model (MMS-FA) on top without a strict lifecycle would OOM.

**Current state snapshot** (as of 2026-04-15):
- `src/transcriber.py::transcribe()` — does ASR + align + speaker-merge in one function.
- `src/pipeline.py::_load_whisperx_model()` — pre-loads WhisperX model for the batch.
- `src/models/transcription.py::TranscriptSegment.words` — where word-level timestamps live today.
- `src/models/transcription.py::TranscriptionResult.alignment_fallback: bool` — tracks when Wav2Vec2 align fails and raw Whisper timestamps are used as a fallback.
- No `openspec/specs/` entries yet; this is a greenfield spec effort.

## Goals / Non-Goals

**Goals:**
- Decouple ASR from word-level alignment so transcription backends become hot-swappable.
- Produce word-level timestamps of equal-or-better precision than the current WhisperX path using `torchaudio.pipelines.MMS_FA`.
- Keep runs OOM-free across all GPU sizes by implementing a hybrid VRAM strategy (co-resident or sequential) with upfront preflight validation.
- Preserve the existing downstream contract: diarization-aware transcripts with per-word speakers and per-segment sentiment still come out of the pipeline.
- Make the aligner language-aware via an explicit `language_code` parameter, so future per-language acoustic models are a config swap, not a code change.

**Non-Goals:**
- Shipping a Cohere Transcribe backend in this change. This change lands the decoupling and the torchaudio alignment; Cohere is a follow-up that plugs into the new transcription interface.
- Supporting ASR backends that don't emit rough segment timestamps. If a backend only emits a monolithic block of text, the pipeline falls back to VAD-based chunking — that fallback ships later.
- Changing the Demucs or pyannote stages. Their interfaces stay the same.
- Rewriting sentiment analysis. It already consumes `TranscriptSegment` and will continue to; the segment shape changes minimally (see Decisions).
- Removing WhisperX from the repo in this change. It stays installable as a legacy backend flag for one release cycle.

## Decisions

### Decision 1: Split `transcriber.py` into an ASR-only wrapper; add a new `aligner.py` stage

The existing `transcribe()` function becomes ASR-only. It runs faster-whisper directly (WhisperX's underlying engine) via a thin wrapper and returns rough segments plus a detected language code — no `words`, no align, no speaker merge. Alignment moves to a new `src/aligner.py` module that owns `torchaudio.pipelines.MMS_FA`. Speaker merge moves into either `aligner.py` or a small new `src/merger.py` (see Open Questions).

**Alternatives considered:**
- *Keep WhisperX as the only backend and wrap MMS-FA inside it* — rejected. The whole point is to decouple. Wrapping MMS-FA inside WhisperX reproduces the coupling we're removing.
- *Call faster-whisper via whisperx's `load_model()` and just skip the align step* — rejected. This leaves us married to WhisperX's model-loading lifecycle and its specific transformers/torch pin. Direct faster-whisper gives us a cleaner break.
- *Make the transcription stage pluggable via a `TranscriptionBackend` abstract class from day one* — deferred. The first backend is faster-whisper. The second backend (Cohere or cloud) is what justifies the abstraction; building it before we know the second backend's shape invites wrong abstractions. This change lays the seam; the interface hardens in the Cohere follow-up.

### Decision 2: Use `torchaudio.pipelines.MMS_FA` for forced alignment

MMS-FA ships with torchaudio ≥ 2.1, is multilingual out of the box, and its acoustic model weights are small enough to load/unload cheaply. It exposes the trellis directly, which is what we need for chunked alignment.

**Alternatives considered:**
- *Keep `whisperx.align()` but call it standalone* — possible, but it pulls in WhisperX's dependency chain (pytorch-lightning, lightning, etc.) and still couples us to their language-model mapping table. Wins nothing architecturally.
- *Use `ctc-forced-aligner` from HuggingFace* — viable alternative, but not yet in torchaudio and requires an extra HF download path. Revisit if MMS-FA quality disappoints on non-English content.
- *Skip forced alignment entirely and trust ASR-reported word timings* — rejected. faster-whisper's `word_timestamps=True` path is noisier than Wav2Vec2 alignment, especially near segment boundaries, and that's exactly what downstream consumers care about.

### Decision 3: Chunked alignment driven by ASR segment boundaries, capped at 30 s

Trellis memory is O(audio_frames × tokens). For a 10-minute file, that's a multi-GB matrix; for a 1-hour podcast, it's impossible. The aligner must chunk before running.

**Chunking strategy:**
- Primary: use the ASR stage's rough segment boundaries. For each segment (or contiguous group of segments), pack until the audio window reaches ~25 s, then emit a chunk. Cap hard at 30 s.
- Fallback (when a single ASR segment exceeds 30 s — rare, happens on music or silence runs): split on the nearest silence using a lightweight energy-based VAD, or fail the chunk with a clear error and mark it in `AlignmentResult.fallback_reason`.
- Each chunk is aligned independently; results are offset back into the global timeline using the chunk's start time.

**Alternatives considered:**
- *Fixed 30 s stride, ignoring ASR boundaries* — simpler, but splits mid-word and produces worse alignment at chunk edges.
- *VAD-driven chunking independent of ASR* — adds a second model to the pipeline (silero-vad or similar) just for chunking. Unnecessary when ASR already gives us rough boundaries for free.
- *Stream alignment with a sliding window* — premature complexity. Chunked alignment is good enough for batch processing, which is this tool's core use case.

### Decision 4: Hybrid VRAM strategy — co-resident by default, sequential by flag or auto-fallback

Each stage module exposes:

```
load_model(device, ...) -> ModelHandle
unload_model(handle) -> None   # calls del, gc.collect(), torch.cuda.empty_cache()
```

Two VRAM strategies are available via `--vram-strategy {co-resident, sequential}`:

**Co-resident (default):** All in-process GPU models (pyannote, faster-whisper, MMS-FA) are loaded once before the batch loop and stay resident for the entire run, matching the current pipeline's pre-load pattern. This is the fastest option because model load time is paid once. Demucs already runs as a subprocess (separate process, separate VRAM), and sentiment runs on CPU, so neither competes for in-process GPU memory.

Typical co-resident VRAM budget:
```
  pyannote diarization       ~1,500 MiB
  faster-whisper large-v3    ~5,800 MiB
  MMS-FA alignment             ~400 MiB
  ────────────────────────────────────────
  Total co-resident           ~7,700 MiB   (fits comfortably on 24 GB)
```

**Sequential:** Each stage loads its model, processes all files (or the current file), unloads the model, then the next stage loads. Peak VRAM equals the single largest stage. Required when co-resident doesn't fit — e.g., when a future heavyweight stage (emotion/SER, CLAP, or a large cloud-class ASR model) is enabled alongside existing stages.

Per-file sequential overhead: ~10.5 s of model loading per file (pyannote ~3 s + faster-whisper ~6 s + MMS-FA ~1.5 s). For a 10-file batch, this adds ~105 s vs. ~11 s for co-resident. **In sequential mode, default to stage-batched execution** (all files through one stage before the next) to amortize load cost across files. Per-file interleaved mode stays available behind a flag for small runs.

**Auto-fallback:** If the user doesn't specify a strategy, the pipeline runs the VRAM preflight check (see Decision 7). If co-resident fits, it's used. If it doesn't, the pipeline automatically falls back to sequential and emits a warning explaining why.

**Alternatives considered:**
- *Strict sequential only (original proposal)* — rejected after analysis. MMS-FA is only ~400 MiB; co-resident pyannote + faster-whisper + MMS-FA totals ~7.7 GB, well within 24 GB consumer cards. Forcing sequential load/unload on every run adds ~10 s of dead time per file for no VRAM benefit on most hardware. The strict approach was designed for a hypothetical future where all models are large; the hybrid defers that cost until it's actually needed.
- *Co-resident only (no sequential option)* — rejected. Users on smaller GPUs (12 GB), users adding future heavyweight stages, and cloud deployments with variable instance types all need the escape hatch. Different transcription models also vary widely in VRAM (small ~1.2 GB vs. large-v3 ~5.8 GB), making a one-size-fits-all assumption untenable.
- *Swap models via CPU offload (`accelerate`-style)* — adds a heavy dependency and doesn't actually free VRAM for the next model; it just moves weights to CPU RAM while the previous model's allocator fragments persist.

### Decision 5: New `AlignmentResult` model; `TranscriptionResult` loses `words`

```python
# src/models/transcription.py (changed)
class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    # words: REMOVED — now lives on AlignmentResult
    speaker: str | None = None
    sentiment: SegmentSentiment | None = None

# src/models/alignment.py (new)
class AlignedWord(BaseModel):
    word: str
    start: float
    end: float
    score: float
    speaker: str | None = None  # populated by the merge step

class AlignmentResult(BaseModel):
    input_file: Path
    transcription_file: Path  # provenance pointer back to the ASR output
    language: str
    aligned_words: list[AlignedWord]
    segment_index: list[int]  # aligned_words[i] belongs to TranscriptSegment[segment_index[i]]
    acoustic_model: str  # e.g. "torchaudio.pipelines.MMS_FA"
    device: str
    chunks_processed: int
    fallback_reason: str | None = None  # filled when a chunk fell back or failed
    processing_time_seconds: float
    started_at: datetime
    completed_at: datetime
```

`segment_index` is the explicit back-pointer that lets downstream consumers reconstruct "which words belong to which segment" without relying on timestamp overlap math. Sentiment analysis and the RAG merge step both need this.

**Alternatives considered:**
- *Inline aligned words back into `TranscriptSegment.words` after alignment* — easier migration (downstream code doesn't change), but hides the decoupling in the data model and re-couples segments to word-level data. Rejected because it defeats the point.
- *Ditch segments entirely and make aligned words the only output* — too aggressive. Segments are useful for RAG chunking and captioning; keeping them as the coarse unit and treating aligned words as the fine unit matches how consumers think.

### Decision 6: Text normalization is a small, deterministic module — no ML

`src/text_normalizer.py` exposes one function: `normalize_for_alignment(text: str, language: str) -> str`. It expands numbers (`100` → `"one hundred"`), currency (`$100` → `"one hundred dollars"`), lowercases, strips punctuation and symbols except intra-word apostrophes, and collapses whitespace. Use the `num2words` library (pure Python, multilingual) for number expansion. No tokenizer, no model.

**Alternatives considered:**
- *Nemo/ESPnet text normalizers* — massive overkill for the symbol set that actually breaks MMS-FA.
- *Let the aligner's tokenizer handle everything* — it doesn't. MMS-FA crashes or produces garbage on `$`, `%`, raw digits, emoji, etc. Cleaning upstream is the only reliable path.
- *Run normalization inside the aligner module* — rejected for testability. A pure text-in/text-out module is trivially unit-testable; bundling it inside the aligner forces the aligner's fixtures to load a torchaudio bundle just to test string transforms.

### Decision 7: VRAM budget registry (`model_budgets.toml`) and preflight validation

A new `src/model_budgets.toml` file maps model identifiers to their expected VRAM footprint (weights + typical peak activations, in MiB). This is a separate file from `gpu_tflops.toml` — one maps GPU names to performance, the other maps model names to VRAM cost.

```toml
# src/model_budgets.toml
[vram_mib]
"pyannote/speaker-diarization-3.1"  = 1500
"faster-whisper/large-v3"           = 5800
"faster-whisper/distil-large-v3"    = 3500
"faster-whisper/medium"             = 2900
"faster-whisper/small"              = 1200
"torchaudio/MMS-FA"                 = 400
```

A new `src/vram_preflight.py` module exposes `validate_vram_budget()`:

1. Queries GPU(s) via existing `gpu_utils.query_gpu_info()` to get total VRAM.
2. Queries active processes via `gpu_utils.query_compute_processes()` to determine already-used VRAM.
3. Subtracts a configurable headroom (default 512 MiB) for CUDA context and fragmentation.
4. Looks up the VRAM cost of each enabled stage's model from `model_budgets.toml`.
5. For co-resident mode: sums all enabled stages. For sequential mode: takes the max.
6. Compares projected budget against available VRAM.
7. Returns a structured `VramBudget` result with per-model breakdowns, the verdict (fits / doesn't fit), and actionable suggestions if it fails (switch strategy, use smaller model, free GPU memory, switch GPU).

For multi-GPU parallel runs, the preflight checks the **smallest** GPU across all workers, since workers run identical configurations. This keeps the validation simple and prevents the weakest link from OOM-ing mid-batch.

**Alternatives considered:**
- *Extend `gpu_tflops.toml` with VRAM data* — rejected. Separate concerns: TFLOPS drives GPU ranking, VRAM budget drives pipeline planning. Mixing them makes both files harder to maintain.
- *Query VRAM at runtime via `torch.cuda.mem_get_info()`* — used as a supplemental check, but not the primary source. The TOML registry gives budget estimates *before* loading anything, which is the whole point of preflight. Runtime queries confirm the estimate after the first stage loads.
- *Skip preflight and just catch OOM at runtime* — rejected. An OOM 20 minutes into a 50-file batch is a terrible user experience. Catching it before any work begins is the correct UX.

### Decision 8: Interactive run planner (`audio-refinery plan`)

A new `plan` command provides an interactive CLI interface for configuring, validating, and optionally saving pipeline runs. This addresses the growing complexity of flag combinations as stages are added (separation, diarization, transcription, alignment, sentiment, and future emotion/events stages).

**The planner workflow:**
1. Discovers source files and queries available GPUs.
2. Renders a Rich panel showing: file count and total audio duration, GPU specs, enabled pipeline stages with model choices, a VRAM budget bar chart comparing projected usage against available capacity, and an estimated run time based on historical RTF data.
3. Lets the user toggle stages, switch models, and change VRAM strategy. Each change immediately re-renders the budget and estimate.
4. On confirmation, either launches the pipeline directly or saves the configuration as a **run profile** — a small TOML file that `pipeline --plan profile.toml` replays non-interactively.

**Run profile format:**
```toml
# saved by `audio-refinery plan --save my-batch.toml`
[source]
dir = "/audio/test/extracted"

[pipeline]
stages = ["separate", "diarize", "transcribe", "align", "sentiment"]
vram_strategy = "co-resident"
mode = "stage-batched"

[transcription]
model = "large-v3"
compute_type = "float16"
batch_size = 16
language = "en"

[alignment]
backend = "mms-fa"

[devices]
primary = "cuda:0"
```

**Why an interactive planner instead of more flags:**
- Flag combinatorics grow multiplicatively. Today with 4 configurable stages × 3 models × 2 VRAM strategies = ~24 meaningful configs. With emotion + CLAP: 6 stages × 3 models × 2 strategies = ~384. No one reasons about this from `--help`.
- The planner shows *interactions* between choices. Picking `--model large-v3` + `--emotion` might OOM on a 12 GB card; the planner visualizes this immediately and suggests alternatives.
- Run profiles make scheduled/CI runs reproducible without memorizing flags.

**Implementation notes:**
- Built on Rich (already in the stack) — panels, tables, bars. No new dependency for basic interaction (`rich.prompt.Prompt` for simple choices).
- The `plan` command is a standalone Click subcommand alongside `pipeline`, `transcribe`, etc.
- `pipeline --plan <file>` is syntactic sugar that reads the profile TOML and maps it to the existing pipeline kwargs. No separate execution path.

**Alternatives considered:**
- *YAML/JSON config file only (no interactive mode)* — misses the core value: immediate feedback on VRAM budget as you adjust choices. A config file without a planner is just another thing to edit blindly.
- *TUI framework (textual, urwid)* — too heavy for what's needed. Rich prompts + panels + re-renders are sufficient. Revisit if the planner grows into a full dashboard (unlikely for a batch processing tool).
- *Web-based config UI* — wrong modality for a CLI tool that runs on GPU servers often accessed via SSH.

## Risks / Trade-offs

- **[MMS-FA quality regression on English vs. whisperx's English Wav2Vec2]** → Mitigation: ship an integration test that runs both backends on a known-good clip and compares word boundary deltas. If regression is material, keep WhisperX behind a `--aligner whisperx` flag for one release cycle.
- **[Sequential mode load overhead slows batch runs]** → Mitigation: co-resident is the default. Sequential mode defaults to stage-batched execution to amortize load cost. The preflight check warns users about the time impact before the run starts. Measure wall-clock on a representative batch in both modes before merging and document the trade-off in `docs/DEVELOPMENT.md`.
- **[Chunk boundaries cut words when an ASR segment's audio exceeds 30 s and no silence is present]** → Mitigation: prefer silence-based fallback; if no silence exists in a 30 s window (e.g., continuous music under speech), emit a clear error in `fallback_reason`, use raw ASR word timings for that chunk, and log a warning. Rare in the target domain (podcasts, interviews, dialog).
- **[num2words produces odd output for contextual currencies]** (e.g., "$1.5M" → unclear expansion) → Mitigation: pre-regex common cases (`\$(\d+(?:\.\d+)?)([kKmMbB])?`) into explicit text before calling num2words, and fall back to stripping the symbol if expansion fails. Log failures for corpus analysis.
- **[Breaking change to `TranscriptionResult.segments[].words`]** → Mitigation: bump the minor version (0.2.0), add a migration note to `CHANGELOG.md`, and provide a compat helper `load_transcription_with_words(path) -> list[TranscriptSegment]` that reads both the transcription and alignment JSON and re-stitches words onto segments for callers that want the old shape.
- **[Legacy whisperx backend flag adds a maintenance surface]** → Mitigation: scope it as "one release cycle only." Delete in the next change after this one lands and downstream callers confirm migration.
- **[MMS-FA weights download on first use may fail behind corporate proxies]** → Mitigation: document the HuggingFace cache path in `docs/DEVELOPMENT.md` and support `HF_HOME` / `TRANSFORMERS_CACHE` env vars (already honored by torchaudio).
- **[VRAM budget estimates are approximate]** → Mitigation: `model_budgets.toml` values include typical peak activation overhead (not just weight size), measured empirically at default batch sizes. The 512 MiB headroom margin absorbs minor variance. For edge cases, the planner shows both the projected budget and actual available VRAM, so users can judge the margin themselves. Runtime `torch.cuda.mem_get_info()` provides a secondary safety net.
- **[Interactive planner assumes a TTY]** → Mitigation: `audio-refinery plan` detects non-interactive mode (`not sys.stdin.isatty()`) and exits with a message pointing to `--plan profile.toml` for headless use. The preflight validation also runs automatically when `pipeline` starts, so the planner is not required for safety — it's an ergonomic layer on top.

## Migration Plan

1. **Land the new modules and models in parallel with existing code** — `aligner.py`, `text_normalizer.py`, `models/alignment.py`, and the refactored `transcriber.py` ship alongside the old code paths. The old `transcriber.transcribe()` continues to work (marked deprecated).
2. **Add new CLI subcommands** — `audio-refinery align` runs the aligner standalone against a transcription JSON. `audio-refinery plan` provides interactive run configuration and validation. `audio-refinery pipeline` gains `--aligner {mms-fa,whisperx,none}`, `--vram-strategy {co-resident,sequential}`, and `--plan <profile.toml>` flags; default aligner stays `whisperx` for one release to avoid surprising users mid-upgrade.
3. **Flip the default** — in the follow-up 0.3.0 release, default `--aligner` becomes `mms-fa` and `whisperx` prints a deprecation warning.
4. **Remove WhisperX** — in the release after that, delete the `whisperx` branch from the pipeline and drop the optional-dependency entry from `pyproject.toml`.

**Rollback strategy:** Each intermediate release keeps the old path reachable via flag. A user who hits a regression can pin the previous audio-refinery version *or* pass `--aligner whisperx` until the issue is triaged.

## Open Questions

1. Should speaker merge live in `aligner.py` or in a new `merger.py` module? Leaning toward `merger.py` because the merge is independent of the alignment acoustic model — it's just "given aligned words and diarization segments, assign speakers." Keeping them separate matches the one-responsibility-per-module convention.
2. ~~Does the stage-batched pipeline mode reuse the existing `pipeline.py` structure or warrant a new `pipeline_staged.py`?~~ **Resolved:** extend `pipeline.py` with a mode flag. Stage-batched is the default for sequential VRAM strategy; per-file interleaved is the default for co-resident. Both paths live in the same module.
3. What does the CLI report for alignment metrics? Current transcription report shows words/RTF/VRAM. Alignment will show chunks processed, mean chunk duration, fallback count. Need to confirm the exact fields with the CLI report's existing Rich table layout.
4. Can we ship `faster-whisper` as a direct dependency in `pyproject.toml` under the main `dependencies` (not under the `conflicting` optional), or does it still hit the same torch version conflict that forced whisperx into optional deps? Needs a clean `uv sync` test.
5. Should the `plan` command support editing all pipeline flags (batch_size, compute_type, segment) or only the high-impact ones (stages, model, VRAM strategy, device)? Leaning toward high-impact only to keep the TUI simple; power users use flags directly.
6. How should `model_budgets.toml` handle compute_type variations? `float16` vs. `int8` halves the weight VRAM. Options: (a) separate entries per compute type, (b) a single entry with a multiplier formula, (c) always budget for float16 and treat int8 as extra headroom. Leaning toward (a) for accuracy.
