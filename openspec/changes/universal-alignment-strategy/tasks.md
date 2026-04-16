## 1. Scaffolding & Dependencies

- [ ] 1.1 Add `num2words` to `pyproject.toml` main dependencies and re-sync with `uv`
- [ ] 1.2 Add `faster-whisper` to `pyproject.toml` main dependencies; verify `uv sync` succeeds without reintroducing the WhisperX/torch 2.1.2 conflict (if conflict persists, keep under `[project.optional-dependencies].conflicting` and update `make install-whisperx` accordingly)
- [ ] 1.3 Confirm `torchaudio>=2.1.2` exposes `torchaudio.pipelines.MMS_FA` in the CUDA 12.1 wheel; run a one-liner smoke test from `make install-torch-cuda`
- [ ] 1.4 Create empty module skeletons: `src/text_normalizer.py`, `src/aligner.py`, `src/merger.py`, `src/vram_preflight.py`, `src/models/alignment.py`
- [ ] 1.5 Update `CLAUDE.md` and `docs/DEVELOPMENT.md` project-structure sections to list the new modules

## 2. Data Models

- [ ] 2.1 Add `AlignedWord` and `AlignmentResult` Pydantic models in `src/models/alignment.py` with fields from design §Decision 5
- [ ] 2.2 Remove `words: list[WordSegment]` from `TranscriptSegment` in `src/models/transcription.py`
- [ ] 2.3 Remove `alignment_fallback: bool` from `TranscriptionResult` (fallback now lives on `AlignmentResult.fallback_reason`)
- [ ] 2.4 Export `AlignedWord`, `AlignmentResult` from `src/models/__init__.py`
- [ ] 2.5 Add a compat helper `load_transcription_with_words(transcription_path, alignment_path) -> list[TranscriptSegment]` that reads both JSONs and returns segments with word lists stitched back on (for callers that want the old shape)
- [ ] 2.6 Add unit tests under `tests/models/` for `AlignmentResult` round-trip JSON serialization and `load_transcription_with_words` stitching

## 3. Text Normalization Stage

- [ ] 3.1 Implement `normalize_for_alignment(text: str, language: str) -> str` in `src/text_normalizer.py` covering: lowercase, whitespace collapse, punctuation strip (preserving intra-word apostrophes), number expansion via `num2words`, currency pre-regex for `$`/`€`/`£` amounts including `k/M/B` suffixes
- [ ] 3.2 Implement graceful failure: on `num2words` exception for a token, strip symbols, log a warning with the offending token, and continue
- [ ] 3.3 Add `tests/test_text_normalizer.py` covering every scenario in `specs/text-normalization/spec.md` (English simple, currency, bare numbers, Spanish numbers, emoji strip, apostrophe preservation, unparseable currency fallback, determinism)

## 4. Transcription Stage Refactor

- [ ] 4.1 Rewrite `src/transcriber.py::transcribe()` to call `faster-whisper` directly (no `whisperx.align`, no `whisperx.assign_word_speakers`). Output rough segments only (`text`, `start`, `end`) plus detected language
- [ ] 4.2 Add `load_model(device, compute_type, batch_size, language, model)` and `unload_model(handle)` entry points. `unload_model` SHALL `del handle`, `gc.collect()`, and `torch.cuda.empty_cache()`
- [ ] 4.3 Replace `_parse_whisperx_device` with a faster-whisper-appropriate device parser (faster-whisper also splits `cuda:N` into `device="cuda", device_index=N`)
- [ ] 4.4 Update `TranscriptionResult` construction in `transcriber.py` to drop `words`, `alignment_fallback`
- [ ] 4.5 Remove the speaker-merge branch (`whisperx.assign_word_speakers`) from `transcriber.py`; speaker assignment moves to `merger.py`
- [ ] 4.6 Define a minimal backend seam: a module-level `Backend` protocol/type-alias or single `_run_asr(audio, model_handle, ...) -> (segments, language)` function that future backends can replace without touching the stage's public interface
- [ ] 4.7 Rewrite `tests/test_transcriber.py` to mock faster-whisper, assert no `words` on output, and cover both auto-detect and explicit-language paths
- [ ] 4.8 Verify transcription VRAM release: add a test that loads, transcribes, unloads, and asserts `torch.cuda.memory_allocated` is within 50 MB of baseline (skip under `not has_cuda`)

## 5. Forced Alignment Stage

- [ ] 5.1 Implement `load_model(device, language_code) -> AlignerHandle` in `src/aligner.py` that fetches `torchaudio.pipelines.MMS_FA` and prepares its tokenizer for the given language
- [ ] 5.2 Implement `unload_model(handle)` mirroring the transcription unload pattern
- [ ] 5.3 Implement `align(audio_path, transcription_result, normalizer, handle) -> AlignmentResult`: iterate ASR segments, pack into ≤30 s chunks, run the trellis per chunk, offset word timings back into the global timeline, populate `segment_index`
- [ ] 5.4 Implement the silence-based fallback for oversize segments: on an ASR segment exceeding 30 s, split on the lowest-energy window within the segment (simple numpy RMS, no new dependency); if no viable split, populate `fallback_reason` with `"no_silence_in_segment"` and use the ASR's raw start/end for that span
- [ ] 5.5 Raise a clear `AlignerError` for unsupported language codes (pre-check against `MMS_FA.get_labels()` or equivalent) — do NOT silently fall back to a different language
- [ ] 5.6 Add `tests/test_aligner.py` covering: mocked MMS-FA bundle returning synthetic trellis output, chunk boundary math (ensure offsets applied), `segment_index` correctness, language-not-supported error, oversize-segment fallback
- [ ] 5.7 Add an integration test under `tests/test_integration.py` marked `@pytest.mark.integration` that runs the real MMS-FA pipeline on a short recorded clip and asserts monotonic word start times across the file

## 6. Speaker Merge Module

- [ ] 6.1 Implement `merge_speakers(alignment_result, diarization_result) -> AlignmentResult` in `src/merger.py` — iterate aligned words, find the diarization segment whose `[start, end]` contains each word's midpoint, assign `speaker` field; do not require a model
- [ ] 6.2 Handle edge cases: word falls in a gap between diarization segments (nearest neighbor), word spans two speakers (pick the speaker covering more of the word's duration)
- [ ] 6.3 Add `tests/test_merger.py` covering the midpoint-inside case, the gap case, and the boundary-spanning case with deterministic fixture data

## 7. VRAM Budget Registry & Preflight Validation

- [ ] 7.1 Create `src/model_budgets.toml` with VRAM estimates (MiB) for: `pyannote/speaker-diarization-3.1`, `faster-whisper/large-v3`, `faster-whisper/distil-large-v3`, `faster-whisper/medium`, `faster-whisper/small`, `torchaudio/MMS-FA`. Values include weight size + typical peak activation overhead at default batch sizes
- [ ] 7.2 Implement `src/vram_preflight.py` with: `load_model_budgets()` (reads the TOML), `VramBudget` dataclass (per-model breakdown, total projected, available, verdict, suggestions), and `validate_vram_budget(stages, models, device, strategy, headroom_mib=512) -> VramBudget`
- [ ] 7.3 Preflight logic for co-resident: sum all enabled stage budgets. For sequential: take max of any single stage. Compare against `gpu_utils.query_gpu_info(device).vram_mib - used_mib - headroom`
- [ ] 7.4 Generate actionable suggestions on failure: switch to sequential, use a smaller model (list alternatives with their budgets), free GPU memory, switch GPU (if multi-GPU detected)
- [ ] 7.5 Handle unknown models gracefully: log a warning, use a conservative fallback estimate (e.g., 4000 MiB), do not crash
- [ ] 7.6 For multi-GPU parallel runs, validate against the smallest GPU's available VRAM across all target devices
- [ ] 7.7 Add `tests/test_vram_preflight.py` covering: budget passes co-resident, budget blocks co-resident, auto-fallback to sequential, unknown model warning, multi-GPU smallest-GPU logic, headroom subtraction

## 8. Pipeline Integration

- [ ] 8.1 Update `src/pipeline.py` stage order: separate → diarize → transcribe (ASR only) → normalize → align → merge speakers → sentiment. Unload behavior depends on VRAM strategy
- [ ] 8.2 Add `--vram-strategy {co-resident, sequential}` parameter to `run_pipeline()`. Co-resident: load all in-process GPU models (pyannote, ASR, aligner) before the file loop. Sequential: load/unload each stage's model around its processing pass
- [ ] 8.3 Integrate VRAM preflight: call `validate_vram_budget()` before loading any models. On failure with explicit co-resident, block with error. On failure with no explicit strategy, auto-fallback to sequential with a warning
- [ ] 8.4 Replace `_load_whisperx_model` with new `transcriber.load_model()` call; wire pipeline to use the new lifecycle entry points for all stages
- [ ] 8.5 Add a `mode` flag to the pipeline: `per_file` (interleaved) and `stage_batched`. Default: per-file for co-resident, stage-batched for sequential
- [ ] 8.6 Add `AlignmentResult` to `PipelineResult` as a new `alignment: StageResult` field; update `FileOutcome` with alignment-specific metrics (`chunks_processed`, `fallback_count`)
- [ ] 8.7 Update Slack notifier (`src/notifier.py`) and CLI Rich reports to render the new alignment stage stats and VRAM strategy used
- [ ] 8.8 Update `tests/test_pipeline.py` and `tests/test_pipeline_parallel.py` to reflect the new stage order, both VRAM strategies, and both execution modes
- [ ] 8.9 Add a pipeline test for co-resident mode that asserts all models are loaded before the file loop
- [ ] 8.10 Add a pipeline test for sequential mode that asserts peak VRAM per stage reflects only one model at a time

## 9. CLI

- [ ] 9.1 Add `audio-refinery align` subcommand: inputs are `--audio`, `--transcription`, `--output`; options include `--device`, `--language` (override auto)
- [ ] 9.2 Update `audio-refinery transcribe` to drop the `--no-align` flag (moot now) and to stop mentioning alignment in help text
- [ ] 9.3 Update `audio-refinery pipeline` and `pipeline-parallel` to surface `--aligner {mms-fa,whisperx,none}`, `--vram-strategy {co-resident,sequential}`, `--mode {per-file,stage-batched}`, and `--plan <profile.toml>` flags. Default aligner stays `whisperx` for one release cycle
- [ ] 9.4 Implement `--plan <profile.toml>` flag: read TOML, map to pipeline kwargs, run VRAM preflight against current GPU state before proceeding
- [ ] 9.5 Update `tests/test_cli.py` for the new subcommands and flags

## 10. Interactive Run Planner

- [ ] 10.1 Add `audio-refinery plan` Click subcommand with `--source-dir`, `--device`, and `--save <path>` options
- [ ] 10.2 Implement the discovery panel: file count, total audio duration (via `probe_audio_file` on each), GPU name/VRAM from `query_gpu_info()`
- [ ] 10.3 Implement the stage configuration panel: list all pipeline stages with current enable/disable state and model choices; use Rich prompts for toggling stages and selecting models
- [ ] 10.4 Implement the VRAM budget visualization: Rich bar chart showing per-model projected VRAM vs. available, with color-coded pass/fail. Re-renders on every configuration change
- [ ] 10.5 Implement run time estimation: use model-specific RTF constants (bootstrapped from known benchmarks, refined from historical runs) × total audio duration to project wall-clock. Show separately for co-resident vs. sequential
- [ ] 10.6 Implement the run profile save: serialize confirmed configuration to TOML at the `--save` path (or a default like `.audio-refinery-plan.toml`)
- [ ] 10.7 Implement non-TTY detection: `not sys.stdin.isatty()` → exit with message directing to `pipeline --plan`
- [ ] 10.8 Add `tests/test_plan.py` covering: discovery panel output, VRAM budget calculation matches preflight, profile round-trip (save then load), non-TTY exit behavior

## 11. Legacy WhisperX Shim

- [ ] 11.1 Extract the current `transcriber.py` logic into `src/backends/whisperx_legacy.py` as a single callable that reproduces the old end-to-end behavior (ASR + Wav2Vec2 align + speaker merge) and emits an `AlignmentResult` in the new shape
- [ ] 11.2 Wire the `--aligner whisperx` pipeline branch to the legacy shim
- [ ] 11.3 Add a deprecation warning in the legacy branch's startup log; document the removal schedule in `CHANGELOG.md`

## 12. Documentation

- [ ] 12.1 Update `README.md`: pipeline-flow diagram, stage descriptions for the new 6-stage order, VRAM strategy overview, `plan` command usage, updated CLI examples, and new dependency notes (`faster-whisper`, `num2words`, `model_budgets.toml`)
- [ ] 12.2 Update `CLAUDE.md`: "Project Structure" (new modules: `aligner.py`, `text_normalizer.py`, `merger.py`, `vram_preflight.py`, `model_budgets.toml`, `models/alignment.py`), "Pipeline Stages" (new stage order + alignment stage), "CLI" (new `align` and `plan` commands, new flags), "Critical Dependency Notes" (`faster-whisper` as direct dep, WhisperX deprecation), "Architecture" (hybrid VRAM lifecycle)
- [ ] 12.3 Update `docs/DEVELOPMENT.md`: VRAM strategy trade-offs (co-resident vs. sequential), per-file vs. stage-batched execution modes, `model_budgets.toml` maintenance guide (how to measure and add entries), HF cache path for MMS-FA weights (`HF_HOME`/`TRANSFORMERS_CACHE`), `audio-refinery plan` usage and run profile format, updated `make` targets if any change
- [ ] 12.4 Update `docs/ARCHITECTURE.md`: new pipeline stage flow diagram (6 stages), forced alignment stage description, text normalization stage description, speaker merge module, VRAM preflight system, hybrid model lifecycle, `model_budgets.toml` registry role
- [ ] 12.5 Update `docs/USE_CASES.md`: add alignment-related use cases (RAG-ready word-level timestamps from any ASR backend, model-agnostic transcription), VRAM-constrained GPU use case (sequential mode on 12 GB cards), run planning use case
- [ ] 12.6 Update `docs/PERFORMANCE.md`: document expected VRAM budgets per model/strategy, co-resident vs. sequential throughput comparison, MMS-FA alignment RTF benchmarks, chunked alignment overhead
- [ ] 12.7 Update `docs/DEPLOYMENT.md`: document VRAM requirements per strategy and model combination, `model_budgets.toml` customization for cloud GPU instances, run profile usage for scheduled/CI pipelines, MMS-FA model weight pre-download for air-gapped deployments
- [ ] 12.8 Update `CONTRIBUTING.md` if needed: note that new pipeline stages must expose `load_model`/`unload_model` and add an entry to `model_budgets.toml`

## 13. Release v0.2.0

- [ ] 13.1 Bump version in `pyproject.toml` from `0.1.1` to `0.2.0`
- [ ] 13.2 Write `CHANGELOG.md` entry for v0.2.0: breaking changes (`TranscriptSegment.words` removed, new `AlignmentResult`), new features (forced alignment via MMS-FA, text normalization, model-agnostic transcription, hybrid VRAM strategy, VRAM preflight validation, interactive `plan` command, run profiles), deprecations (WhisperX backend — removal scheduled for v0.3.0), new dependencies (`faster-whisper`, `num2words`)
- [ ] 13.3 Run `make all-checks` and `make test-integration` on a GPU host; capture wall-clock and peak-VRAM deltas vs. the v0.1.1 baseline for both VRAM strategies
- [ ] 13.4 Create `release/v0.2.0` branch, open PR to `main`, merge
- [ ] 13.5 Create annotated tag `v0.2.0` on merged commit and push — triggers `.github/workflows/release.yml` to build and create the GitHub release
- [ ] 13.6 Verify GitHub release artifact is published and release notes match CHANGELOG

## 14. Validation Gates

- [ ] 14.1 Every spec requirement in `specs/forced-alignment/spec.md`, `specs/text-normalization/spec.md`, `specs/transcription/spec.md`, `specs/model-lifecycle/spec.md` maps to at least one test (unit or integration) that can fail
- [ ] 14.2 `make test` passes (unit only, no GPU)
- [ ] 14.3 `make test-integration` passes on a GPU host for both `--aligner mms-fa` and `--aligner whisperx`, and for both `--vram-strategy co-resident` and `--vram-strategy sequential`
- [ ] 14.4 A 10-minute English podcast clip aligned with MMS-FA has word boundary deltas within ±150 ms of the WhisperX baseline on a hand-labeled subset
- [ ] 14.5 Co-resident mode: total model load time for a 10-file batch is within 15% of the current pipeline's load time (models loaded once, not per-file)
- [ ] 14.6 Sequential mode: peak VRAM on a single-GPU run of a representative file does not exceed the largest single stage's working-set memory
- [ ] 14.7 VRAM preflight correctly blocks an over-budget co-resident run on a real GPU and the error message includes all required fields (per-model breakdown, projected total, available, suggestions)
- [ ] 14.8 `audio-refinery plan` renders the VRAM budget visualization correctly and a saved profile round-trips through `pipeline --plan`
