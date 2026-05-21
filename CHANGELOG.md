# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-05-21

### Fixed

- Service mode now pins `CUDA_DEVICE_ORDER=PCI_BUS_ID` at startup (matching the CLI), so `REFINERY_DEVICE=cuda:N` selects the GPU at `nvidia-smi` index N rather than CUDA's default FASTEST_FIRST ordering. Previously a multi-GPU host could load models onto the wrong card.
- WhisperX forced-alignment failures are now logged at debug level (with the detected language and device) before falling back to transcription-only segments, instead of being swallowed silently.

### Documentation

- Corrected the service-mode tmpfs example to `--mount type=tmpfs,dst=/scratch,tmpfs-mode=1777`. A bare `--tmpfs /scratch` mounts root-owned, so the non-root `refinery` user could not write scratch and every job failed with `PermissionError`. Applied the same fix in `docs/deployment.md`.
- Added service-mode GPU-selection guidance, an end-to-end local-directory batch runbook, and troubleshooting entries (tmpfs permissions, wrong-GPU selection, batch-size cap) to `docs/service.md`.
- Added the project banner to the README.

### Security

- Upgraded vulnerable transitive and dev dependencies to patched versions: `aiohttp` 3.13.3 → 3.13.5, `requests` 2.32.5 → 2.34.2, `Pygments` 2.19.2 → 2.20.0, and `pytest` 9.0.2 → 9.0.3. Closes 13 Dependabot alerts (10 `aiohttp` plus `requests`, `Pygments`, `pytest`). The `torch` and `transformers` alerts remain deferred to v0.3.0, when WhisperX removal lifts the `torch==2.1.2` / `transformers<4.40` pins.

## [0.2.0] - 2026-05-20

### Added

- HTTP service mode: a long-lived FastAPI service (`audio-refinery-service` entry point) exposing `POST /transcribe`, `GET /jobs/{id}`, and `GET /health`. Accepts multi-job batches, processes them serially on a background worker, and writes one combined transcript per job plus one summary per batch. Models load once at container startup and stay resident across jobs.
- Combined transcript schema (v1.0.0) and per-batch summary schema (v1.0.0) — service-mode output documents that envelope the existing per-stage results so consumers read a single document per job/batch.
- Bearer-token authentication via the `REFINERY_API_KEYS` allowlist; request and job-lifecycle logs carry a non-reversible caller fingerprint.
- URI-driven I/O for the service: `https://` (presigned GET for input, presigned PUT for output/summary) and `file://` for local-dev and shared-volume deployments.
- Multi-stage Dockerfile (CUDA 12.1 runtime base, non-root `refinery` user) published as `lunarcommand/audio-refinery` on Docker Hub.
- `audio-refinery serve` CLI subcommand — a convenience wrapper around the service entry point.
- Optional GPU thermal guard in service mode via `REFINERY_GPU_TEMP_LIMIT` / `REFINERY_GPU_TEMP_POLL_SECONDS`, reusing the CLI's monitor.
- Service configuration via environment variables: `REFINERY_API_KEYS`, `REFINERY_DEVICE`, `REFINERY_WHISPER_MODEL`, `REFINERY_COMPUTE_TYPE`, `REFINERY_DEFAULT_LANGUAGE`, `REFINERY_SENTIMENT_ENABLED`, `REFINERY_SCRATCH_DIR`, `REFINERY_INTERMEDIATE_DIR`, `REFINERY_MAX_BATCH_SIZE`, `REFINERY_MAX_QUEUE_SIZE`, `REFINERY_JOB_RETENTION_SECONDS`, `REFINERY_PORT`, and `REFINERY_LOG_FORMAT`.

### Changed

- Demucs scratch directory is now host-agnostic: it defaults to `tempfile.gettempdir()/audio-refinery-demucs` and honors `REFINERY_SCRATCH_DIR` in both CLI and service mode, replacing the hard-coded `/mnt/fast_scratch` path.
- CLI batch input now accepts any `*.wav` file; the `audio_` filename prefix is optional and stripped from the derived content ID, replacing the previous required `audio_<id>.wav` pattern.
- Sentiment analysis treats segments with no transcribed speech (e.g. silent audio) as no-ops rather than failures.
- Added `fastapi`, `uvicorn[standard]`, and `httpx` as main dependencies for service mode.
- Documentation restructured into a two-path (CLI + service) layout: lowercase `docs/` filenames, a slimmed README with a "choose your path" landing, and new `docs/cli.md`, `docs/service.md`, and `docs/index.md`.

### Security

- Scoped both GitHub Actions workflows (`ci.yml`, `release.yml`) to least-privilege permissions. Workflow default is now `contents: read`; the `github-release` job retains an explicit `contents: write` override. Closes CodeQL workflow-permission alerts.
- Upgraded vulnerable transitive dependencies: `idna` 3.11 → 3.15, `Mako` 1.3.10 → 1.3.12, `Pillow` 12.1.1 → 12.2.0, `urllib3` 2.6.3 → 2.7.0. Closes nine Dependabot alerts covering path-traversal, redirect-header forwarding, decompression-bomb, OOB-write, and integer-overflow CVEs.

### Known security debt

- 18 Dependabot alerts remain open against `torch` and `transformers` (including one critical and several high severity). Both packages are pinned to WhisperX-compatible versions (`torch==2.1.2`, `transformers>=4.30,<4.40`) and cannot be upgraded without breaking the current ASR pipeline. These alerts close together when v0.3.0 lands and removes WhisperX from the critical path; see the universal alignment plan for the migration path.

### Tests

- Suppressed live Slack webhook calls during the test run via an autouse `_no_slack` fixture in `tests/conftest.py`. Previously, `notifier._send()` loaded the developer's `.env` on every invocation and POSTed real messages whenever pipeline tests triggered end-of-batch notifications.

## [0.1.1] - 2026-03-06

### Added

- `combined_report.json` now includes four derived metrics: `avg_time_per_file_seconds`, `avg_time_per_mb_seconds`, `processing_speed_ratio` (real-time factor), and `words_per_audio_hour` (transcription density)
- Slack notifications now include detailed per-stage stats (processed / skipped / failed counts) and average processing time per file
- `make test-slack` Makefile target for validating Slack webhook integration
- Dockerfile and `.dockerignore` for containerized deployment
- Sentiment output directory (`<base>/sentiment/`) support in batch pipeline

### Changed

- Centralized Demucs scratch directory resolution in CLI — RAM disk detection and fallback confirmation now happen in one place
- Worker status reporting and failure aggregation in `pipeline-parallel` refactored for improved accuracy
- `python-dotenv` import in Slack notifier is now conditional — avoids import-time failure when the package is absent
- DEPLOYMENT.md expanded: Hugging Face token setup, NVIDIA driver requirements, cloud instance guidelines, and Docker usage
- Combined report fields documented in README under the Parallel Pipeline section

### Fixed

- Narrowed exception handling in `gpu_utils.py`, `transcriber.py`, and `notifier.py` to avoid masking unexpected errors
- Typo in `SeparationError` docstring

## [0.1.0] - 2026-03-01

### Added

- Initial release
- `separate` command: GPU-accelerated vocal separation via Demucs `htdemucs`
- `diarize` command: Speaker diarization via Pyannote `speaker-diarization-3.1`
- `transcribe` command: Transcription with word-level alignment via WhisperX `large-v3`
- `sentiment` command: Per-segment sentiment analysis via `cardiffnlp/twitter-roberta-base-sentiment-latest`
- `pipeline` command: Single-GPU batch processing through all four stages
- `pipeline-parallel` command: Multi-GPU batch processing with worker distribution
- GPU pre-flight checks via `nvidia-smi` with active process detection
- Thermal monitoring with configurable shutdown threshold (default 80°C)
- Slack webhook notifications for pipeline completion and thermal events
- Resume behavior for batch pipelines (skips already-completed files)
- Pydantic data models for all pipeline outputs with full provenance tracking
- Rich terminal output with progress spinners and result tables
- VRAM usage tracking and peak memory reporting per stage
- Scratch space management with automatic cleanup of intermediate files
- GPU performance-based ordering via `gpu_tflops.toml` lookup table

### Fixed

- WhisperX model loading: pinned install to commit `741ab9a` (v3.7.6) — the `v3.1.1` tag uses an older API (`transcribe.py`) that lacks the `device_index` parameter required by the ctranslate2 backend
- `make install-whisperx` now also installs WhisperX runtime deps (`av`, `ctranslate2`, `faster-whisper`, `flatbuffers`, `nltk`, `onnxruntime`) which were previously missing after a `--no-deps` install
- `transformers` capped at `<4.40.0` — versions 4.40+ use `torch.utils._pytree.register_pytree_node`, an API introduced in PyTorch 2.2, which breaks with the pinned PyTorch 2.1.2
- `make dev-setup` now reinstalls CUDA torch wheels (`torch==2.1.2+cu121`, `torchaudio==2.1.2+cu121`) as its final step — `uv sync` resolves torch from PyPI and installs the CPU-only build, silently breaking GPU inference

[Unreleased]: https://github.com/LunarCommand/audio-refinery/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/LunarCommand/audio-refinery/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/LunarCommand/audio-refinery/releases/tag/v0.1.0
