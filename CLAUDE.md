# CLAUDE.md — AI Assistant Development Guide

This file helps AI coding assistants understand the audio-refinery project structure,
architecture, and conventions to provide accurate code suggestions.

## Project Overview

**audio-refinery** is a GPU-accelerated audio processing pipeline that takes raw audio
files through four sequential stages: vocal separation → speaker diarization →
transcription → sentiment analysis.

- **Language**: Python 3.11.x (strictly — pyannote.audio and WhisperX require 3.11)
- **Package manager**: uv
- **CLI framework**: Click
- **Terminal output**: Rich
- **Data models**: Pydantic
- **Testing**: pytest with `integration` marker for GPU-dependent tests

## Project Structure

```
audio-refinery/
├── src/                        # Main package (imported as `src.*`)
│   ├── __init__.py
│   ├── cli.py                 # Click command group — all CLI entry points (1849 lines)
│   ├── separator.py           # Demucs vocal separation wrapper
│   ├── diarizer.py            # Pyannote speaker diarization wrapper
│   ├── transcriber.py         # WhisperX transcription wrapper
│   ├── sentiment_analyzer.py  # HuggingFace sentiment analysis
│   ├── pipeline.py            # Batch pipeline orchestration (947 lines)
│   ├── gpu_utils.py           # GPU queries via nvidia-smi
│   ├── notifier.py            # Slack webhook notifications
│   ├── gpu_tflops.toml        # GPU performance lookup table
│   ├── models/                # Pydantic output models
│   │   ├── __init__.py
│   │   ├── audio.py           # AudioFileInfo, SeparationResult
│   │   ├── diarization.py     # DiarizationResult, SpeakerSegment
│   │   ├── transcription.py   # TranscriptionResult, TranscriptSegment, WordSegment
│   │   └── sentiment.py       # SentimentResult, SegmentSentiment, SentimentScore
│   └── service/               # HTTP service mode (parallel to CLI; same core pipeline)
│       ├── __init__.py
│       ├── app.py             # FastAPI app, endpoints, lifespan, `audio-refinery-service` entry
│       ├── auth.py            # Bearer-token middleware + allowlist
│       ├── jobs.py            # Job registry, FIFO queue, background-thread worker
│       ├── lifecycle.py       # Model warmup, readiness state, pre-loaded handles
│       ├── api_schemas.py     # HTTP transport schemas (request/response Pydantic models)
│       ├── config.py          # ServiceConfig + PipelineHandles (pure data)
│       ├── schemas.py         # Combined transcript + batch summary Pydantic schemas (v1.0.0)
│       └── uri_io.py          # URI fetch/upload (https://, file://)
├── tests/                      # Test suite
│   ├── conftest.py            # Shared fixtures (GPU mock, tmp dirs, synthetic audio)
│   ├── test_cli.py
│   ├── test_separator.py
│   ├── test_diarizer.py
│   ├── test_transcriber.py
│   ├── test_sentiment_analyzer.py
│   ├── test_pipeline.py
│   ├── test_pipeline_parallel.py
│   ├── test_gpu_utils.py
│   ├── test_integration.py    # GPU-required tests (mark: integration)
│   ├── models/                # Pydantic model validation tests
│   └── service/               # Service-mode unit/integration tests
├── docs/
│   └── development.md         # Developer guide
├── .github/
│   └── workflows/
│       ├── ci.yml             # CI: unit tests + lint + type check
│       └── release.yml        # Release: test → build → GitHub release
├── .pre-commit-config.yaml
├── pyproject.toml
├── Makefile
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

## Architecture

### Pipeline Stages

Each stage is a standalone module with a pure-function API:

```
separator.py    → separate(input_file, ...) → SeparationResult
diarizer.py     → diarize(audio_file, ...) → DiarizationResult
transcriber.py  → transcribe(audio_file, ...) → TranscriptionResult
sentiment_analyzer.py → analyze_sentiment(transcription_file, ...) → SentimentResult
```

`pipeline.py` orchestrates these in sequence for batch processing.

### Service Mode (src/service/)

A long-lived HTTP service that wraps the same `run_pipeline()` core as the CLI.
Adds: bearer-auth HTTP API (`POST /transcribe`, `GET /jobs/{id}`, `GET /health`),
URI fetch/upload (`https://` presigned + `file://`), background-thread worker
processing jobs serially, single combined transcript JSON output, and
`.error.json` sidecars for failures. Models load once at container startup
and stay resident. See `_reqs/service-mode.md` and `_plans/service-mode-plan.md`.

### CLI (cli.py)

- Click command group: `audio-refinery`
- Commands: `separate`, `diarize`, `transcribe`, `sentiment`, `pipeline`, `pipeline-parallel`
- All commands use Rich panels/tables for output
- GPU pre-flight check runs before any GPU operation (via `query_compute_processes()`)
- Device strings follow PyTorch convention: `cuda`, `cuda:0`, `cuda:1`, `cpu`

### Data Models (src/models/)

All pipeline outputs are Pydantic models with full provenance:
- Input file path, model used, device, timestamps, processing time
- Results are serialized to JSON in the output directory
- Downstream stages read the JSON output of upstream stages

### GPU Utilities (gpu_utils.py)

- `query_compute_processes()` — detect active GPU users via nvidia-smi
- `query_gpu_temperature()` — monitor thermal state
- `detect_gpu_order()` — rank GPUs by TFLOPS from `gpu_tflops.toml`

### Test Markers

- `@pytest.mark.integration` — requires real GPU, Demucs, Pyannote, WhisperX
- Unit tests mock all external dependencies (GPU, models, subprocess calls)
- `conftest.py` provides `_gpu_free` autouse fixture that patches nvidia-smi

## Critical Dependency Notes

- **PyTorch**: pinned to `2.1.2` (WhisperX constraint)
- **NumPy**: must stay `<2.0.0` (WhisperX ctranslate2 backend)
- **WhisperX**: in `[project.optional-dependencies] conflicting` — install separately AFTER main deps due to PyTorch version conflict
- **Python**: strictly 3.11.x — pyannote.audio and WhisperX don't support 3.12+

## Development Commands

```bash
make dev-setup        # Install deps + pre-commit hooks
make test             # Run unit tests only
make test-integration # Run integration tests (requires GPU)
make lint             # Check with ruff
make lint-fix         # Auto-fix lint issues
make format           # Format with ruff
make type-check       # Run mypy
make all-checks       # lint + type-check + test
make pre-commit       # Run pre-commit on all files
```

## Code Conventions

- Line length: 120 characters
- Double quotes for strings
- Import sorting: ruff with `I` rules
- Pydantic models for all structured output
- Pure functions in stage modules; orchestration in `pipeline.py` and `cli.py`
- No type annotations required on existing code; encouraged on new public APIs
