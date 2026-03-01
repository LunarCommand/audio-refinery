# Development Guide

This guide covers setting up audio-refinery for development, testing, and contributing.

## Development Setup

### Prerequisites

- Python 3.11.x (strictly — pyannote.audio and WhisperX require 3.11)
- `uv` package manager
- Git
- NVIDIA GPU with 10GB+ VRAM (for integration tests only)
- `ffmpeg` installed on your system

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/LunarCommand/audio-refinery.git
cd audio-refinery

# Complete development setup (installs deps + pre-commit hooks)
make dev-setup

# Or manually:
uv venv --python 3.11.14
source .venv/bin/activate
uv sync --extra dev
cp .env.example .env
# Edit .env with your HF_TOKEN
```

### WhisperX (Optional, Conflicting)

WhisperX must be installed separately after the main dependencies due to PyTorch version constraints:

```bash
uv pip install "whisperx @ git+https://github.com/m-bain/whisperX.git@v3.1.1"
```

See the README for detailed WhisperX installation steps.

## Project Structure

```
audio-refinery/
├── src/                        # Main package
│   ├── __init__.py
│   ├── cli.py                 # Click command group — all CLI entry points
│   ├── separator.py           # Demucs vocal separation wrapper
│   ├── diarizer.py            # Pyannote speaker diarization wrapper
│   ├── transcriber.py         # WhisperX transcription wrapper
│   ├── sentiment_analyzer.py  # HuggingFace sentiment analysis
│   ├── pipeline.py            # Batch pipeline orchestration
│   ├── gpu_utils.py           # GPU queries via nvidia-smi
│   ├── notifier.py            # Slack webhook notifications
│   ├── gpu_tflops.toml        # GPU performance lookup table
│   └── models/                # Pydantic output models
│       ├── audio.py           # AudioFileInfo, SeparationResult
│       ├── diarization.py     # DiarizationResult, SpeakerSegment
│       ├── transcription.py   # TranscriptionResult, TranscriptSegment, WordSegment
│       └── sentiment.py       # SentimentResult, SegmentSentiment, SentimentScore
├── tests/                      # Test suite
│   ├── conftest.py            # Shared fixtures and GPU mock
│   ├── test_*.py              # Unit tests
│   ├── test_integration.py    # GPU-required integration tests
│   └── models/                # Pydantic model validation tests
├── docs/
│   └── DEVELOPMENT.md         # This file
├── .github/
│   └── workflows/
│       ├── ci.yml             # CI: unit tests + lint + type check
│       └── release.yml        # Release automation
├── .pre-commit-config.yaml
├── pyproject.toml
├── uv.lock
├── Makefile
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CLAUDE.md
└── LICENSE
```

## Running Tests

### Unit Tests (No GPU Required)

```bash
# Run unit tests only
make test

# Run with verbose output
make test-verbose

# Run with coverage report
make test-coverage
```

Unit tests mock all GPU operations and model loading. They run on any machine.

### Integration Tests (GPU Required)

Integration tests require a real GPU, and the Demucs, Pyannote, and WhisperX models
to be downloaded. They are excluded from CI.

```bash
make test-integration
```

### Test Coverage

The test suite uses `@pytest.mark.integration` to separate GPU-dependent tests.
The `conftest.py` `_gpu_free` autouse fixture patches `nvidia-smi` calls so unit tests
run without GPU access.

**Coverage targets:**

- Minimum 80% code coverage for unit-testable modules
- All public functions and CLI commands should have unit tests
- Edge cases and error paths covered

## Code Quality

### Linting

```bash
# Check for linting issues
make lint

# Auto-fix linting issues
make lint-fix
```

### Formatting

```bash
# Format code
make format

# Check formatting without changing files
make format-check
```

### Type Checking

```bash
# Run mypy (loose configuration — --ignore-missing-imports)
make type-check
```

### All Checks

```bash
# Run lint + type-check + unit tests
make all-checks
```

## Pre-commit Hooks

audio-refinery uses pre-commit hooks for automated quality checks on every commit:

```bash
# Install hooks (done automatically by make dev-setup)
make pre-commit-install

# Run manually on all files
make pre-commit
```

Hooks run:

1. **pre-commit-hooks**: trailing whitespace, EOF newline, YAML/JSON/TOML validation,
   merge conflict detection, debug statement detection
2. **ruff**: lint check with auto-fix
3. **ruff-format**: code formatting check
4. **mypy**: type checking with `--ignore-missing-imports`

## Dependencies

### Production Dependencies

- **torch** (2.1.2): PyTorch — pinned for WhisperX compatibility
- **torchaudio** (2.1.2): Audio I/O utilities
- **demucs** (4.0.1): Vocal separation via `htdemucs`
- **soundfile** (0.12.1): Audio file reading and metadata
- **pyannote.audio** (>=3.1.1): Speaker diarization
- **python-dotenv**: `.env` file loading for `HF_TOKEN` and `SLACK_WEBHOOK_URL`
- **pydantic**: Data validation and pipeline output models
- **numpy** (<2.0.0): Pinned — WhisperX ctranslate2 backend requires <2.0
- **pandas**: DataFrame support for diarization output
- **ffmpeg-python**: FFmpeg subprocess wrapper
- **click** (>=8.1): CLI framework
- **rich** (>=13.0): Terminal formatting, tables, and progress spinners
- **transformers** (>=4.30.0): HuggingFace models for sentiment analysis

### Conflicting Dependencies (Install Separately)

- **whisperx** (git+v3.1.1): Transcription with word-level alignment — must be installed
  after main dependencies due to PyTorch version constraints

### Development Dependencies

- **ruff**: Fast linter and formatter
- **mypy**: Static type checker (loose configuration)
- **pytest**: Testing framework
- **pytest-mock**: Mock fixtures
- **pytest-cov**: Coverage reporting
- **pre-commit**: Git hook management

## Architecture Overview

### Pipeline Stages

Each stage is a standalone module with a pure-function API:

| Module | Function | Output Model |
|--------|----------|--------------|
| `separator.py` | `separate()` | `SeparationResult` |
| `diarizer.py` | `diarize()` | `DiarizationResult` |
| `transcriber.py` | `transcribe()` | `TranscriptionResult` |
| `sentiment_analyzer.py` | `analyze_sentiment()` | `SentimentResult` |

### Data Flow

```
WAV file
  → separate()     → vocals.wav + SeparationResult.json
  → diarize()      → DiarizationResult.json
  → transcribe()   → TranscriptionResult.json (with speaker labels if diarization provided)
  → analyze_sentiment() → SentimentResult.json (updates TranscriptionResult in-place)
```

### Batch Pipeline (`pipeline.py`)

Orchestrates all four stages for a directory of WAV files:

- Loads all models once at startup
- Interleaved per-file processing (separation → diarize → transcribe → sentiment per file)
- VRAM tracking with peak memory reporting per stage
- Scratch space cleanup (ghost-track stems deleted as soon as unneeded)
- Resume behavior: skips files with existing output JSON
- Thermal monitoring with configurable shutdown (default 80°C)

### GPU Utilities (`gpu_utils.py`)

- `query_compute_processes()`: Detect active GPU processes via `nvidia-smi` (preflight check)
- `query_gpu_temperature()`: Monitor thermal state
- `detect_gpu_order()`: Rank GPUs by TFLOPS from `gpu_tflops.toml` for multi-GPU distribution

## Publishing Releases

audio-refinery is distributed as a GitHub release. Releases are automated via GitHub Actions.

### Semantic Versioning

- **MAJOR** (X.0.0): Breaking changes to CLI interface or output JSON schema
- **MINOR** (x.Y.0): New features, new pipeline stages, backward compatible
- **PATCH** (x.y.Z): Bug fixes, documentation updates

### Release Process

#### 1. Prepare the Release

```bash
# Create a release branch from main
git checkout main
git pull origin main
git checkout -b release/vX.Y.Z

# Update version in pyproject.toml
# Update CHANGELOG.md with release notes

# Run all quality checks
make all-checks

# Commit changes
git add pyproject.toml CHANGELOG.md
git commit -m "Prepare release vX.Y.Z"
git push origin release/vX.Y.Z
```

#### 2. Create PR and Merge

```bash
gh pr create --base main --head release/vX.Y.Z --title "Release vX.Y.Z"
# After PR is approved and CI passes, merge via GitHub UI
```

#### 3. Tag and Release

```bash
# Switch to main and pull the merged changes
git switch main
git pull

# Create and push an annotated tag
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z

# Clean up release branch
git branch -d release/vX.Y.Z
```

#### 4. Automated Pipeline

When you push the tag, GitHub Actions automatically:

1. **Runs unit tests** — Ensures all tests pass
2. **Builds package** — Creates wheel and sdist with `python -m build`
3. **Creates GitHub Release** — With package artifacts and auto-generated release notes

Monitor at: https://github.com/LunarCommand/audio-refinery/actions

### Release Checklist

Before creating a release:

- [ ] All unit tests pass: `make test`
- [ ] Code is linted: `make lint`
- [ ] Type checking passes: `make type-check`
- [ ] Version updated in `pyproject.toml`
- [ ] `CHANGELOG.md` updated with release notes
- [ ] All PRs for this release are merged
- [ ] README is up to date

After release:

- [ ] GitHub Release page shows correct version and artifacts
- [ ] Release notes are accurate

### Quick Reference

```bash
# Complete release workflow
git checkout main && git pull
git checkout -b release/vX.Y.Z
# Update pyproject.toml version and CHANGELOG.md
make all-checks
git add pyproject.toml CHANGELOG.md
git commit -m "Prepare release vX.Y.Z"
git push origin release/vX.Y.Z
gh pr create --base main --head release/vX.Y.Z --title "Release vX.Y.Z"
# Merge PR via GitHub UI
git switch main && git pull
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
git branch -d release/vX.Y.Z
```

### Hotfix Releases

For urgent bug fixes:

```bash
git checkout main && git pull
git checkout -b hotfix/vX.Y.Z
# Fix the bug, then follow the same release process
# Use PATCH version bump (e.g., 0.1.0 -> 0.1.1)
```

## Troubleshooting

### Virtual Environment Issues

```bash
rm -rf .venv
uv venv --python 3.11.14
uv sync --extra dev
```

### WhisperX Import Errors

WhisperX must be installed separately after the main dependencies:

```bash
uv pip install "whisperx @ git+https://github.com/m-bain/whisperX.git@v3.1.1"
```

### Test Failures

```bash
# Run with verbose output to see details
uv run pytest -vv tests/ -m "not integration"

# Run a specific test
uv run pytest tests/test_separator.py::TestSeparate -vv
```

### Mypy Errors on Third-Party Libraries

The mypy configuration uses `--ignore-missing-imports` so missing stubs won't cause
failures. If you see unexpected errors:

```bash
# Run with more detail
uv run mypy src/ --ignore-missing-imports --show-error-codes
```

## Resources

- [Demucs](https://github.com/facebookresearch/demucs) — Vocal separation model
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio) — Speaker diarization
- [WhisperX](https://github.com/m-bain/whisperX) — Transcription with alignment
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — Sentiment model
- [Click](https://click.palletsprojects.com/) — CLI framework
- [Rich](https://rich.readthedocs.io/) — Terminal UI library
- [Pydantic](https://docs.pydantic.dev/) — Data validation

## License

MIT License — see [LICENSE](../LICENSE) for details.
