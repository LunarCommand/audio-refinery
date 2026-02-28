# Contributing to audio-refinery

Thank you for your interest in contributing!

## Development Setup

**Requirements:** Python 3.11.x, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/your-username/audio-refinery
cd audio-refinery
uv venv --python 3.11.14
source .venv/bin/activate
uv sync
```

Copy the example env file and fill in your tokens:

```bash
cp .env.example .env
# Edit .env with your HF_TOKEN
```

## Running Tests

Non-integration tests (no GPU required):

```bash
pytest tests/ -m "not integration" -v
```

Integration tests (requires GPU, Demucs, WhisperX, and a test audio file):

```bash
pytest tests/ -m "integration" -v
```

## Linting and Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/):

```bash
ruff check src/ tests/          # lint
ruff format src/ tests/         # format
ruff check --fix src/ tests/    # auto-fix
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Write tests for any new functionality
3. Ensure all non-integration tests pass: `pytest tests/ -m "not integration"`
4. Ensure ruff checks pass: `ruff check src/ tests/`
5. Update the README if you've changed CLI options or pipeline behaviour
6. Open a pull request with a clear description of your changes

## Dependency Notes

- **NumPy must stay <2.0.0** — required by WhisperX's ctranslate2 backend
- **WhisperX is in `[project.optional-dependencies] conflicting`** — install after the main dependencies due to PyTorch version constraints; see README for the exact steps
- **Python 3.11.x strictly** — pyannote.audio and WhisperX do not support 3.12+ yet

## Code Style

- Line length: 120 characters
- Double quotes for strings
- Import sorting enforced by ruff (I rules)
- No type annotations required on existing code, but appreciated on new public APIs
