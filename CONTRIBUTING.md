# Contributing to audio-refinery

Thank you for considering contributing! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, GPU model and VRAM, CUDA version)
- Relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:

- A clear description of the enhancement
- The motivation and use case
- Any implementation ideas you have

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** — follow existing code style, keep changes focused
3. **Write tests** for any new functionality
4. **Ensure unit tests pass**: `make test`
5. **Ensure lint and type checks pass**: `make all-checks`
6. **Update documentation** if you've changed CLI options or pipeline behavior
7. **Update CHANGELOG.md** under `[Unreleased]`
8. **Open a pull request** with a clear description of your changes

## Development Setup

**Requirements:** Python 3.11.x, [uv](https://github.com/astral-sh/uv), Git

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
pre-commit install
cp .env.example .env
# Edit .env with your HF_TOKEN
```

### WhisperX (Optional)

WhisperX must be installed separately after the main dependencies:

```bash
uv pip install "whisperx @ git+https://github.com/m-bain/whisperX.git@v3.1.1"
```

## Running Tests

Non-integration tests (no GPU required):

```bash
make test
```

Integration tests (requires GPU, Demucs, Pyannote, WhisperX):

```bash
make test-integration
```

Run all quality checks at once:

```bash
make all-checks   # lint + type-check + unit tests
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting,
and [mypy](https://mypy-lang.org/) for type checking.

```bash
make lint          # check for issues
make lint-fix      # auto-fix issues
make format        # format code
make type-check    # run mypy
```

- **Line length**: 120 characters
- **Quotes**: double quotes for strings
- **Import sorting**: enforced by ruff (`I` rules)
- **Type annotations**: not required on existing code, but appreciated on new public APIs

## Dependency Notes

- **NumPy must stay <2.0.0** — required by WhisperX's ctranslate2 backend
- **WhisperX is in `[project.optional-dependencies] conflicting`** — install separately after
  the main dependencies due to PyTorch version constraints
- **Python 3.11.x strictly** — pyannote.audio and WhisperX do not support 3.12+ yet
- **PyTorch pinned at 2.1.2** — required for WhisperX compatibility

## Architecture Notes

Before making major architectural changes, please open an issue to discuss the approach.
Key design decisions:

- Each processing stage is a standalone module with a pure-function API
- All stage outputs are Pydantic models serialized to JSON
- `cli.py` and `pipeline.py` are the only orchestrators; stage modules have no CLI coupling
- GPU pre-flight checks (via `query_compute_processes()`) run before any GPU operation
- Test mocking: the `_gpu_free` autouse fixture in `conftest.py` patches nvidia-smi

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
