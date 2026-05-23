.PHONY: help install install-dev install-whisperx test test-verbose test-integration test-coverage lint lint-fix format format-check type-check clean pre-commit pre-commit-install all-checks ci dev-setup stats build-image run-service-local generate-schemas check-schemas

# Read the version from pyproject.toml so build-image stays in sync with releases.
VERSION := $(shell awk -F'"' '/^version = /{print $$2; exit}' pyproject.toml)
IMAGE_NAME := lunarcommand/audio-refinery

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "audio-refinery - GPU-accelerated audio processing pipeline"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies (excludes whisperx — see install-whisperx)
	uv sync

install-dev: ## Install development dependencies (excludes whisperx — see install-whisperx)
	uv sync --extra dev

install-whisperx: ## Install whisperx and its runtime deps separately (must run after install or install-dev)
	uv pip install --no-deps "whisperx @ git+https://github.com/m-bain/whisperX.git@741ab9a2a8a1076c171e785363b23c55a91ceff1"
	uv pip install "av==16.1.0" "ctranslate2==4.7.1" "faster-whisper==1.2.1" "flatbuffers==25.12.19" "nltk==3.9.2" "onnxruntime==1.24.1"

test: ## Run unit tests (no GPU required)
	uv run python -m pytest tests/ -m "not integration"

test-verbose: ## Run unit tests with verbose output
	uv run python -m pytest tests/ -m "not integration" -vv

test-integration: ## Run integration tests (GPU + REFINERY_TEST_AUDIO_DIR + HF_TOKEN required)
	@echo "Integration tests use WAV files from REFINERY_TEST_AUDIO_DIR (or tests/_audio_fixtures/)."
	@echo "Set HF_TOKEN for pyannote model download. Tests skip cleanly when fixtures are missing."
	uv run python -m pytest tests/ -m "integration" -v

test-coverage: ## Run unit tests with coverage report (fails under 75%)
	uv run python -m pytest tests/ -m "not integration" --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=75
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

lint: ## Run linter (ruff)
	uv run ruff check src/ tests/

lint-fix: ## Run linter and auto-fix issues
	uv run ruff check --fix src/ tests/

format: ## Format code with ruff
	uv run ruff format src/ tests/

format-check: ## Check code formatting without changing files
	uv run ruff format --check src/ tests/

type-check: ## Run type checker (mypy)
	uv run mypy src/ --ignore-missing-imports

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

clean: ## Remove generated files and caches
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache coverage.xml dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

generate-schemas: ## Regenerate docs/schemas/*.json from the Pydantic models
	uv run python scripts/generate_schemas.py

check-schemas: ## Fail if docs/schemas/*.json is out of sync with the Pydantic models
	@uv run python scripts/generate_schemas.py >/dev/null
	@if [ -n "$$(git status --porcelain docs/schemas/)" ]; then \
		echo ""; \
		echo "docs/schemas/ is out of sync — run 'make generate-schemas' and commit."; \
		git status --short docs/schemas/; \
		exit 1; \
	fi

all-checks: lint type-check test check-schemas ## Run all checks (lint, type-check, test, schema drift)
	@echo ""
	@echo "All checks passed!"

ci: install-dev all-checks ## Run CI pipeline (install deps + all checks)
	@echo ""
	@echo "CI pipeline completed successfully!"

install-torch-cuda: ## Reinstall PyTorch with CUDA 12.1 wheels (run after uv sync, which pulls CPU-only builds)
	uv pip install torch==2.1.2+cu121 torchaudio==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

dev-setup: install-dev install-whisperx install-torch-cuda pre-commit-install ## Complete development setup (installs all deps including whisperx + hooks)
	@echo ""
	@echo "Development environment setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and add your HF_TOKEN"
	@echo "  2. Run 'make test' to verify everything works"
	@echo "  3. Run 'audio-refinery --help' to see available commands"

test-slack: ## Send a test Slack notification to verify SLACK_WEBHOOK_URL is configured
	@uv run python -c "\
from dotenv import load_dotenv; \
load_dotenv(); \
import os, sys, json, urllib.request; \
url = os.getenv('SLACK_WEBHOOK_URL') or (print('SLACK_WEBHOOK_URL is not set — add it to .env or export it') or sys.exit(1)); \
data = json.dumps({'text': ':white_check_mark: *Test notification* from \`audio-refinery\` — Slack integration is working.'}).encode(); \
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}); \
urllib.request.urlopen(req, timeout=5); \
print('Test notification sent — check your Slack channel')"

build-image: ## Build the Docker image, tagged with the pyproject.toml version + :latest
	docker build -t $(IMAGE_NAME):$(VERSION) -t $(IMAGE_NAME):latest .
	@echo ""
	@echo "Built $(IMAGE_NAME):$(VERSION) and $(IMAGE_NAME):latest"

run-service-local: ## Run the service container against bind-mounted /inbox + /outbox + /summaries
	@if [ -z "$(REFINERY_API_KEYS)" ]; then echo "REFINERY_API_KEYS must be set in your environment"; exit 1; fi
	@if [ -z "$(HF_TOKEN)" ]; then echo "HF_TOKEN must be set in your environment"; exit 1; fi
	docker run --rm --gpus all \
		-p 8000:8000 \
		-e REFINERY_API_KEYS \
		-e HF_TOKEN \
		-e REFINERY_LOG_FORMAT=console \
		-v $(PWD)/.docker-inbox:/inbox \
		-v $(PWD)/.docker-outbox:/outbox \
		-v $(PWD)/.docker-summaries:/summaries \
		$(IMAGE_NAME):latest

stats: ## Show project statistics
	@echo "Project Statistics:"
	@echo "==================="
	@printf "Lines of code (src/): "
	@find src -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $$1}'
	@printf "Lines of tests: "
	@find tests -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $$1}'
	@printf "Test files: "
	@find tests -name "test_*.py" | wc -l | awk '{print $$1}'
	@printf "Python modules: "
	@find src -name "*.py" | wc -l | awk '{print $$1}'
	@printf "Python version: "
	@cat .python-version
