.PHONY: help install install-dev install-whisperx test test-verbose test-integration test-coverage lint lint-fix format format-check type-check clean pre-commit pre-commit-install all-checks ci dev-setup stats

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

install-whisperx: ## Install whisperx separately (must run after install or install-dev)
	uv pip install --no-deps "whisperx @ git+https://github.com/m-bain/whisperX.git@v3.1.1"

test: ## Run unit tests (no GPU required)
	uv run python -m pytest tests/ -m "not integration"

test-verbose: ## Run unit tests with verbose output
	uv run python -m pytest tests/ -m "not integration" -vv

test-integration: ## Run integration tests (requires GPU + models)
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

all-checks: lint type-check test ## Run all checks (lint, type-check, test)
	@echo ""
	@echo "All checks passed!"

ci: install-dev all-checks ## Run CI pipeline (install deps + all checks)
	@echo ""
	@echo "CI pipeline completed successfully!"

dev-setup: install-dev install-whisperx pre-commit-install ## Complete development setup (installs all deps including whisperx + hooks)
	@echo ""
	@echo "Development environment setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and add your HF_TOKEN"
	@echo "  2. Run 'make test' to verify everything works"
	@echo "  3. Run 'audio-refinery --help' to see available commands"

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
