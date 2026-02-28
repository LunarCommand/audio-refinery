.PHONY: lint format test test-integration clean

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ -m "not integration" -v

test-integration:
	pytest tests/ -m "integration" -v

clean:
	rm -rf .venv __pycache__ src/__pycache__ tests/__pycache__ *.egg-info dist build .pytest_cache
