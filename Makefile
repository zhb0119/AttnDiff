.PHONY: help install test lint format clean

help:
	@echo "AttnDiff Development Commands"
	@echo ""
	@echo "install     - Install dependencies with UV"
	@echo "test        - Run test suite"
	@echo "lint        - Run linting checks"
	@echo "format      - Format code"
	@echo "type-check  - Run type checking"
	@echo "clean       - Remove build artifacts"

install:
	uv sync --all-extras
	uv run pre-commit install

test:
	uv run pytest tests/ -v --cov=attndiff --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check . --fix

type-check:
	uv run mypy src/attndiff

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
