.PHONY: help install install-all format lint typecheck security unit-test test coverage fixtures check clean

help:
	@echo "Available commands:"
	@echo "  make install        - Install the project with dev tools (uv sync)"
	@echo "  make install-all    - Install with all extras (docling, torch, qdrant, ragas)"
	@echo "  make format         - Format, typecheck, lint and security-check the codebase"
	@echo "  make lint           - Lint only (ruff)"
	@echo "  make typecheck      - Strict type checking (mypy)"
	@echo "  make security       - Security scan (ruff bandit rules)"
	@echo "  make unit-test      - Run unit tests in parallel"
	@echo "  make test           - Run all tests (unit + integration, fully offline)"
	@echo "  make coverage       - Coverage report (fails under 80%)"
	@echo "  make fixtures       - Download a Gutenberg book EPUB+PDF into fixtures/"
	@echo "  make check          - Lint + typecheck + tests (what CI runs)"
	@echo "  make clean          - Remove caches and coverage output"

install:
	uv sync

install-all:
	uv sync --all-extras

format:
	@echo ""
	@echo "FORMATTING CODE:"
	@echo ""
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts
	@echo ""
	@echo "CHECKING CODE STILL NEEDS FORMATTING:"
	@echo ""
	uv run ruff format --check src tests scripts || exit 1
	@echo ""
	@echo "CHECKING TYPING:"
	@echo ""
	uv run mypy || exit 1
	@echo ""
	@echo "CHECKING LINT:"
	@echo ""
	uv run ruff check src tests scripts || exit 1
	@echo ""
	@echo "CHECKING SECURITY:"
	@echo ""
	uv run ruff check --select S src || exit 1

lint:
	uv run ruff check src tests scripts

typecheck:
	uv run mypy

security:
	uv run ruff check --select S src

unit-test:
	uv run pytest tests/unit -vv -n auto || exit 1

test:
	uv run pytest tests/ -v --tb=short || exit 1

coverage:
	uv run pytest tests/ --cov --cov-report=term-missing -n auto || exit 1

fixtures:
	uv run python scripts/fetch_gutenberg.py --id 1661 --out fixtures/ --slug sherlock-holmes

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
