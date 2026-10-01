# distiller documentation

Start here for all project documentation. Source-level details live in module
docstrings; this directory holds the process and reference docs.

## Index

| Document | Purpose |
|----------|---------|
| [Contributing & Conventions](CONTRIBUTING.md) | Code style, testing, exceptions, docs rules |
| [QLoRA runbook](qlora-runbook.md) | Manual T4 training: upload, train, download, register, compare |
| [GGUF runbook](gguf-runbook.md) | Phase 4: export download, register, serve with Ollama/llama.cpp, evaluate |
| [../README.md](../README.md) | Install, quickstart, architecture overview |
| [../AGENTS.md](../AGENTS.md) | Rules for AI agents working in this repo |

## Conventions in one paragraph

`distiller` follows the Devotion project's engineering standards: 88-character
lines, `ruff` as the single formatter/linter (black-compatible, isort ordering,
complexity gate at 10, bandit security rules), `mypy --strict` on `src/`,
Google-style docstrings on all public API, a `DistillerError` exception
hierarchy, pydantic models for boundary data and dataclasses for internal
values. The test suite is fully offline and split into `tests/unit/` and
`tests/integration/` with factory fixtures.
