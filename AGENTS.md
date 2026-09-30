# distiller - Agent Guide

**Quick Links:**
- 📋 **[Contributing & Conventions](docs/CONTRIBUTING.md)** - Code and documentation rules (MANDATORY read)
- 📚 **[Documentation Index](docs/README.md)** - All project documentation
- 🏗️ **[README](README.md)** - Install, quickstart and architecture

## Project Overview

**distiller** turns a PDF/ePub **book** into a searchable, cited knowledge index and
(on the roadmap) into a specialized 3-4B model distilled from that book via
RAFT-style fine-tuning.

- **Stack**: Python 3.12+, pydantic/pydantic-settings, Typer CLI, NumPy/Qdrant,
  sentence-transformers (optional), ruff + mypy(strict) + pytest.
- **Status**: v1 complete (ingest → chunk → hybrid index → cited answers → eval),
  fully offline test suite (22+ tests, hash embedder + fake LLM).

---

## Critical Rules for Agents

### ⚠️ RULE #0: NO MARKDOWN IN PROJECT ROOT

All markdown files MUST go in `docs/`. ONLY exceptions: `AGENTS.md` and
`README.md` (required by the packaging metadata).

- ❌ **NEVER**: `/home/matheus/dev/py/distiller/FILENAME.md`
- ✅ **ALWAYS**: `/home/matheus/dev/py/distiller/docs/FILENAME.md`

### ⚠️ RULE #1: Git Commits Require Explicit Approval

**NEVER commit or push to Git without explicit maintainer approval.**

1. **ALWAYS** ask before using any git command (`git add`, `git commit`, `git push`,
   `git stash`).
2. Wait for explicit written approval - phrases like "approved", "go ahead", "yes".
3. This applies to ALL commits - features, fixes, chores, everything.

### ⚠️ RULE #2: Spec-Driven Development for Non-Trivial Features

New features follow this order:

```
1. spec folder        → specs/<feature>/{requirements,design,tasks}.md
2. TDD loop           → failing test → implement → green (per tasks.md)
3. code-review skill  → checks spec presence, task completion, conventions
```

Bug fixes (single file, no new behavior) and trivial config changes need no spec.

---

## For New Agents: Start Here

1. Read **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)** ⭐ MANDATORY
2. Read the module map below to find where your change belongs
3. Run `make test` before and after your change (it must stay green, fully offline)

---

## Architecture Map

```
src/distiller/
├── config.py            # pydantic-settings (env DISTILLER_* + distiller.toml)
├── exceptions.py        # DistillerError hierarchy
├── models.py            # pydantic domain models (BookDocument, Chunk, Answer, ...)
├── paths.py             # artifacts/<book-id>/ layout
├── optional_deps.py     # dynamic loading of extra-gated dependencies
├── ingest/              # book -> BookDocument (epub, pdf backends, markdown, text)
├── chunking/            # BookDocument -> chunks (structure-aware, chapter-bounded)
├── indexing/            # chunks -> vectors (embedders, numpy/qdrant stores, BM25)
├── rag/                 # retriever, reranker, prompts, generator, QA pipeline
├── evaluation/          # golden sets, deterministic metrics, RAGAS runner
└── cli/                 # Typer commands (thin) + context/render helpers
```

Layering: `ingest → chunking → indexing → rag → evaluation`, wired by `cli/`.
Cross-package access goes through each package's `__init__`.

---

## Commands

| Command | Purpose |
|---------|---------|
| `make install` | Install project + dev tools (`uv sync`) |
| `make install-all` | Install with all extras (docling, torch, qdrant, ragas) |
| `make format` | Format, typecheck, lint and security-check |
| `make unit-test` | Unit tests only, parallel |
| `make test` | All tests (unit + integration), fully offline |
| `make coverage` | Coverage report (fails under 80%) |
| `make check` | Lint + typecheck + tests (CI gate) |

---

## Roadmap

| Phase | Scope | State |
|-------|-------|-------|
| v1 | Ingest → chunk → hybrid index → cited answers → eval harness | ✅ done |
| Phase 1 | Contextual chunk enrichment, reranking ablations | planned |
| Phase 2 | Synthetic QA + RAFT dataset generation (cloud teacher) | planned |
| Phase 3 | Qwen3-4B QLoRA on a free T4 + eval vs baseline | planned |
| Phase 4 | GGUF export + CPU/GPU-hybrid serving (llama.cpp/Ollama) | planned |
| Phase 5 | LightRAG/RAPTOR thematic layer (whole-book questions) | planned |
