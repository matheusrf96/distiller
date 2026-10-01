# distiller - Agent Guide

**Quick Links:**
- 📋 **[Contributing & Conventions](docs/CONTRIBUTING.md)** - Code and documentation rules (MANDATORY read)
- 📚 **[Documentation Index](docs/README.md)** - All project documentation
- 🏗️ **[README](README.md)** - Install, quickstart and architecture
- 🌳 **[Thematic questions](docs/thematic-questions.md)** - Summary tree, global ask/eval

## Project Overview

**distiller** turns a PDF/ePub **book** into a searchable, cited knowledge index and
(on the roadmap) into a specialized 3-4B model distilled from that book via
RAFT-style fine-tuning.

- **Stack**: Python 3.12+, pydantic/pydantic-settings, Typer CLI, NumPy/Qdrant,
  sentence-transformers (optional), ruff + mypy(strict) + pytest.
- **Status**: v1 complete (ingest → chunk → hybrid index → cited answers → eval);
  Phase 1 complete (contextual retrieval + reranking ablations), Phase 2
  complete (synthetic QA + RAFT dataset), Phase 3 harness complete (chat
  splits, QLoRA config, T4 notebook, adapter registry, base-vs-adapter eval —
  the GPU run is manual, see `docs/qlora-runbook.md`), Phase 4 harness
  complete (GGUF validation/registry, Ollama Modelfile + llama.cpp `serve.sh`,
  served-model eval — the T4 export and local server are manual, see
  `docs/gguf-runbook.md`) and Phase 5 complete (hierarchical summary tree +
  global whole-book answering, see `docs/thematic-questions.md`); fully offline
  test suite (hash embedder + fake LLM).

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

Every non-trivial feature, refactor or behaviour change follows the SDD
lifecycle documented in **[specs/README.md](specs/README.md)**:

```
draft ──(approve)──> active ──(implement)──> archived
```

1. **Author** — the `spec-author` agent writes a draft into `specs/drafts/`
   from `specs/TEMPLATE.md`; assign the next `SDD-NNNN`.
2. **Review** — the `spec-reviewer` agent validates front-matter, EARS
   requirements, testable ACs (each referencing its requirement), non-goals,
   file-change plan and test plan; it must PASS every section.
3. **Approve** — the maintainer approves explicitly; the file moves to
   `specs/active/` with `status: approved`.
   **NEVER implement before this gate.**
4. **Implement** — `feature-implementer` works from the spec, test-first, and
   prints an AC-coverage report; `status: implementing`.
5. **Archive** — all ACs covered, `make check` green, `Outcomes` filled with
   commit SHAs; the file moves to `specs/archived/` with `status: done`.

Gate: `make spec-check` runs as part of `make check`. Commit subjects carry the
spec id (e.g. `feat(synthesis): SDD-0003 add grounded QA generation`).
Trivial fixes (single file, no new behaviour) and config/typo changes are exempt.

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
├── enrichment/          # optional contextual retrieval (LLM context per chunk)
├── indexing/            # chunks -> vectors (embedders, numpy/qdrant stores, BM25)
├── rag/                 # retriever, reranker, prompts, generator, QA pipeline
├── thematic/            # Phase 5: summary tree, cached summarizer, global map-reduce
├── synthesis/           # Phase 2: QA generation, filters, RAFT dataset
├── training/            # Phase 3: chat formatting, splits, QLoRA config, registry
├── gguf/                # Phase 4: dependency-free GGUF reader, registry, serving
├── evaluation/          # golden sets, deterministic metrics, ablations, RAGAS
└── cli/                 # Typer commands (thin) + context/render helpers
```

Layering: `ingest → chunking → enrichment → indexing → rag → evaluation` plus
`thematic` (consumes the book, chunks and the LLM; answers whole-book questions
over the summary tree), `synthesis` (consumes chunks and the LLM), `training`
(consumes the RAFT dataset; the heavy training stack is the optional `training`
extra) and `gguf` (validates/registers the downloaded export and emits serving
commands), wired by `cli/`. Cross-package access goes through each package's
`__init__`; `rag` never imports `thematic`.

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
| Phase 1 | Contextual chunk enrichment + reranking ablations (`SDD-0001`, `SDD-0002`) | ✅ done |
| Phase 2 | Synthetic QA + RAFT dataset generation (`SDD-0003`) | ✅ done |
| Phase 3 | Qwen3-4B QLoRA harness + eval vs baseline (`SDD-0004`); T4 run manual | 🔧 harness done |
| Phase 4 | GGUF export + CPU/GPU-hybrid serving (`SDD-0005`); T4 export + local serving manual | 🔧 harness done |
| Phase 5 | Hierarchical summary tree + global whole-book answering (`SDD-0006`) | ✅ done |
