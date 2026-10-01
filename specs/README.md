# Spec-Driven Development — Workflow Guide

This directory houses all feature **specifications** for the distiller project.
Every feature, refactor, or non-trivial change starts here — from acceptance-criteria
definition through implementation and archival.

## Lifecycle

```
draft ──(approve)──> active ──(implement)──> archived
  │  │                  │                        │
  │  │ (rejected)       │ (done / superseded)    │
  │  ▼                  ▼                        │
  │ drafts/          archived/                   │
  │                                               │
  └─ (still draft, waiting for review) ──────────┘
```

| Status | Directory | Description |
|--------|-----------|-------------|
| `draft` | `specs/drafts/` | Being written, not yet implementable |
| `approved` | `specs/active/` | Reviewed and accepted; ready to implement |
| `implementing` | `specs/active/` | Work in progress by `feature-implementer` |
| `done` | `specs/archived/` | Fully implemented, all ACs verified |
| `superseded` | `specs/archived/` | Replaced by a newer spec |

## Index

| ID | Spec | Status | Summary |
|----|------|--------|---------|
| [SDD-0001](archived/SDD-0001-contextual-retrieval.md) | Contextual Retrieval | ✅ done | LLM-generated situating context per chunk for embedding/BM25 (Phase 1) |
| [SDD-0002](archived/SDD-0002-reranking-ablations.md) | Reranking Ablations | ✅ done | Compare retrieval configs (rerank on/off, top-k sweep) on one golden set (Phase 1) |
| [SDD-0003](archived/SDD-0003-synthetic-qa.md) | Synthetic QA + RAFT Dataset | ✅ done | Grounded QA generation, deterministic filtering, RAFT training examples (Phase 2) |
| [SDD-0004](archived/SDD-0004-qwen3-qlora-training.md) | Qwen3-4B QLoRA Training | ✅ done | Phase 3 harness: RAFT→chat formatting, QLoRA config, T4 runbook, adapter registry, base-vs-adapter eval |
| [SDD-0005](active/SDD-0005-gguf-serving.md) | GGUF Serving | 🔧 approved | Phase 4: GGUF validation + registration, Ollama Modelfile, llama.cpp partial-offload serving, served-model eval |

## File naming

```
specs/{{status}}/SDD-NNNN-short-kebab-description.md
```

- `SDD-NNNN` — sequential ID (see next section)
- No two specs share an ID, even after archival.

## ID assignment

All specs are numbered sequentially from the highest existing ID + 1.
To find the next ID:

```bash
ls specs/drafts/ specs/active/ specs/archived/ 2>/dev/null \
  | grep -oP 'SDD-\K\d+' | sort -n | tail -1
```

If empty, start at `SDD-0001`.

## Workflow steps (for AI agents)

1. **Author** — Use the `spec-author` agent to produce a draft in `specs/drafts/`
   following `TEMPLATE.md`. It gathers requirements from user intent and codebase
   exploration.
2. **Review** — Use the `spec-reviewer` agent. It validates front-matter, EARS
   keyword usage, numbered testable acceptance criteria, non-goals, file change
   plan and test plan. Returns a pass/fail checklist.
3. **Approve** — The user reviews the report and gives an explicit "approved" signal.
4. **Promote** — Move the file to `specs/active/`, update `status` to `approved`.
5. **Implement** — `feature-implementer` reads the active spec and implements it
   test-first. It prints an AC-coverage report on completion.
6. **Archive** — Set `status` to `done` (or `superseded`), move the file to
   `specs/archived/`, fill the `Outcomes` section with commit SHAs.

## Template

Every spec MUST use `specs/TEMPLATE.md`. Key requirements:

- **Front-matter** (YAML): `id`, `status`, `supersedes`, `owner`, `created`
  (+ `archived` when archived)
- **Requirements** in EARS notation using the canonical keywords:
  - Trigger keywords: `WHEN`, `IF`/`UNLESS`, `WHILE`, `WHERE`, `AFTER`, `BEFORE`, `UNTIL`
  - Modal keywords: `SHALL` (mandatory), `MUST` (hard), `SHOULD` (nice-to-have)
  - The word `THEN` is part of the EARS pattern and is always allowed after a trigger keyword.
  - Requirements are numbered `1..N` and also carry a stable id
    (`**REQ-<FEATURE>-NNN**`) so tests can reference them — a distiller addition
    for traceability into the Python test suite.
- **Acceptance Criteria** numbered `AC1`…`ACn`, each testable and referencing the
  requirement it proves: `AC1 (Req 1) …`
- **Non-Goals** section
- **File-change plan** (table)
- **Test plan** mapping each AC → test file + test name
- **Outcomes** section, filled on archival

## Validation

`make spec-check` (part of `make check`) enforces this guide:

- the three lifecycle directories exist and every spec sits in the directory its
  `status` requires;
- front-matter is complete and the `id` matches the file name;
- ids are unique; every spec is linked from this README;
- all required sections are present, ACs are sequential and reference existing
  requirements, and every AC appears in the Test Plan;
- `done` specs have a filled `Outcomes` section.

## Commit convention

Include the spec ID in every related commit subject:

```
feat(synthesis): SDD-0003 add grounded QA generation
fix(enrichment): SDD-0001 handle corrupt cache
test(ablation): SDD-0002 verify AC1-AC3
```

This creates traceability from commit history back to the spec.

## PR template

Every PR must reference its spec ID and check which acceptance criteria it
satisfies (see `.github/pull_request_template.md`).
