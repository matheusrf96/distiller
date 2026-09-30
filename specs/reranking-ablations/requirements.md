# Requirements — reranking-ablations

## Context

Phase 1 shipped contextual retrieval; the other half is being able to *measure*
retrieval choices. Today `distiller eval` runs one configuration and the
`--rerank` flag exists only on `ask`. This feature adds an ablation command that
runs the same golden set through several retrieval configurations (reranking
on/off, `top_k_final` sweeps) and writes one comparable report, so decisions like
"is the cross-encoder worth it on this book?" are data-driven.

---

## Functional Requirements

Each requirement uses EARS (Easy Approach to Requirements Syntax):

    WHEN <trigger>
    THE SYSTEM SHALL <response>
    WHILE <precondition / invariant>

### RERANKING-ABLATIONS

- **REQ-RA-001**: WHEN `distiller ablation` runs THE SYSTEM SHALL evaluate every requested variant against the same golden set and write one comparison report.
- **REQ-RA-002**: WHEN no `top_k` sweep is requested THE SYSTEM SHALL run two variants in order: `hybrid` (baseline, no reranking) and `hybrid+rerank`.
- **REQ-RA-003**: WHEN `--top-k` provides a comma-separated list THE SYSTEM SHALL add a variant per value for both reranking states, named with the value (e.g. `hybrid-k4`, `hybrid+rerank-k4`), keeping the baseline first.
- **REQ-RA-004**: WHEN the reranker dependency is unavailable THE SYSTEM SHALL record rerank variants as skipped with an actionable reason and still run the remaining variants.
- **REQ-RA-005**: WHEN variants run THE SYSTEM SHALL reuse the already-loaded index and SHALL NOT mutate global settings.
- **REQ-RA-006**: WHEN the ablation completes THE SYSTEM SHALL write `eval/ablation.json` containing the index identity, item count, baseline variant name and per-variant metrics plus per-item results.
- **REQ-RA-007**: WHEN the CLI prints the ablation THE SYSTEM SHALL show a table with per-variant metrics and deltas versus the baseline variant.
- **REQ-RA-008**: WHEN `--limit N` is given THE SYSTEM SHALL evaluate the same first N golden items in every variant.
- **REQ-RA-009**: WHEN `--no-rerank` is given THE SYSTEM SHALL run only non-reranked variants.
- **REQ-RA-010**: WHEN `--json` is given THE SYSTEM SHALL print the full report as JSON instead of the table.
- **REQ-RA-011**: WHEN a variant raises during its run THE SYSTEM SHALL record the failure as that variant's skip reason without aborting the others.
- **REQ-RA-012**: WHEN the ablation runs THE SYSTEM SHALL reuse the deterministic metric suite (`summarize`) so numbers are directly comparable with `distiller eval` runs.

---

## Constraints / Non-Functional

- **CON-RA-001**: The default ablation (baseline only, reranking unavailable) requires no optional dependencies.
- **CON-RA-002**: Fully testable offline: tests inject stub answer functions and a stub reranker; CI has no sentence-transformers.
- **CON-RA-003**: Variants are validated pydantic models; configuration is copied via the existing validated-override helper (no mutation of `settings.retrieval`).
- **CON-RA-004**: `mypy --strict` clean, 88-char lines, coverage gate (80%) stays green.
- **CON-RA-005**: The report carries the index identity (embedder, contextual flag) so ablations across different indexes remain distinguishable.

---

## Out of Scope

- Automatic winner selection or statistical significance testing.
- Sweeps over chunking or contextual enrichment (index-time knobs: those need
  separate index builds; `index/metadata.json` already makes such runs comparable).
- LLM-judged metrics (RAGAS stays opt-in via `distiller eval --ragas`).
- GPU/batching optimisation of the reranker.
