---
id: SDD-0002
status: done
supersedes:
owner: matheus
created: 2026-09-30
archived: 2026-09-30
---

# Reranking Ablations (Phase 1)

## Problem

`distiller eval` runs one retrieval configuration, and the cross-encoder
reranker is only reachable through `ask --rerank`. There is no way to answer
"is reranking worth it on this book?" with data. This spec adds an ablation
command that runs the same golden set through several retrieval configurations
(reranking on/off, `top_k_final` sweeps) and writes one comparable report.

## Context

- **Retrieval**: `src/distiller/rag/retriever.py` (hybrid dense + BM25 with RRF)
  and `src/distiller/rag/reranker.py` (cross-encoder, needs the `embed` extra).
- **Evaluation**: `src/distiller/evaluation/metrics.py` (`evaluate_item`,
  `summarize`) and the `eval` command, which already writes an `index` identity
  block into `report.json`.
- **Related specs**: SDD-0001 (contextual retrieval) records the `contextual`
  flag this spec copies into ablation reports; SDD-0003 (synthetic QA) reuses
  the same golden-set plumbing.

### Data model deltas

| Model / field | Kind | Notes |
|---|---|---|
| `AblationVariant` | pydantic (`DomainModel`) | `name`, `rerank`, `top_k_final`, `rerank_pool` |
| `VariantResult` | pydantic | `variant`, `skipped_reason`, `metrics`, `items: list[ItemResult]` |
| `AblationReport` | pydantic | `book_id`, `index`, `item_count`, `baseline`, `variants` |
| `AblationRun` | dataclass (runtime only) | `variant`, `answer: Callable \| None`, `skipped_reason` |

`AblationRun` is a dataclass because it carries a callable; everything that is
serialized is pydantic, matching the project's model policy.

### Module deltas

```
src/distiller/evaluation/
├── ablation.py       # variants, runner, deltas  (new)
└── __init__.py       # re-exports the new symbols
```

### CLI

`distiller ablation <book> [--golden] [--limit] [--top-k 4,8,12] [--no-rerank]
[--json]` — loads the index once, builds one pipeline per variant, writes
`eval/ablation.json` and prints a table with deltas versus the baseline.
`cli/context.py` gains `parse_top_k_values` and `build_ablation_runs`.

### Failure modes

| Failure | Behaviour |
|---|---|
| sentence-transformers missing | Rerank variants recorded as skipped with the extra hint; baseline still runs |
| One variant raises mid-run | That variant records the failure as its skip reason; others continue |
| Invalid `--top-k` input | `typer.BadParameter` before any work happens |
| Golden set missing/malformed | Existing friendly-error helper (`load_golden_set`) |

### Design decisions

- **Separate command, not `eval --ablate`.** `eval` answers "how good is this
  index?"; `ablation` answers "which retrieval config is better?" — different
  outputs (single report vs comparison matrix).
- **No automatic winner.** With small golden sets the deltas are noisy; the
  report presents numbers and leaves judgement to the maintainer.
- **Reuse the loaded index and copied settings.** Variants never mutate
  `settings.retrieval`; overrides go through the validated-override helper.

## Requirements

1. **REQ-RA-001** — WHEN `distiller ablation` runs, THEN the system SHALL
   evaluate every requested variant against the same golden set and write one
   comparison report.
2. **REQ-RA-002** — WHEN no `top-k` sweep is requested, THEN the system SHALL
   run two variants in order: `hybrid` (baseline) and `hybrid+rerank`.
3. **REQ-RA-003** — WHEN `--top-k` provides a comma-separated list, THEN the
   system SHALL add a variant per value for both reranking states, named with
   the value, keeping the baseline first.
4. **REQ-RA-004** — WHEN the reranker dependency is unavailable, THEN the
   system SHALL record rerank variants as skipped with an actionable reason and
   still run the remaining variants.
5. **REQ-RA-005** — WHEN variants run, THEN the system SHALL reuse the
   already-loaded index and SHALL NOT mutate global settings.
6. **REQ-RA-006** — WHEN the ablation completes, THEN the system SHALL write
   `eval/ablation.json` containing the index identity, item count, baseline
   name and per-variant metrics plus per-item results.
7. **REQ-RA-007** — WHEN the CLI prints the ablation, THEN the system SHALL
   show a table with per-variant metrics and deltas versus the baseline variant.
8. **REQ-RA-008** — WHEN `--limit N` is given, THEN the system SHALL evaluate
   the same first N golden items in every variant.
9. **REQ-RA-009** — WHEN `--no-rerank` is given, THEN the system SHALL run only
   non-reranked variants.
10. **REQ-RA-010** — WHEN `--json` is given, THEN the system SHALL print the
    full report as JSON instead of the table.
11. **REQ-RA-011** — WHEN a variant raises during its run, THEN the system SHALL
    record the failure as that variant's skip reason without aborting the others.
12. **REQ-RA-012** — WHEN the ablation runs, THEN the system SHALL reuse the
    deterministic metric suite so numbers are directly comparable with
    `distiller eval` runs.

## Acceptance Criteria

- **AC1** (Req 1) The end-to-end ablation runs every requested variant and
  writes one report.
- **AC2** (Req 2) The default matrix is `hybrid` then `hybrid+rerank`, baseline
  first.
- **AC3** (Req 3) `--top-k 4,12` produces `hybrid-k4`, `hybrid+rerank-k4`,
  `hybrid-k12`, `hybrid+rerank-k12`; the parser accepts spaces and rejects
  garbage or non-positive values.
- **AC4** (Req 4) Without the `embed` extra the rerank variant is recorded as
  skipped with the install hint and the baseline still produces metrics.
- **AC5** (Req 5) `settings.retrieval` is unchanged after building runs, and
  skipped variants carry their reason.
- **AC6** (Req 6) The report round-trips through JSON and the CLI writes
  `eval/ablation.json`.
- **AC7** (Req 7) Deltas are computed versus the first non-skipped variant;
  skipped variants are excluded; the table renders a Δ column.
- **AC8** (Req 8) `--limit 1` yields `item_count == 1` in the report and in
  every variant's metrics.
- **AC9** (Req 9) `include_rerank=False` yields only the non-reranked variant.
- **AC10** (Req 10) `--json` output parses and equals the written report's
  variants.
- **AC11** (Req 11) A raising answer callable becomes that variant's skip
  reason; the other variants still produce metrics.
- **AC12** (Req 12) Variant metrics equal `summarize` output over the same
  golden items.

## Non-Goals

- Automatic winner selection or statistical significance testing.
- Sweeps over chunking or contextual enrichment (index-time knobs need
  separate index builds).
- LLM-judged metrics (RAGAS stays opt-in via `distiller eval --ragas`).
- GPU/batching optimisation of the reranker.

## File-change Plan

| Action | Path | Purpose |
|--------|------|---------|
| create | `src/distiller/evaluation/ablation.py` | Variants, runner, metric deltas |
| modify | `src/distiller/evaluation/__init__.py` | Re-export the ablation API |
| modify | `src/distiller/cli/context.py` | `parse_top_k_values`, `build_ablation_runs` |
| modify | `src/distiller/cli/main.py` | The `ablation` command |
| modify | `src/distiller/cli/render.py` | `render_ablation` table with deltas |
| create | `tests/unit/test_ablation.py` | Harness unit tests |
| modify | `tests/integration/test_cli_end_to_end.py` | CLI ablation coverage |
| modify | `README.md`, `AGENTS.md` | Document the command and roadmap status |

## Test Plan

| AC | Test file | Test name |
|----|-----------|-----------|
| AC1 | `tests/integration/test_cli_end_to_end.py` | `test_ablation_command_runs_rerank_variants_with_sweep` |
| AC2 | `tests/unit/test_ablation.py` | `test_build_variants_defaults` |
| AC3 | `tests/unit/test_ablation.py` | `test_build_variants_sweep_and_no_rerank` |
| AC3 | `tests/unit/test_ablation.py` | `test_parse_top_k_values` |
| AC4 | `tests/integration/test_cli_end_to_end.py` | `test_ablation_command_skips_missing_reranker` |
| AC4 | `tests/unit/test_ablation.py` | `test_run_ablation_records_metrics_skips_and_failures` |
| AC5 | `tests/unit/test_ablation.py` | `test_build_ablation_runs_does_not_mutate_settings` |
| AC6 | `tests/unit/test_ablation.py` | `test_report_round_trips_through_json` |
| AC6 | `tests/integration/test_cli_end_to_end.py` | `test_ablation_command_skips_missing_reranker` |
| AC7 | `tests/unit/test_ablation.py` | `test_metric_deltas_against_baseline` |
| AC7 | `tests/unit/test_ablation.py` | `test_metric_deltas_without_a_baseline_is_empty` |
| AC8 | `tests/integration/test_cli_end_to_end.py` | `test_ablation_command_honours_limit` |
| AC9 | `tests/unit/test_ablation.py` | `test_build_variants_sweep_and_no_rerank` |
| AC10 | `tests/integration/test_cli_end_to_end.py` | `test_ablation_command_runs_rerank_variants_with_sweep` |
| AC11 | `tests/unit/test_ablation.py` | `test_run_ablation_records_metrics_skips_and_failures` |
| AC12 | `tests/unit/test_ablation.py` | `test_run_ablation_records_metrics_skips_and_failures` |

## Open Questions

- Decision (2026-09-30): variant names carry the top-k suffix only when a sweep
  is requested (`hybrid` vs `hybrid-k4`), keeping default reports readable.
- Decision (2026-09-30): the report is overwritten on each run, mirroring
  `eval/report.json`; comparing historical runs is out of scope.

## Outcomes

- **Implemented in**: `03e9fa7` (spec), `c73af69` (harness), `4131f1c` (CLI), `9c98246` (docs)
- **Spec archived**: 2026-09-30
- **Post-mortem notes**: the skip path (no `embed` extra) is exercised by the
  offline suite, so the command is useful on a plain install. Converting the
  spec to the SDD format exposed two coverage gaps — REQ-RA-005 (no settings
  mutation) and REQ-RA-008 (`--limit`) — which the AC table now pins to tests.
