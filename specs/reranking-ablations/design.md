# Design — reranking-ablations

## Summary

A new `distiller ablation <book-id>` command evaluates a matrix of retrieval
configurations against one golden set, reusing the loaded index and the existing
deterministic metrics. Reranking variants are skipped with an actionable reason
when the `embed` extra is absent, so the command is useful on a plain install.

## Data model (`src/distiller/evaluation/ablation.py`)

| Model | Kind | Fields |
|---|---|---|
| `AblationVariant` | pydantic (`DomainModel`) | `name`, `rerank`, `top_k_final`, `rerank_pool` |
| `VariantResult` | pydantic (`DomainModel`) | `variant`, `skipped_reason`, `metrics`, `items: list[ItemResult]` |
| `AblationReport` | pydantic (`DomainModel`) | `book_id`, `index` (identity dict), `item_count`, `baseline`, `variants` |
| `AblationRun` | dataclass (runtime only) | `variant`, `answer: Callable[[str], Answer] | None`, `skipped_reason` |

`AblationRun` is a dataclass because it carries a callable; everything that is
serialized is pydantic (consistent with the project's model policy).

## Variant construction

```python
build_variants(
    *, top_k_final: int, rerank_pool: int,
    top_k_values: list[int] | None = None, include_rerank: bool = True,
) -> list[AblationVariant]
```

- Default (`top_k_values=None`): `hybrid` then `hybrid+rerank` (REQ-RA-002).
- With a sweep: for each `k` → `hybrid-k{k}` (+ `hybrid+rerank-k{k}` when
  reranking is included), baseline first (REQ-RA-003).
- `include_rerank=False` (`--no-rerank`): only the non-reranked variants (REQ-RA-009).

## Runner

```python
run_ablation(
    book_id: str,
    golden_items: list[GoldenItem],
    runs: list[AblationRun],
    *,
    index_identity: dict[str, Any],
) -> AblationReport
```

For each run:

- `answer is None` → `VariantResult(skipped_reason=...)` (REQ-RA-004).
- otherwise evaluate every golden item with `evaluate_item` and aggregate with
  `summarize` (REQ-RA-012); exceptions are captured as that variant's skip
  reason (REQ-RA-011).

`metric_deltas(report, metrics)` returns per-variant deltas against the first
non-skipped variant (the baseline) for rendering (REQ-RA-007).

## CLI wiring (`src/distiller/cli/`)

- `context.parse_top_k_values(raw)` — parses `"4,8,12"` into `[4, 8, 12]`,
  raising `typer.BadParameter` on garbage or non-positive values.
- `context.build_ablation_runs(settings, bundle, variants)` — pairs each variant
  with `functools.partial(pipeline.ask)`; rerank variants are skipped when
  `is_available("sentence_transformers")` is false (REQ-RA-004). Per-variant
  retrieval settings come from `apply_overrides(settings.retrieval, ...)`
  copies — no mutation (REQ-RA-005/CON-RA-003).
- `main.ablation` — command `distiller ablation <book-id> [--golden] [--limit]
  [--top-k 4,8,12] [--no-rerank] [--json]`; loads the index once, builds runs,
  writes `eval/ablation.json`, renders the table (or JSON).
- `render.render_ablation(report)` — one row per variant: config, four headline
  metrics, and Δ (versus baseline) for `retrieval_hit_rate` and `contains_rate`;
  skipped variants render as `—` with their reason listed below the table.

## Flow

```
cli.ablation
  → load_golden_set (--limit applied once, same subset for all variants)
  → load_book_index            (once; REQ-RA-005)
  → build_variants
  → build_ablation_runs        (skip rerank variants without the embed extra)
  → run_ablation               (per-variant: ask → evaluate_item → summarize)
  → write eval/ablation.json   (report + index identity; REQ-RA-006)
  → render_ablation            (table with deltas; REQ-RA-007)
```

## Failure modes

| Failure | Behaviour |
|---|---|
| sentence-transformers missing | Rerank variants recorded as skipped with the extra hint; baseline still runs |
| One variant raises | That variant records the failure as its skip reason; others continue |
| Invalid `--top-k` input | `typer.BadParameter` before any work happens |
| Golden set missing/malformed | Existing friendly-error helpers (`load_golden_set`) |

## ADR

- **Why a separate command instead of `eval --ablate`?** `eval` answers "how good
  is this index?"; `ablation` answers "which retrieval config is better?" — they
  have different outputs (single report vs comparison matrix).
- **Why no automatic winner?** With small golden sets the deltas are noisy; the
  report presents numbers and leaves judgement to the maintainer (out of scope).
