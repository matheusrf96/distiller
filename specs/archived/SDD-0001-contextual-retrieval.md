---
id: SDD-0001
status: done
supersedes:
owner: matheus
created: 2026-09-30
archived: 2026-09-30
---

# Contextual Retrieval (Phase 1)

## Problem

Retrieval over a single book is limited by chunks that are ambiguous out of
context — a chunk reading "it was painted white in the spring" carries no clue
that it is about a lighthouse tower, a chapter, or even which book it belongs
to. Anthropic's contextual retrieval showed that prepending a short LLM-written
situation to each chunk before indexing materially improves recall. This spec
adds that enrichment step to distiller as an opt-in, cached, index-only stage.

## Context

- **Chunking pipeline**: `src/distiller/chunking/structural.py` produces
  chapter-bounded chunks with heading paths and page ranges.
- **Indexing**: `src/distiller/indexing/bundle.py::build_index` embeds
  `Chunk.text` and writes `chunks.jsonl`; `load_index` builds BM25 from the same
  text.
- **LLM abstraction**: `src/distiller/llm/` — any OpenAI-compatible endpoint,
  plus a deterministic `FakeLLM` used by every offline test.
- **Related specs**: none. SDD-0002 (reranking ablations) measures retrieval
  configurations and consumes this feature's metadata flag.

### Data model deltas

| Model / field | Kind | Notes |
|---|---|---|
| `Chunk.context: str \| None` | pydantic field | LLM-generated situation; persisted in `chunks.jsonl` (old rows validate with `None`) |
| `Chunk.index_text` | property | `f"{context}\n\n{text}"` when a context exists, else `text`; used only for embedding and BM25 |

### Module deltas

```
src/distiller/enrichment/
├── __init__.py       # exports ContextualEnricher
└── contextual.py     # prompt, cache, normalization, failure tolerance
```

### Configuration

`EnrichmentSettings(enabled=False, max_document_chars=6000, max_context_chars=500)`
under `Settings.enrichment`; environment `DISTILLER_ENRICHMENT__ENABLED=true`.

### CLI

`distiller index <book> --contextual/--no-contextual` (tri-state; default comes
from configuration). `distiller eval` gains an `index` block in `report.json`
(`embedder`, `contextual`, `enriched_chunks`) so enriched and baseline runs are
comparable.

### Failure modes

| Failure | Behaviour |
|---|---|
| LLM error/timeout for one chunk | Warning; that chunk is indexed without context; the run continues |
| Completion is refusal-like or shorter than 10 chars | Discarded; chunk indexed without context |
| Cache file corrupt/unreadable | Warning; treated as empty and regenerated |
| Enrichment disabled | Identical to the pre-feature index: no contexts, no LLM calls |

### Design decisions

- **Index-only contexts.** The synthetic situation never reaches prompts,
  answers or citations — only `index_text` — so the model cannot quote the
  summary instead of the book.
- **Cache by chunk id.** The id embeds a content hash, so re-chunking
  invalidates safely and re-indexing costs zero LLM calls.
- **One call per chunk.** Failures stay isolated and prompts stay simple;
  batching is deferred until books grow.

## Requirements

1. **REQ-CR-001** — WHEN contextual enrichment is enabled and an index is built,
   THEN the system SHALL generate a short situating context for every chunk
   using the configured LLM, and SHALL NOT modify `Chunk.text`.
2. **REQ-CR-002** — WHEN embedding and lexical indexing a book, THEN the system
   SHALL index `Chunk.index_text` for chunks that carry a context, leaving
   context-free chunks unchanged.
3. **REQ-CR-003** — WHEN a context for a chunk id already exists in the
   enrichment cache, THEN the system SHALL reuse it and SHALL NOT call the LLM
   for that chunk.
4. **REQ-CR-004** — WHEN the LLM call for a chunk fails, THEN the system SHALL
   continue indexing that chunk without context and log a warning while the
   rest of the run completes.
5. **REQ-CR-005** — WHEN enrichment is disabled (the default), THEN the system
   SHALL produce the same index content as before this feature (no contexts, no
   LLM calls).
6. **REQ-CR-006** — WHEN an answer is generated, THEN the system SHALL pass the
   original `Chunk.text` to the model, never the synthetic context.
7. **REQ-CR-007** — WHEN an index build completes, THEN the system SHALL record
   in `index/metadata.json` whether enrichment was enabled and how many chunks
   received a context.
8. **REQ-CR-008** — WHEN the CLI is used, THEN the system SHALL allow enabling
   or disabling enrichment for a single index run via
   `--contextual/--no-contextual`.
9. **REQ-CR-009** — WHEN a generated context is empty, shorter than a minimum,
   refusal-like, or longer than the configured cap, THEN the system SHALL
   normalize it (collapse whitespace, cap length) or discard it.
10. **REQ-CR-010** — WHEN generating context, THEN the system SHALL provide the
    model the book title, chapter title, heading path and a bounded excerpt of
    the chapter for situating.
11. **REQ-CR-011** — WHEN `distiller eval` writes `report.json`, THEN the system
    SHALL include the index's embedder identity and contextual flag so enriched
    and baseline runs are comparable.
12. **REQ-CR-012** — WHEN the configured LLM is the offline fake, THEN the
    system SHALL still complete an enriched index build end to end.

## Acceptance Criteria

- **AC1** (Req 1) Enriched chunks carry `context` while `text` is unchanged, and
  the cache file records every generated context.
- **AC2** (Req 2) The dense embedder receives `index_text` (context prefix
  present) and BM25 matches a token that only exists in a context.
- **AC3** (Req 3) A second enrichment pass over the same chunks performs zero
  LLM calls and still yields contexts.
- **AC4** (Req 4) One failing chunk is skipped with a warning; the other chunks
  are enriched and the run completes.
- **AC5** (Req 5) A default build reports `contextual: false`,
  `enriched_chunks: 0`, and `chunks.jsonl` contains no contexts.
- **AC6** (Req 6) The CLI answer quotes original book text; the synthetic
  context never appears in the output.
- **AC7** (Req 7) `index/metadata.json` records `contextual` and
  `enriched_chunks`.
- **AC8** (Req 8) `distiller index --contextual` sets `contextual: true` and
  `--no-contextual` resets it to `false` with zero enriched chunks.
- **AC9** (Req 9) Whitespace is collapsed, too-short and refusal-like
  completions are discarded, and over-long contexts are capped.
- **AC10** (Req 10) The enrichment prompt contains the book title, chapter,
  section and the chunk text.
- **AC11** (Req 11) `eval/report.json` includes an `index` block with the
  embedder identity and contextual flag.
- **AC12** (Req 12) The offline end-to-end test completes an enriched index
  build with the fake LLM.

## Non-Goals

- Reranking or any retrieval-time change (SDD-0002).
- Batching or concurrent LLM calls for enrichment.
- Enriching with summaries longer than the configured cap.
- Applying contexts to generation prompts or citations.
- Graph/RAPTOR indexes (Phase 5).

## File-change Plan

| Action | Path | Purpose |
|--------|------|---------|
| create | `src/distiller/enrichment/__init__.py` | Package exports |
| create | `src/distiller/enrichment/contextual.py` | `ContextualEnricher`: prompt, cache, normalization, tolerance |
| modify | `src/distiller/models.py` | `Chunk.context` + `Chunk.index_text` |
| modify | `src/distiller/config.py` | `EnrichmentSettings` + `Settings.enrichment` |
| modify | `src/distiller/paths.py` | `enrichment_jsonl` artifact path |
| modify | `src/distiller/llm/fake.py` | Deterministic `<excerpt>` branch |
| modify | `src/distiller/indexing/bundle.py` | Enrichment wiring, `index_text` for dense + BM25, metadata flags |
| modify | `src/distiller/cli/main.py` | `--contextual/--no-contextual`; `index` block in eval report |
| create | `tests/unit/test_contextual_enrichment.py` | Enricher unit tests |
| create | `tests/unit/test_contextual_indexing.py` | Indexing integration tests |
| modify | `tests/integration/test_cli_end_to_end.py` | CLI end-to-end coverage |
| modify | `README.md`, `AGENTS.md` | Document the feature and roadmap status |

## Test Plan

| AC | Test file | Test name |
|----|-----------|-----------|
| AC1 | `tests/unit/test_contextual_enrichment.py` | `test_enricher_generates_and_persists_contexts` |
| AC1 | `tests/unit/test_contextual_enrichment.py` | `test_index_text_appends_context_only_when_present` |
| AC2 | `tests/unit/test_contextual_indexing.py` | `test_contextual_index_embeds_index_text` |
| AC2 | `tests/unit/test_contextual_indexing.py` | `test_contextual_bm25_matches_context_only_tokens` |
| AC3 | `tests/unit/test_contextual_enrichment.py` | `test_enricher_reuses_cache_without_calling_the_llm` |
| AC4 | `tests/unit/test_contextual_enrichment.py` | `test_enricher_tolerates_llm_failures` |
| AC5 | `tests/unit/test_contextual_indexing.py` | `test_disabled_enrichment_indexes_original_text` |
| AC6 | `tests/integration/test_cli_end_to_end.py` | `test_contextual_index_end_to_end` |
| AC7 | `tests/unit/test_contextual_indexing.py` | `test_contextual_index_embeds_index_text` |
| AC8 | `tests/integration/test_cli_end_to_end.py` | `test_contextual_index_end_to_end` |
| AC9 | `tests/unit/test_contextual_enrichment.py` | `test_enricher_normalizes_and_discards` |
| AC9 | `tests/unit/test_contextual_enrichment.py` | `test_enricher_caps_overlong_contexts` |
| AC10 | `tests/unit/test_contextual_enrichment.py` | `test_prompt_carries_book_chapter_and_section` |
| AC11 | `tests/integration/test_cli_end_to_end.py` | `test_contextual_index_end_to_end` |
| AC12 | `tests/unit/test_contextual_indexing.py` | `test_fake_llm_answers_excerpt_prompts` |

## Open Questions

- Decision (2026-09-30): contexts are capped at 500 characters by default;
  raising it is a config change, not a spec change.
- Decision (2026-09-30): one LLM call per chunk. If a book's generation cost
  becomes a problem, batching gets its own spec.

## Outcomes

- **Implemented in**: `189aad6` (spec), `eee62fc` (enrichment), `0a45588` (indexing), `f271ba0` (CLI), `0bc20a7` (docs)
- **Spec archived**: 2026-09-30
- **Post-mortem notes**: the cache made re-indexing free, confirmed by the smoke
  test. The `extra="forbid"` policy on `DomainModel` caught a stale constructor
  kwarg during the rename pass — a silent bug the tests would have missed.
