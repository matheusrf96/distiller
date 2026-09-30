# Design — contextual-retrieval

## Summary

Add an opt-in enrichment step between chunking and indexing that asks the
configured LLM for a 1–2 sentence situating context per chunk. The context is
used **only** for embedding and BM25 indexing (`Chunk.index_text`); prompts,
answers and citations keep using the original `Chunk.text`. Contexts are cached
per book so re-indexing is free after the first run.

## Data model deltas (`src/distiller/models.py`)

| Field | Type | Notes |
|---|---|---|
| `Chunk.context` | `str \| None = None` | LLM-generated situating context; persisted in `chunks.jsonl`. |
| `Chunk.index_text` | property | `f"{context}\n\n{text}"` when a context exists, else `text`. Used by embedder and BM25 only. |

Backward compatibility: existing `chunks.jsonl` rows have no `context` key and
validate with the default `None`. Artifacts remain readable; only new fields are
added.

## New module (`src/distiller/enrichment/`)

```
enrichment/
├── __init__.py
└── contextual.py     # ContextualEnricher + prompt templates + normalization
```

### `ContextualEnricher`

```python
class ContextualEnricher:
    def __init__(
        self,
        llm: LLMClient,
        book: BookDocument,
        *,
        cache_path: Path | None = None,
        max_document_chars: int = 6000,
        max_context_chars: int = 500,
        min_context_chars: int = 20,
    ) -> None: ...

    def enrich(self, chunks: list[Chunk]) -> list[Chunk]:
        """Return chunk copies with `context` filled, reusing the on-disk cache."""
```

Behaviour:

1. Loads the cache (`enrichment.jsonl`, rows `{"chunk_id": str, "context": str}`)
   once per run; cache hits skip the LLM entirely (REQ-CR-003).
2. Builds one truncated chapter excerpt per chapter at construction time
   (REQ-CR-010).
3. For each miss, calls `llm.complete(system=..., user=..., temperature=0.0,
   max_tokens=200)`; on any exception logs a warning and leaves the chunk
   without context (REQ-CR-004).
4. Normalizes the completion: collapse whitespace, discard if shorter than
   `min_context_chars`, refusal-like, or empty; truncate to `max_context_chars`
   (REQ-CR-009).
5. Appends new entries to the cache file immediately after generation so an
   interrupted run keeps its paid work (CON-CR-002).

Noise sources deliberately excluded from context generation: chapter
excerpts are capped by `max_document_chars`; the locator (chapter + heading
path) is passed as structured fields, not raw markdown.

## Indexing integration (`src/distiller/indexing/bundle.py`)

```
build_index:
    chunks = chunk_document(book, settings.chunking)
    if settings.enrichment.enabled:
        chunks = ContextualEnricher(get_llm(settings), book, cache_path=paths.enrichment_jsonl,
                                    max_document_chars=..., max_context_chars=...).enrich(chunks)
    write_jsonl(paths.chunks_jsonl, chunks)
    vectors = embedder.embed_documents([chunk.index_text for chunk in chunks])
    store.upsert(...)                       # payload unchanged
    metadata["contextual"] = settings.enrichment.enabled
    metadata["enriched_chunks"] = sum(1 for c in chunks if c.context)

load_index:
    bm25 = BM25Index(ids, [chunk.index_text for chunk in chunks])   # same text as dense side
```

`BookPaths` gains `enrichment_jsonl` (`artifacts/<book>/enrichment.jsonl`).

## Configuration (`src/distiller/config.py`)

```python
class EnrichmentSettings(BaseModel):
    enabled: bool = False
    max_document_chars: int = 6000
    max_context_chars: int = 500
```

`Settings.enrichment: EnrichmentSettings`. Environment:
`DISTILLER_ENRICHMENT__ENABLED=true`.

## CLI (`src/distiller/cli/main.py`)

- `index` gains `--contextual/--no-contextual` (tri-state; `None` = use
  configuration) applied through the existing validated-override helper.
- `eval` adds an `index` block to `report.json`:
  `{"embedder": ..., "contextual": ...}` (REQ-CR-011).

## Fake LLM (`src/distiller/llm/fake.py`)

Answer `<excerpt>` blocks with a deterministic short sentence so the offline
end-to-end path can exercise enrichment (REQ-CR-012). Existing `<doc>` and
refusal behaviour is unchanged.

## Failure modes

| Failure | Behaviour |
|---|---|
| LLM error / timeout for one chunk | Warning, chunk indexed without context, run continues |
| Completion is refusal-like or tiny | Discarded; chunk indexed without context |
| Cache file corrupt/unreadable | Warning; cache treated as empty (regenerates) |
| Enrichment disabled | Identical to v1 indexing (no LLM calls, no contexts) |

## Sequence (enabled path)

```
cli.index --contextual
  → build_index
      → chunk_document
      → ContextualEnricher.enrich        (cache hits: 0 LLM calls)
          → llm.complete (misses only)   →  enrichment.jsonl append
      → write chunks.jsonl (with context)
      → embed index_text  + BM25 index_text
      → metadata.json {contextual: true, enriched_chunks: N}
ask → retriever (unchanged) → generator (original text only)
```

## Verification approach

- Unit: normalization rules, cache reuse (LLM call counting stub), failure
  tolerance (caplog), prompt contents, `index_text`.
- Unit (indexing): recording embedder proves `index_text` reaches the dense
  side; `chunks.jsonl` and metadata prove persistence.
- Integration: CLI `ingest → index --contextual → ask` with the fake LLM;
  eval report carries the index block.

## ADR

- **Why not store the context as part of `Chunk.text`?** Answers and citations
  must remain quotable book text; mixing synthetic prose into prompts invites
  the model to cite the summary instead of the book.
- **Why per-chunk calls and not batched prompts?** One call per chunk keeps
  failures isolated and prompts simple; batching is a later optimisation
  (out of scope).
