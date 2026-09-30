# Design — synthetic-qa

## Summary

`distiller synth <book-id>` reads the book's indexed chunks, asks the configured
LLM for grounded QA pairs, filters them deterministically, and emits RAFT-style
training examples. Nothing in this phase needs an embedder; the teacher is the
same `LLMClient` abstraction used elsewhere (cloud API or local server).

## New package (`src/distiller/synthesis/`)

```
synthesis/
├── __init__.py
├── qa.py          # QAPair, prompts, generation + JSON parsing
├── filtering.py   # RejectedPair, FilterOutcome, deterministic filters
├── raft.py        # RaftContext, RaftExample, example builder
└── dataset.py     # chunk sampling, manifest, orchestration helpers
```

## Models

| Model | Module | Fields |
|---|---|---|
| `QAPair` | `qa.py` | `id`, `book_id`, `chunk_id`, `chapter`, `heading_path`, `question`, `answer`, `quotes: list[str]`, `model` |
| `RejectedPair` | `filtering.py` | `pair_id`, `question`, `reason` |
| `FilterOutcome` | `filtering.py` | `kept: list[QAPair]`, `rejected: list[RejectedPair]`, `rejection_counts()` |
| `RaftContext` | `raft.py` | `chunk_id`, `text`, `is_golden` |
| `RaftExample` | `raft.py` | `id`, `book_id`, `question`, `contexts`, `answer`, `answerable` |
| `DatasetManifest` | `dataset.py` | `book_id`, `created_at`, `model`, `source`, `index`, `config`, counts..., `rejected: dict[str, int]` |

`QAPair.id = f"{chunk_id}:{stable_hash_hex(question, 12)}"` — stable across runs,
so cached `qa.jsonl` rows and RAFT ids stay reproducible.

## Generation (`qa.py`)

- One LLM call per chunk; the prompt carries book/chapter/section plus the chunk
  in `<chunk>` tags and asks for **exactly N** questions as a JSON array
  (`question`, `answer`, `quotes`).
- `parse_pairs(text)`: `json.loads` → fallback to the first `[...]` block
  (models sometimes wrap JSON in prose or fences). Malformed items are dropped
  individually; a wholly unparseable completion is skipped with a warning
  (REQ-SQ-004).
- LLM exceptions are caught per chunk (same tolerance pattern as enrichment).

## Filtering (`filtering.py`)

`filter_pairs(pairs, *, chunks_by_id) -> FilterOutcome`, with these rules:

| Rule | Effect |
|---|---|
| quote not verbatim in the chunk (normalized whitespace, case-insensitive) | drop that quote |
| quote shorter than `MIN_QUOTE_CHARS` (15) | drop that quote |
| no quotes left | reject pair — reason `no verified quotes` |
| question shorter than 12 chars / answer shorter than 20 | reject — reason `too short` |
| normalized question already kept | reject — reason `duplicate question` |
| `SequenceMatcher(question, answer).ratio() >= 0.7` | reject — reason `question echoes answer` |

Rejections are recorded per pair so the manifest can explain the yield.

## RAFT formatting (`raft.py`)

`build_examples(book_title, pairs, chunks, *, distractors, negative_ratio, seed)`:

- `random.Random(seed)` makes everything reproducible (CON-SQ-004).
- Positive example: contexts = golden chunk + `distractors` other chunks,
  shuffled; answer =
  `The relevant passage says: "<primary quote>" [n]\n\n<answer>` where `n` is
  the golden context's 1-based position (REQ-SQ-007).
- Negative example (probability `negative_ratio` per pair): contexts = the same
  distractor pool *without* the golden chunk; answer = `refusal_text(book_title)`
  (REQ-SQ-008). The refusal sentence moves to `models.py` so the RAG prompt and
  the training target share one source of truth.
- `RaftExample.id = f"{pair.id}:raft"`.

## Orchestration and CLI

- `dataset.sample_chunks(chunks, *, max_chunks, seed)`: seeded sample, returned
  in reading order (REQ-SQ-011).
- `dataset.build_manifest(...)`: assembles the manifest from the run inputs.
- `cli.context.load_book_and_chunks(settings, book_id)`: loads `book.json` and
  `chunks.jsonl` with friendly errors (REQ-SQ-015); no embedder involved.
- `distiller synth <book-id> [--max-chunks N] [--questions-per-chunk N]
  [--distractors N] [--negative-ratio F] [--seed N] [--regenerate] [--json]`:
  applies overrides through `apply_overrides`, reuses `qa.jsonl` unless
  `--regenerate` (REQ-SQ-010), writes the four dataset files, renders the
  summary table (REQ-SQ-012/REQ-SQ-014).
- `SynthesisSettings` in `config.py`: `max_chunks=100`, `questions_per_chunk=2`,
  `distractors=4`, `negative_ratio=0.15`, `seed=13`, validated bounds.

## Flow

```
cli.synth
  → load_book_and_chunks            (book.json + chunks.jsonl; no embedder)
  → sample_chunks                   (seed, max_chunks)
  → qa.jsonl exists? ── yes ─→ reuse pairs          (source="cache")
                        └─ no ─→ generate_pairs     (one LLM call per chunk)
  → filter_pairs                    (verified quotes, dedupe, echo check)
  → build_examples                  (golden + distractors, positives + negatives)
  → write dataset/{qa,rejected,raft}.jsonl + manifest.json
  → render_synthesis                (table or --json)
```

## Failure modes

| Failure | Behaviour |
|---|---|
| Chunks missing (book not indexed) | Friendly error: run `distiller index` first |
| One chunk's completion unparseable | Skipped with a warning; the run continues |
| LLM error for one chunk | Skipped with a warning; the run continues |
| `qa.jsonl` corrupt | Treated as absent (regenerate) with a warning |
| No pairs survive filtering | Manifest and RAFT file are written with zero examples; the table says so |

## ADR

- **Why deterministic filters, not LLM judges?** Filters must be cheap,
  explainable and CI-stable; RAGAS-scored filtering is a later iteration once the
  raw yield is understood.
- **Why reuse `qa.jsonl`?** Generation costs money; the filtered and RAFT layers
  are free to iterate on without re-paying the teacher.
- **Why no embedder?** Synthesis consumes chunks, not vectors; keeping it
  embedder-free means it works on a plain install and after any re-embedding.
