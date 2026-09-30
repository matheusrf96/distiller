# distiller

Turn a PDF or ePub **book** into a searchable knowledge index — and, on the roadmap,
into a specialized 3–4B model distilled from that book via RAFT-style fine-tuning.

The project's thesis (backed by 2026 research):

- **Retrieval is the source of truth.** Fine-tuning facts into small models makes them
  confidently wrong; retrieval over the book keeps answers grounded.
- **Small models fail at *using* context, not at *finding* it.** The real win is
  fine-tuning the model specifically for retrieval-augmented answering — cite verbatim,
  ignore distractors, refuse when the book doesn't say — which is exactly what
  [RAFT](https://arxiv.org/abs/2403.10131) does.
- **So we build both:** a rigorous RAG pipeline first (this release), then a
  distillation pipeline (synthetic QA → LoRA on Qwen3-4B) that makes a tiny model
  excellent at this one book.

## Status

| Phase | Scope | State |
| --- | --- | --- |
| **v1** | Ingest → chunk → hybrid index → cited answers → eval harness | ✅ implemented |
| Phase 1 | Contextual chunk enrichment, reranking ablations | planned |
| Phase 2 | Synthetic QA + RAFT dataset (cloud teacher) | planned |
| Phase 3 | Qwen3-4B QLoRA on free cloud GPU (T4) + eval vs baseline | planned |
| Phase 4 | GGUF export + CPU/GPU-hybrid serving (llama.cpp/Ollama) | planned |
| Phase 5 | LightRAG/RAPTOR thematic layer (whole-book questions) | planned |

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Core: ingest, index and query books (no torch, no docling)
make install

# Full: adds docling (AI PDF parsing), sentence-transformers, qdrant, ragas
make install-all
```

Heavy extras can be picked individually: `uv sync --extra pdf-ai`, `--extra embed`,
`--extra index`, `--extra eval`.

## Quickstart

```bash
# 1. Parse a book (EPUB, PDF, Markdown or plain text)
uv run distiller ingest fixtures/sherlock.epub

# 2. Chunk + embed + index it
uv run distiller index the-adventures-of-sherlock-holmes

# 3. Ask questions — answers carry citations into the book
uv run distiller ask the-adventures-of-sherlock-holmes "What does Holmes infer from the hat?"
uv run distiller ask the-adventures-of-sherlock-holmes "What colour was the tower?" --json

# 4. Evaluate against a golden question set
cp examples/golden.example.yaml artifacts/<book-id>/golden.yaml
uv run distiller eval the-adventures-of-sherlock-holmes --ragas

# List / inspect
uv run distiller books
uv run distiller info the-adventures-of-sherlock-holmes
```

Everything for one book lives in `artifacts/<book-id>/`:

```
artifacts/the-lantern-keeper/
├── book.json        # structured parse: chapters → blocks
├── parsed.md        # human-readable rendering
├── chunks.jsonl     # retrieval units with provenance (chapter, section, pages)
├── index/
│   ├── metadata.json # embedder identity, dim, store, chunk_count
│   └── store/       # vectors + records (numpy) or qdrant local db
└── eval/report.json # metrics per run
```

## Configuration

Environment variables (`DISTILLER_*`, `__` nests) or a `distiller.toml` in the
working directory. Precedence: env > toml.

```bash
# Generation backend: any OpenAI-compatible endpoint
export DISTILLER_LLM__BASE_URL=https://api.openai.com/v1   # or http://localhost:11434/v1 (Ollama)
export DISTILLER_LLM__API_KEY=sk-...
export DISTILLER_LLM__MODEL=gpt-4o-mini

# Local, no backend at all (deterministic snippet answers — useful for smoke tests)
export DISTILLER_LLM__MODEL=fake

# Embeddings
export DISTILLER_EMBEDDING__MODEL=Qwen/Qwen3-Embedding-0.6B  # default
export DISTILLER_EMBEDDING__DIM=512                          # optional MRL truncation
export DISTILLER_EMBEDDING__BACKEND=hash                     # offline test embedder

# Retrieval
export DISTILLER_RETRIEVAL__RERANK=true                      # cross-encoder reranking
export DISTILLER_RETRIEVAL__TOP_K_FINAL=8

# Vector store: numpy (default, zero-infra) | qdrant (local mode)
export DISTILLER_STORE__BACKEND=numpy
```

`distiller.toml` example:

```toml
[llm]
model = "gpt-4o-mini"
temperature = 0.1

[retrieval]
top_k_final = 6
rerank = true

[chunking]
target_chars = 1800
overlap_chars = 250
max_chars = 3600
```

## How it works

```
book.pdf / book.epub
        │
        ├── PDF:  docling → pymupdf4llm → pymupdf   (auto, best available)
        └── EPUB: ebooklib + TOC-aware spine walk
        ▼
  Markdown → Blocks → Chapters            (ingest/)
        ▼
  Structure-aware chunking                (chunking/)  chapter-bounded,
        │                                              heading-tracked,
        │                                              page-annotated
        ▼
  Qwen3-Embedding-0.6B (CPU)  +  BM25     (indexing/)
        ▼
  NumpyStore / Qdrant local
        ▼
  hybrid retrieval (RRF) → optional cross-encoder rerank
        ▼
  citation-enforcing prompt → LLM         (rag/)      refuses when not in book
        ▼
  Answer{text, citations[], contexts[]} → eval metrics (evaluation/)
```

Design decisions worth knowing:

- **Generation is pluggable.** `LLMClient` speaks OpenAI-compatible HTTP, so the same
  config works with cloud APIs, Ollama, or llama.cpp. This matters on low-VRAM
  machines: GGUF models run split across the 4 GB GPU and system RAM.
- **The index remembers its embedder.** Mixing embedders returns garbage, so
  `index/metadata.json` records the exact embedder identity and loading fails loudly on
  mismatch.
- **Metrics are deterministic by default.** `distiller eval` computes retrieval hit
  rate, refusal accuracy, snippet coverage and citation coverage with no LLM; RAGAS
  (LLM-judged faithfulness) is opt-in via `--ragas`.
- **Heavy dependencies are optional.** Core install is light; docling/torch/qdrant/ragas
  live in extras, and the parsers/embedders degrade with explicit messages.

## Code conventions

`distiller` follows the Devotion project's engineering standards — the full rules
live in **[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)**. The essentials:

- **88-character lines**, `ruff` as the single formatter + linter (black-compatible,
  isort ordering, mccabe complexity ≤ 10, bandit security rules).
- **`mypy --strict`** on `src/`; every public function is fully type-hinted.
- **Google-style docstrings** on all public functions and classes.
- **Pydantic for boundaries, dataclasses for internals.** Books, chunks, answers,
  eval results and settings are validated, serializable pydantic models; hot-path
  values (`_Piece`, `SearchHit`) and runtime containers (`IndexBundle`) are
  slotted dataclasses.
- **Exceptions:** domain/I/O failures raise `DistillerError` subclasses
  (`IngestError`, `IndexNotFoundError`, `MissingDependencyError`, ...); the CLI turns
  them into friendly errors.
- **Imports:** core libraries at module top; heavy extras (docling, PyMuPDF4LLM,
  sentence-transformers, qdrant, ragas) load dynamically through
  `distiller.optional_deps.require()`. No literal imports inside functions.
- **Naming:** no abbreviations (`book` not `doc`, `metadata` not `meta`,
  `chunk_count` not `n_chunks`).

## Development

```bash
make format        # format + typecheck + lint + security (run before committing)
make unit-test     # unit tests only, parallel
make test          # all tests (unit + integration), fully offline
make coverage      # coverage report (fails under 80%)
make check         # lint + typecheck + tests (the CI gate)
```

The test suite never touches the network: `tests/unit/` covers parsing, chunking,
indexing, retrieval, generation and metrics in isolation, while
`tests/integration/` drives the full CLI pipeline. Fixtures build their own
EPUB/PDF files and the pipeline runs with
`DISTILLER_EMBEDDING__BACKEND=hash` + `DISTILLER_LLM__MODEL=fake`.

### Real-book fixtures

```bash
python scripts/fetch_gutenberg.py --id 1661 --out fixtures/   # Sherlock Holmes EPUB + generated PDF
```

## Licensing notes

Parsing backends have different licenses — relevant if this ever ships publicly:

| Library | License |
| --- | --- |
| docling | MIT |
| pymupdf / pymupdf4llm | AGPL-3.0 |
| EbookLib | AGPL-3.0 |
| marker | GPL-3.0 |

Also note that ingesting and/or training on copyrighted books has legal implications
if outputs are distributed.

## Roadmap details

- **Phase 2 (distillation data):** generate grounded QA pairs from chunks (teacher
  model via API), filter with RAGAS/quality heuristics, format as RAFT examples
  (golden + distractor chunks, verbatim-citation chain-of-thought answers, explicit
  "not in the book" negatives).
- **Phase 3 (training):** QLoRA on Qwen3-4B (Apache 2.0, best fine-tunability at this
  size, 119 languages) with Unsloth on a free T4; evaluate fine-tuned-student+RAG
  against base+RAG on the same golden set.
- **Phase 4 (serving):** merge adapters, export GGUF (Q4_K_M), run via Ollama with
  partial GPU offload; per-book adapter registry.
