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
| Phase 1 | Contextual chunk enrichment, reranking ablations | ✅ implemented |
| Phase 2 | Synthetic QA + RAFT dataset (cloud teacher) | ✅ implemented |
| Phase 3 | Qwen3-4B QLoRA harness: dataset prep, T4 notebook, adapter registry, base-vs-adapter eval | 🔧 harness implemented — the T4 run is manual (see [docs/qlora-runbook.md](docs/qlora-runbook.md)) |
| Phase 4 | GGUF export + CPU/GPU-hybrid serving (llama.cpp/Ollama) | 🔧 harness implemented — the T4 export and local serving are manual (see [docs/gguf-runbook.md](docs/gguf-runbook.md)) |
| Phase 5 | Hierarchical summary tree + global whole-book answering | ✅ implemented (see [docs/thematic-questions.md](docs/thematic-questions.md)) |

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Core: ingest, index and query books (no torch, no docling)
make install

# Full: adds docling (AI PDF parsing), sentence-transformers, qdrant, ragas
make install-all
```

Heavy extras can be picked individually: `uv sync --extra pdf-ai`, `--extra embed`,
`--extra index`, `--extra eval`, `--extra training`.

The `training` extra (Unsloth + TRL + bitsandbytes) is deliberately **not** part of
`make install-all`: Unsloth pins its own torch/CUDA build, so the portable install
stays portable. It is only needed to run the training notebook, not to prepare data
or evaluate adapters.

### Try it offline first

Verify the install end to end with the deterministic fake LLM and hash embedder —
no API key, no model download, no GPU:

```bash
make fixtures    # one-time: public-domain Sherlock Holmes sample (EPUB + PDF)

export DISTILLER_LLM__MODEL=fake
uv run distiller ingest fixtures/sherlock-holmes.epub
uv run distiller index the-adventures-of-sherlock-holmes --embedding-backend hash
uv run distiller ask the-adventures-of-sherlock-holmes "Who is Watson?"
```

## Quickstart

Before the first real run, point the pipeline at a generation backend and make sure
the embedding extra is installed (the commands below assume `make install-all`; with
the core install, add `--embedding-backend hash` or `uv sync --extra embed`):

```bash
# Any OpenAI-compatible endpoint: cloud API, Ollama, llama.cpp, vLLM...
export DISTILLER_LLM__BASE_URL=http://localhost:11434/v1
export DISTILLER_LLM__MODEL=qwen3:8b
export DISTILLER_LLM__API_KEY=...      # only when the endpoint requires one

# ...or run without any backend at all (deterministic smoke answers)
export DISTILLER_LLM__MODEL=fake
```

Then walk the pipeline:

```bash
# 1. Parse a book (EPUB, PDF, Markdown or plain text)
uv run distiller ingest fixtures/sherlock-holmes.epub

# 2. Chunk + embed + index it
uv run distiller index the-adventures-of-sherlock-holmes

# ...or with contextual retrieval (Phase 1): an LLM situates each chunk
uv run distiller index the-adventures-of-sherlock-holmes --contextual

# 3. Ask questions — answers carry citations into the book
uv run distiller ask the-adventures-of-sherlock-holmes "What does Holmes infer from the hat?"
uv run distiller ask the-adventures-of-sherlock-holmes "What colour was the tower?" --json

# 4. Evaluate against a golden question set
cp examples/golden.example.yaml artifacts/<book-id>/golden.yaml
uv run distiller eval the-adventures-of-sherlock-holmes --ragas

# ...or compare retrieval configurations (rerank on/off, top-k sweep)
uv run distiller ablation the-adventures-of-sherlock-holmes --top-k 4,8,12

# 5. Distill training data: grounded QA + RAFT examples (Phase 2)
uv run distiller synth the-adventures-of-sherlock-holmes --max-chunks 100

# 6. Prepare the QLoRA training data + T4 notebook (Phase 3)
#    (steps 6-8 need the manual T4 run; see the runbooks linked in Status)
uv run distiller train the-adventures-of-sherlock-holmes

# ...train manually on a free T4 with training/train_t4.ipynb, then register:
uv run distiller train the-adventures-of-sherlock-holmes --register ./adapter

# 7. Compare base vs fine-tuned adapter on the same golden set
uv run distiller train-eval the-adventures-of-sherlock-holmes

# 8. Validate + register the Q4_K_M GGUF exported by the T4 notebook (Phase 4)
uv run distiller gguf register the-adventures-of-sherlock-holmes ~/Downloads/model-Q4_K_M.gguf
uv run distiller gguf serve the-adventures-of-sherlock-holmes   # prints the exact commands

# ...serve it with Ollama or llama.cpp (see docs/gguf-runbook.md), then evaluate:
export DISTILLER_GGUF__BASE_URL=http://localhost:8080/v1
uv run distiller eval the-adventures-of-sherlock-holmes --gguf
uv run distiller train-eval the-adventures-of-sherlock-holmes --gguf

# 9. Whole-book questions: build the thematic summary tree (Phase 5)
uv run distiller tree build the-adventures-of-sherlock-holmes
uv run distiller ask the-adventures-of-sherlock-holmes "What are the book's main themes?" --global
uv run distiller eval the-adventures-of-sherlock-holmes --global
# ...see docs/thematic-questions.md for window sizing, caching and thematic golden items

# List / inspect
uv run distiller books
uv run distiller info the-adventures-of-sherlock-holmes
```

Everything for one book lives in `artifacts/<book-id>/`:

```
artifacts/the-lantern-keeper/
├── book.json         # structured parse: chapters → blocks
├── parsed.md         # human-readable rendering
├── chunks.jsonl      # retrieval units with provenance (chapter, section, pages)
├── enrichment.jsonl  # cached LLM contexts (contextual retrieval only)
├── index/
│   ├── metadata.json # embedder identity, dim, store, chunk_count, contextual
│   └── store/        # vectors + records (numpy) or qdrant local db
├── dataset/          # Phase 2: qa.jsonl, rejected.jsonl, raft.jsonl, manifest.json
├── training/         # Phase 3: train.jsonl, validation.jsonl, manifest.json,
│   │                 #          qlora.json, train_t4.ipynb
│   ├── adapter/      #          registered T4 adapter + run.json
│   └── gguf/         # Phase 4: model.gguf, gguf.json, Modelfile, serve.sh
├── thematic/         # Phase 5: tree.json, manifest.json, summaries.jsonl
└── eval/
    ├── report.json   # metrics per run (+ index/generator/retrieval identity)
    └── training.json # base-vs-adapter comparison (train-eval)
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

# Contextual retrieval (opt-in): LLM-generated context prefixes per chunk
export DISTILLER_ENRICHMENT__ENABLED=true
export DISTILLER_ENRICHMENT__MAX_CONTEXT_CHARS=500

# Training data split (Phase 3)
export DISTILLER_TRAINING__SEED=13
export DISTILLER_TRAINING__VAL_RATIO=0.1

# Served LoRA adapter (Phase 3 eval); model=None means "no adapter configured"
export DISTILLER_ADAPTER__BASE_URL=http://localhost:8000/v1
export DISTILLER_ADAPTER__MODEL=the-lantern-keeper-lora

# Served GGUF (Phase 4); base_url=None disables --gguf
export DISTILLER_GGUF__BASE_URL=http://localhost:8080/v1     # llama-server (or :11434 for Ollama)
export DISTILLER_GGUF__MODEL=distiller-the-lantern-keeper    # optional; default is the registered name

# Thematic summary tree (Phase 5)
export DISTILLER_THEMATIC__WINDOW_SIZE=4                     # chapters per level-2 window
export DISTILLER_THEMATIC__MAP_TOP_K=6                       # summaries mapped per global question
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

## Troubleshooting the first run

| Symptom | Fix |
| --- | --- |
| `MissingDependencyError: ... --extra embed` during `index` | `uv sync --extra embed` (or `make install-all`), or run with `--embedding-backend hash` |
| `index` fails with "built with embedder ..." | the index remembers its embedder; re-run `distiller index <book>` |
| `ask` fails with an API-key or connection error | set `DISTILLER_LLM__BASE_URL`/`__API_KEY`, or use `DISTILLER_LLM__MODEL=fake` |
| `Golden set not found` on `eval` | `cp examples/golden.example.yaml artifacts/<book-id>/golden.yaml` |
| PDF parses poorly or fails | install `--extra pdf-ai` (docling), or force a backend with `--pdf-backend pymupdf`; parsing warnings name each fallback |
| `--gguf` errors | register the export first (`distiller gguf register ...`) and start the server; the error names the exact command |
| First `index` is slow | the embedding model downloads once and is cached locally; later runs are offline |
| Lost track of a book | `distiller books` lists everything; `distiller info <book-id>` shows the index identity |

## Command reference

| Command | What it does |
| --- | --- |
| `distiller ingest <file>` | Parse a PDF/EPUB/Markdown/text book into `artifacts/<book-id>/` |
| `distiller index <book>` | Chunk + embed + index (`--contextual` for LLM-situated chunks) |
| `distiller ask <book> "<question>"` | Cited answer (`--json`, `--chapter`, `--top-k`, `--rerank`; `--global` for whole-book) |
| `distiller eval <book>` | Golden-set metrics (`--adapter`, `--gguf`, `--global`, `--ragas`) |
| `distiller ablation <book>` | Compare retrieval configs (rerank on/off, top-k sweep) |
| `distiller synth <book>` | Synthetic QA + RAFT dataset from the indexed chunks |
| `distiller train <book>` | Prepare chat splits + QLoRA config + T4 notebook (`--register <dir>` registers an adapter) |
| `distiller train-eval <book>` | Base vs adapter vs served GGUF on one golden set |
| `distiller gguf register <book> <file.gguf>` | Validate + register a downloaded GGUF export |
| `distiller gguf serve <book>` | Print the Modelfile and llama.cpp/Ollama commands (launches nothing) |
| `distiller tree build <book>` | Build the thematic summary tree for `--global` questions |
| `distiller books` / `info <book>` | List ingested books / inspect one book and its index |

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
  (optional) contextual enrichment        (enrichment/) 1-2 sentence LLM context
        │                                              per chunk (cached, index-only)
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
        ▼
  summary tree (chapters → windows → root) → map-reduce whole-book answers
  (thematic/)                                 → ask --global / eval --global
        ▼
  synth → RAFT dataset → chat splits → QLoRA notebook → adapter
  (synthesis/)            (training/)                    → base-vs-adapter eval
```

Design decisions worth knowing:

- **Generation is pluggable.** `LLMClient` speaks OpenAI-compatible HTTP, so the same
  config works with cloud APIs, Ollama, or llama.cpp. This matters on low-VRAM
  machines: GGUF models run split across the 4 GB GPU and system RAM.
- **The index remembers its embedder.** Mixing embedders returns garbage, so
  `index/metadata.json` records the exact embedder identity and loading fails loudly on
  mismatch.
- **Contextual retrieval is index-only.** With `--contextual`, an LLM writes a
  1–2 sentence context per chunk that is used **only** for embedding and BM25
  (`Chunk.index_text`); prompts, answers and citations keep the original book text.
  Contexts are cached per book (`enrichment.jsonl`), so re-indexing costs nothing.
- **Metrics are deterministic by default.** `distiller eval` computes retrieval hit
  rate, refusal accuracy, snippet coverage and citation coverage with no LLM; RAGAS
  (LLM-judged faithfulness) is opt-in via `--ragas`.
- **Heavy dependencies are optional.** Core install is light; docling/torch/qdrant/ragas
  live in extras, and the parsers/embedders degrade with explicit messages. The
  `training` extra is separate from `all` because Unsloth pins its own torch/CUDA stack.
- **Training data reuses the RAG prompt contract.** The chat formatter imports
  `build_system_prompt`/`build_user_prompt` instead of duplicating them, so training
  and inference cannot drift; the dataset manifest records a prompt hash to make any
  drift visible.
- **The repo ships the training harness; the T4 run is manual.** `distiller train`
  prepares the chat splits, pins every hyperparameter in `training/qlora.json` and
  emits a self-contained notebook; the adapter comes back via `--register` and is
  compared with `distiller train-eval`. CI and the offline suite never train a model.
- **One variable per comparison.** `eval/report.json` gains a `generator` block
  (kind base|adapter|gguf, model, adapter/GGUF provenance) next to `index`, and
  `train-eval` records one shared `index`/`retrieval` identity per variant, so a
  reader can verify that fine-tuned-student + RAG, base + RAG and the served Q4
  differ in exactly the generator. Deltas reuse the ablation helper; no automatic
  winner is declared.
- **Serving is just another OpenAI-compatible endpoint.** `distiller gguf register`
  validates the downloaded Q4_K_M file with a dependency-free reader, records its
  hash, quantization and metadata in `gguf.json`, and emits the exact Ollama
  Modelfile and llama.cpp `serve.sh` commands for partial GPU offload. The repo
  never launches a server; `eval --gguf` and `train-eval --gguf` evaluate the served
  model over the same golden set.
- **Whole-book questions map-reduce over a summary tree.** `distiller tree build`
  summarizes chapters, deterministic windows and the whole book into
  `thematic/tree.json`, cached per node so re-runs with unchanged inputs make zero
  LLM calls. `ask --global` maps over the most similar summaries and reduces them
  into one cited answer (`Answer.summary_citations`), reusing the local system
  prompt, `<doc>` evidence blocks and refusal sentence; `eval --global` records the
  mode and tree identity in `eval/report.json`. Local retrieval stays the default.

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
make coverage      # coverage report (fails under 95%)
make spec-check    # validate the SDD spec tree
make check         # lint + typecheck + tests + spec-check (the CI gate)
```

Non-trivial features follow the spec-driven workflow in
**[specs/README.md](specs/README.md)** (draft → review → approve → implement →
archive); `make spec-check` enforces it.

The test suite never touches the network: `tests/unit/` covers parsing, chunking,
indexing, retrieval, generation and metrics in isolation, while
`tests/integration/` drives the full CLI pipeline. Fixtures build their own
EPUB/PDF files and the pipeline runs with
`DISTILLER_EMBEDDING__BACKEND=hash` + `DISTILLER_LLM__MODEL=fake`.

### Real-book fixtures

```bash
make fixtures   # same as:
python scripts/fetch_gutenberg.py --id 1661 --out fixtures/ --slug sherlock-holmes
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

- **Phase 3 (training):** QLoRA on Qwen3-4B (Apache 2.0, best fine-tunability at this
  size, 119 languages) with Unsloth + TRL on a free T4. The harness (chat formatting,
  pinned `qlora.json`, notebook emission, adapter registry, base-vs-adapter eval) is
  implemented; the GPU run itself is a manual step documented in
  [docs/qlora-runbook.md](docs/qlora-runbook.md).
- **Phase 4 (serving):** the T4 notebook merges the adapter and exports a Q4_K_M
  GGUF; `distiller gguf register` validates it with a dependency-free reader,
  records its identity and emits the Ollama Modelfile + llama.cpp `serve.sh` for
  partial GPU offload on a 4 GB card; `eval --gguf` and `train-eval --gguf` evaluate
  the served model. The T4 export and the local server are manual steps documented
  in [docs/gguf-runbook.md](docs/gguf-runbook.md).
- **Phase 5 (thematic layer):** `distiller tree build` summarizes chapters,
  deterministic windows and the whole book into `thematic/tree.json` with a
  per-node cache (`summaries.jsonl`) and a provenance manifest; `ask --global`
  and `eval --global` answer whole-book questions by mapping over the most
  relevant summaries and reducing them into one cited answer. See
  [docs/thematic-questions.md](docs/thematic-questions.md). Entity graphs,
  embedding clustering and automatic local/global routing remain future work.
