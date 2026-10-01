---
id: SDD-0006
status: approved
supersedes:
owner: matheus
created: 2026-10-01
---

# Hierarchical Summary Tree + Global Answering (Phase 5)

## Problem

Hybrid retrieval answers local questions well — a passage, a fact, a definition —
but fails whole-book sensemaking: "What are the book's main themes?", "How does
the argument develop across chapters?", "Summarize chapter 3 in relation to the
whole". Point RAG has no representation of the book above the chunk level, so
those questions retrieve arbitrary passages and produce shallow or partial
answers. This spec adds a hierarchical summary tree (RAPTOR-style, but built
from the book's own units) and an explicit global answering mode that maps over
the tree's summaries and reduces them into one grounded, cited answer.

## Context

- **Local pipeline**: `src/distiller/rag/` (`Retriever` → optional rerank →
  `Generator` → `QAPipeline`), evaluated by `distiller eval`. Local behaviour
  must not change: global mode is opt-in and the local path stays untouched.
- **Chunks**: `chunks.jsonl` written by `distiller index`; `load_book_and_chunks`
  (`cli/context.py`, SDD-0003) is the no-embedder loading path the tree build
  reuses, so summarization never needs a model download.
- **LLM abstraction**: `src/distiller/llm/` (any OpenAI-compatible endpoint plus
  the deterministic `FakeLLM`). The fake already branches on `<excerpt>`
  (SDD-0001), `<chunk>` (SDD-0003) and `<doc>` (v1); this phase adds `<source>`.
- **Prompt contract**: `rag/prompts.py::build_system_prompt` enforces grounding,
  `[n]` citations and the shared `refusal_text` (`models.py`, SDD-0003). Global
  map and reduce calls reuse that exact system prompt and render their evidence
  as `<doc>` blocks, so the contract is extended, not duplicated.
- **Report identity**: `eval/report.json` carries an `index` block (SDD-0001)
  and a `generator` block with kinds `base`/`adapter`/`gguf` (SDD-0004/0005);
  this phase adds a `retrieval` block with the mode and tree identity, so local
  and global runs are distinguishable.
- **Cache precedent**: `enrichment/contextual.py` caches per-chunk contexts in
  `enrichment.jsonl` and ignores a corrupt cache with a warning; the summary
  cache copies that pattern at node granularity.
- **Related specs**: Phase 5 in the roadmap. It consumes the index (embedder for
  selection) but does not depend on Phases 2-4.

### Data model deltas

| Model / field | Kind | Notes |
|---|---|---|
| `SummaryNode` | pydantic (`DomainModel`) | `id`, `level` (1 chapter, 2 window, 3 root), `title`, `summary` (None when the node failed), `children`, `chunk_ids` |
| `SummaryTree` | pydantic (`DomainModel`) | `book_id`, `root_id`, `nodes` in reading order (chapters, windows, root) |
| `TreeManifest` | pydantic (`DomainModel`) | `model`, `config`, `source_hash`, `tree_hash`, node counts, generated/reused counts and failed node ids |
| `TreeRun` | dataclass (runtime only) | The built tree plus its manifest |
| `SummaryCitation` | pydantic (`DomainModel`, `models.py`) | `index`, `node_id`, `title`, `level` |
| `Answer` | pydantic (extend) | `mode` (`local`/`global`, default `local`) and `summary_citations` |
| `ThematicSettings` | pydantic settings | Root `Settings.thematic`: `window_size`, `map_top_k`, `max_source_chars`, `max_summary_chars` |

### Module deltas

```
src/distiller/thematic/
├── __init__.py       # exports
├── prompts.py        # summarization + global map/reduce prompt builders
├── summarizer.py     # cached per-node summarization (summaries.jsonl)
├── tree.py           # SummaryNode/SummaryTree/TreeManifest/TreeRun, build_tree
└── global_qa.py      # GlobalPipeline: selection + map-reduce answering
```

Layering: `thematic` sits next to `synthesis` — it consumes the book, chunks,
the LLM and the `rag` prompt contract; `rag` never imports `thematic`.

### Tree shape

- **Level 1** — one node per chapter, summarizing the chapter text.
- **Level 2** — windows of `window_size` consecutive chapters, each summarizing
  its children's summaries. Windows exist only when the book has more chapters
  than `window_size`; a trailing window may be shorter.
- **Level 3 (root)** — one whole-book node summarizing its children (window
  summaries when level 2 exists, chapter summaries otherwise).
- Ids are deterministic: `"<book_id>:chapter:<index>"`,
  `"<book_id>:window:<start>-<end>"` (1-based, inclusive) and `"<book_id>:root"`.
  Titles: the chapter title, `"Chapters <start>-<end>"`, the book title.
- `chunk_ids`: chunks whose `chapter` matches the node's chapters (the root
  covers every chunk); `children`: child node ids.
- `source_hash` hashes the ordered chunk ids (chunk ids embed content hashes, so
  edits and re-chunking invalidate it); `tree_hash` hashes the ordered node ids
  and summaries.

### Configuration

`ThematicSettings(window_size=4, map_top_k=6, max_source_chars=6000,
max_summary_chars=1200)` under `Settings.thematic`, with validated bounds
(`window_size >= 2`, `map_top_k >= 1`, positive char budgets). Environment:
`DISTILLER_THEMATIC__WINDOW_SIZE=…` (nested like every other setting).

### CLI

- `distiller tree build <book> [--window-size N] [--regenerate] [--json]` —
  summarize chapter → window → root, write the artifacts, print a summary table.
- `distiller ask <book> <question> --global` — map-reduce over selected
  summaries; prints a Sources table naming the cited nodes; without `--global`
  the command is unchanged.
- `distiller eval <book> --global` — run the golden items through the global
  pipeline and record mode + tree identity in `eval/report.json`.
- `--global` selects the retrieval mode and composes with the existing
  generator flags (`--adapter`, `--gguf`); the local-only flags are rejected
  with a friendly error when combined with `--global`: the retrieval options
  (`--top-k`, `--chapter`, `--rerank`) on `ask`, `--ragas` on `eval`.

### Artifact layout

```
artifacts/<book>/
├── thematic/
│   ├── tree.json          # SummaryTree: nodes, ids, levels, summaries, coverage
│   ├── manifest.json      # TreeManifest: model, config, hashes, counts
│   └── summaries.jsonl    # append-only cache: cache key + node id + summary
└── eval/
    └── report.json        # retrieval.mode + tree identity (+ generator, index)
```

### Failure modes

| Failure | Behaviour |
|---|---|
| Book not ingested | Friendly error: run `distiller ingest` first |
| Book not indexed (no chunks) | Friendly error: run `distiller index` first |
| No tree for `ask --global` / `eval --global` | Friendly error: run `distiller tree build <book>` first |
| One node's LLM call fails or the completion is unusable | Node recorded as failed with a warning; build continues; a re-run regenerates only what the failure invalidated |
| Every chapter summary fails | `ThematicError` → friendly CLI error naming the configured model; nothing is written |
| `summaries.jsonl` missing or corrupt | Ignored with a warning; affected summaries regenerate |
| `tree.json` corrupt | Friendly error: re-run `distiller tree build` |
| Book has no chapters or chunks | Friendly error naming the book; nothing is written |
| `--global` with a local-only flag | Rejected: `--top-k`/`--chapter`/`--rerank` on `ask`, `--ragas` on `eval` |
| No summary-bearing node selected | The shared refusal sentence, `refused=True`, no reduce call |

### Design decisions

- **Book-native hierarchy, not clustering.** Chapters are the natural unit the
  book's TOC already provides; deterministic windows keep the tree reproducible
  and dependency-free. UMAP/GMM/k-means clustering over embeddings is a future
  refinement, not a prerequisite.
- **Higher levels summarize their children's summaries.** Chapter text is the
  only raw input; window and root calls see bounded, already-distilled text, so
  the build costs ~`#chapters + #windows + 1` calls and never re-reads the book.
- **Cache keyed by node content + model + prompt version.** The key embeds the
  source text hash, so re-chunking, re-ingesting or switching models regenerates
  cleanly; a re-run with unchanged inputs makes zero LLM calls. Only successful
  summaries are cached, which is why a re-run retries exactly the holes.
- **Skip-with-warning on node failure, not fail-fast.** One flaky call must not
  discard the paid summaries of every other node; the partial tree stays usable
  and the manifest records what is missing.
- **Reuse the RAG prompt contract.** Map and reduce calls use
  `build_system_prompt` and render summaries and partial answers as `<doc>`
  blocks, so grounding, `[n]` citations and the refusal sentence stay identical
  to local answers and to the RAFT training targets.
- **Selection happens at query time; no summary vectors are persisted.** The
  tree hash stays embedder-independent, and a book without a loadable index
  still answers globally by mapping every chapter summary in reading order.
- **Explicit modes, no routing.** `--global` is opt-in; local `ask`/`eval` keep
  their exact behaviour and reports gain only an additive `retrieval` block.
  Automatic local-vs-global routing is a future phase.
- **Summary citations are first-class.** `Answer.summary_citations` maps the
  reduce answer's `[n]` markers back to tree nodes; the deterministic metrics
  count them like chunk citations, so global golden items need no new metrics.
- **Manifest separate from the tree.** `tree.json` holds the content a reader
  wants; `manifest.json` holds provenance and counts, mirroring
  `dataset/manifest.json` (SDD-0003).

## Requirements

1. **REQ-TH-001** — WHEN `distiller tree build <book>` runs for an indexed book,
   THEN the system SHALL summarize every chapter (level 1), every window of
   `window_size` consecutive chapters (level 2, only when the book has more
   chapters than `window_size`) and the whole book (root), writing
   `thematic/tree.json` and `thematic/manifest.json`, and SHALL accept
   `--window-size`, `--regenerate` and `--json`.
2. **REQ-TH-002** — WHEN a tree is built, THEN every node SHALL carry a
   deterministic id, its level (1 chapter, 2 window, 3 root), a title, its
   summary (None when the node failed), its child ids and the source chunk ids
   it covers, validated as pydantic models; the root SHALL cover every chunk.
3. **REQ-TH-003** — WHEN a summary for a node's current source text, model and
   parameters exists in `thematic/summaries.jsonl`, THEN the system SHALL reuse
   it without calling the LLM unless `--regenerate` is given, and the manifest
   SHALL record generated and reused counts.
4. **REQ-TH-004** — IF `thematic/summaries.jsonl` is missing or corrupt, THEN
   the system SHALL ignore it with a warning and generate the affected
   summaries.
5. **REQ-TH-005** — IF an LLM call for one node fails or returns an unusable
   completion, THEN the system SHALL record the node as failed with a warning,
   exclude it from higher-level inputs and continue; a later `tree build` SHALL
   retry the failed summaries and reuse the summaries whose inputs are
   unchanged. IF every chapter summary fails, THEN the system SHALL fail with a
   `ThematicError` naming the configured model and SHALL NOT write thematic
   artifacts.
6. **REQ-TH-006** — IF the book is not ingested, has no chunks or has no
   chapters, THEN the system SHALL fail with an actionable message naming the
   next command (`distiller ingest` or `distiller index`) and SHALL NOT write
   thematic artifacts.
7. **REQ-TH-007** — WHEN `distiller ask <book> <question> --global` runs with a
   built tree, THEN the system SHALL select summaries (the top `map_top_k` by
   embedding similarity when the index embedder is available, otherwise every
   chapter summary in reading order), ask the LLM for one partial answer per
   selected summary, reduce the partials into one final answer, and SHALL reject
   the local-only retrieval flags (`--top-k`, `--chapter`, `--rerank`) combined
   with `--global`; `ask` without `--global` SHALL use the local pipeline
   unchanged.
8. **REQ-TH-008** — WHEN a global answer is produced, THEN it SHALL cite the
   summary nodes it relies on (`[n]` markers mapped back to node ids and
   titles), SHALL use the shared refusal sentence when the summaries do not
   answer the question, and SHALL carry `mode == "global"`. IF no
   summary-bearing node is selected, THEN the system SHALL return the shared
   refusal sentence with `refused=True` and SHALL NOT call the reduce step.
9. **REQ-TH-009** — IF `ask --global` or `eval --global` runs without
   `thematic/tree.json` or with a corrupt one, THEN the system SHALL fail with a
   friendly error naming `distiller tree build` and SHALL NOT print a
   traceback.
10. **REQ-TH-010** — WHEN `distiller eval <book> --global` runs, THEN the system
    SHALL evaluate the golden items through the global pipeline and write
    `eval/report.json` with a `retrieval` block recording `mode: "global"` and
    the tree identity (tree hash, model, chapter/window/node counts) next to the
    existing `generator` and `index` blocks; a local eval SHALL record
    `mode: "local"` with a null tree; `--ragas` combined with `--global` SHALL
    be rejected with a friendly error.
11. **REQ-TH-011** — WHEN global golden items are evaluated, THEN summary
    citations SHALL count toward `citation_coverage`, expected snippets toward
    `contains_rate` and refusals toward `refusal_accuracy`, with no global-only
    metric.
12. **REQ-TH-012** — WHEN the offline fake LLM and hash embedder are configured,
    THEN `ingest → index → tree build → ask --global → eval --global` SHALL
    complete with no network, GPU or new runtime dependency, and `FakeLLM` SHALL
    gain a deterministic `<source>` summarization branch.
13. **REQ-TH-013** — WHEN `tree build` completes, THEN `thematic/manifest.json`
    SHALL record the model, parameters, source hash, tree hash, node counts,
    generated/reused counts and the failed node ids, and `--json` SHALL print
    the same manifest.
14. **REQ-TH-014** — The repository SHALL ship `docs/thematic-questions.md`
    documenting the tree build, global ask/eval and how to write thematic golden
    items, and the doc SHALL be linked from `docs/README.md`.

## Acceptance Criteria

- **AC1** (Req 1) A build over the three-chapter fixture writes
  `thematic/tree.json` with three level-1 nodes and one level-3 root (no level-2
  nodes at `window_size=4`); with `--window-size 2` it writes two level-2
  windows; the manifest counts match the file.
- **AC2** (Req 2) Node ids are identical across rebuilds, levels are 1/2/3,
  window children are consecutive chapter ids, and each node's `chunk_ids`
  cover exactly its chapters while the root covers every chunk.
- **AC3** (Req 3) A second build with an LLM that raises on any call reports zero
  generated summaries and regenerates nothing; `--regenerate` calls the LLM
  again.
- **AC4** (Req 4) A truncated `summaries.jsonl` is ignored with a warning and
  every summary is regenerated.
- **AC5** (Req 5) An LLM failing one chapter records that node as failed with a
  warning while the other nodes are summarized; a second build retries the
  failed chapter and reuses the unchanged chapter summaries; an LLM failing
  every chapter exits non-zero naming the model, with no `thematic/` directory.
- **AC6** (Req 6) A book without chunks exits non-zero naming `distiller index`,
  with no traceback and no `thematic/` directory.
- **AC7** (Req 7) `ask --global` makes one map call per selected summary plus one
  reduce call and returns an answer whose CLI output prints a Sources table
  naming the cited nodes; plain `ask` is unchanged, and `--top-k`, `--chapter`
  and `--rerank` combined with `--global` are rejected with a friendly error.
- **AC8** (Req 8) The global answer's `summary_citations` reference selected node
  ids and titles, an unanswerable question yields the shared refusal sentence
  with `refused=True`, `mode == "global"`, and a tree with no summary-bearing
  node yields the refusal sentence without a reduce call.
- **AC9** (Req 9) `ask --global` and `eval --global` without a tree — or with a
  corrupt `tree.json` — exit non-zero with "distiller tree build" in the message
  and no traceback.
- **AC10** (Req 10) `eval --global` writes `retrieval.mode == "global"` with a
  tree hash, model and counts matching `manifest.json`, alongside the
  `generator` block; a plain eval writes `mode == "local"` and a null tree, and
  `--ragas` combined with `--global` is rejected with a friendly error.
- **AC11** (Req 11) A thematic golden item with `expected_answer_contains` and no
  `expected_chapters` yields a non-null `contains_rate` and `citation_coverage`,
  and an unanswerable item affects `refusal_accuracy`.
- **AC12** (Req 12) The offline chain `ingest → index → tree build → ask --global
  → eval --global` completes with the fake LLM and hash embedder, and the fake
  answers a `<source>` prompt deterministically; importing `distiller.thematic`
  adds no dependency.
- **AC13** (Req 13) The manifest records model, window size, source/tree hashes,
  node/generated/reused counts and the failed node ids; `--json` output equals
  the written file.
- **AC14** (Req 14) `docs/thematic-questions.md` exists, names `distiller tree
  build`, `ask --global`, `eval --global` and the thematic golden-item shape, and
  is linked from `docs/README.md`.

## Non-Goals

- Entity/relation knowledge graphs (LightRAG/MS GraphRAG), community detection
  and graph databases.
- Embedding clustering (UMAP/GMM/k-means) for tree construction; a future
  refinement of the level-2 units.
- Incremental tree updates, multi-book/corpus-level themes and cross-book
  synthesis.
- New runtime dependencies, vision/multimodal input and serving changes
  (Phase 4 covers serving).
- Automatic query routing between local and global modes; the user chooses the
  mode explicitly.
- Persisting summary embeddings or a summary vector store; selection embeds at
  query time.
- Cross-encoder reranking of summaries and LLM-judged (RAGAS) metrics for
  global answers.
- Recursive map-reduce beyond one map and one reduce (collapsed-tree answering).
- LLM-generated golden items; thematic golden items are hand-written.

## File-change Plan

| Action | Path | Purpose |
|--------|------|---------|
| create | `src/distiller/thematic/__init__.py` | Package exports |
| create | `src/distiller/thematic/prompts.py` | Summarization prompt + global map/reduce prompt builders |
| create | `src/distiller/thematic/summarizer.py` | Cached per-node summarization (`summaries.jsonl`) |
| create | `src/distiller/thematic/tree.py` | `SummaryNode`/`SummaryTree`/`TreeManifest`/`TreeRun`, `build_tree` |
| create | `src/distiller/thematic/global_qa.py` | `GlobalPipeline`: summary selection + map-reduce |
| modify | `src/distiller/models.py` | `SummaryCitation`; `Answer.mode` + `Answer.summary_citations` |
| modify | `src/distiller/evaluation/metrics.py` | Count summary citations and cited node titles |
| modify | `src/distiller/exceptions.py` | `ThematicError(DistillerError)` failure domain |
| modify | `src/distiller/config.py` | `ThematicSettings` + `Settings.thematic` |
| modify | `src/distiller/paths.py` | `thematic_dir`, `thematic_tree_json`, `thematic_manifest_json`, `thematic_summaries_jsonl` |
| modify | `src/distiller/llm/fake.py` | Deterministic `<source>` summarization branch |
| modify | `src/distiller/cli/context.py` | Tree load/build helpers; optional-index + global pipeline construction |
| modify | `src/distiller/cli/main.py` | `tree` sub-app; `--global` on `ask`/`eval`; `retrieval` report block |
| modify | `src/distiller/cli/render.py` | `render_tree_build` summary table; summary-citation Sources table in `render_answer` (global only) |
| create | `docs/thematic-questions.md` | Tree build, global ask/eval, thematic golden items |
| modify | `docs/README.md` | Link the thematic doc |
| modify | `README.md`, `AGENTS.md` | Quickstart, artifact layout, package map, roadmap status |
| create | `tests/unit/test_thematic_tree.py` | Tree build, ids, cache, failure, manifest and doc tests |
| create | `tests/unit/test_thematic_summarizer.py` | Prompt, cache and fake-LLM tests |
| create | `tests/unit/test_thematic_global_qa.py` | Selection, map-reduce, citation and refusal tests |
| modify | `tests/unit/test_eval_metrics.py` | Summary-citation metric test |
| modify | `tests/conftest.py` | `tree_factory` fixture |
| modify | `tests/integration/test_cli_end_to_end.py` | CLI thematic coverage |

## Test Plan

| AC | Test file | Test name |
|----|-----------|-----------|
| AC1 | `tests/unit/test_thematic_tree.py` | `test_build_tree_summarizes_chapters_windows_and_root` |
| AC1 | `tests/unit/test_thematic_tree.py` | `test_build_tree_skips_windows_for_short_books` |
| AC1 | `tests/integration/test_cli_end_to_end.py` | `test_tree_build_command_end_to_end` |
| AC2 | `tests/unit/test_thematic_tree.py` | `test_node_ids_are_deterministic_and_cover_chunks` |
| AC3 | `tests/unit/test_thematic_tree.py` | `test_rebuild_reuses_cached_summaries_without_calling_the_llm` |
| AC3 | `tests/unit/test_thematic_tree.py` | `test_regenerate_forces_new_llm_calls` |
| AC4 | `tests/unit/test_thematic_summarizer.py` | `test_corrupt_summary_cache_is_ignored_with_a_warning` |
| AC5 | `tests/unit/test_thematic_tree.py` | `test_failed_node_is_skipped_and_retried_on_rebuild` |
| AC5 | `tests/integration/test_cli_end_to_end.py` | `test_tree_build_fails_when_every_summary_fails` |
| AC6 | `tests/integration/test_cli_end_to_end.py` | `test_tree_build_requires_an_indexed_book` |
| AC7 | `tests/unit/test_thematic_global_qa.py` | `test_global_ask_maps_selected_summaries_and_reduces` |
| AC7 | `tests/unit/test_thematic_global_qa.py` | `test_global_selection_falls_back_to_chapter_summaries_without_an_index` |
| AC7 | `tests/integration/test_cli_end_to_end.py` | `test_ask_global_command_end_to_end` |
| AC8 | `tests/unit/test_thematic_global_qa.py` | `test_global_answer_cites_summary_nodes` |
| AC8 | `tests/unit/test_thematic_global_qa.py` | `test_global_answer_refuses_with_the_shared_sentence` |
| AC8 | `tests/unit/test_thematic_global_qa.py` | `test_global_ask_refuses_when_no_summary_is_available` |
| AC9 | `tests/integration/test_cli_end_to_end.py` | `test_global_commands_require_a_tree` |
| AC9 | `tests/integration/test_cli_end_to_end.py` | `test_global_commands_reject_a_corrupt_tree` |
| AC10 | `tests/integration/test_cli_end_to_end.py` | `test_eval_global_records_the_tree_identity` |
| AC10 | `tests/integration/test_cli_end_to_end.py` | `test_eval_global_rejects_ragas` |
| AC10 | `tests/integration/test_cli_end_to_end.py` | `test_full_offline_pipeline` |
| AC11 | `tests/unit/test_eval_metrics.py` | `test_summary_citations_count_toward_citation_coverage` |
| AC12 | `tests/integration/test_cli_end_to_end.py` | `test_thematic_pipeline_end_to_end` |
| AC12 | `tests/unit/test_thematic_summarizer.py` | `test_fake_llm_summarizes_source_blocks_deterministically` |
| AC13 | `tests/unit/test_thematic_tree.py` | `test_manifest_records_provenance_and_counts` |
| AC13 | `tests/integration/test_cli_end_to_end.py` | `test_tree_build_command_end_to_end` |
| AC14 | `tests/unit/test_thematic_tree.py` | `test_thematic_questions_doc_documents_the_workflow` |

## Open Questions

- **CLI verb shape**: `distiller tree build` (recommended: a sub-app leaves room
  for `tree show`/`tree verify`) vs `distiller summarize` or `theme build`.
- **Mode selector**: `--global` (recommended: matches `--adapter`/`--gguf` and
  keeps the local default) vs `--mode local|global`.
- **Level-2 window size**: fixed default `window_size=4` (recommended, tunable)
  vs a size that scales with the chapter count.
- **Level-2 input**: child summaries (recommended: bounded input, cheap) vs raw
  chapter text (higher fidelity, more tokens).
- **Map selection**: top `map_top_k` by embedding similarity across all levels
  (recommended) vs always including the root, or level-1 only.
- **Does `--global` imply the tree build?** Recommended no: the build is explicit
  so LLM spend is predictable; ask/eval name `distiller tree build` when it is
  missing.
- **Prompt shape**: reuse `build_system_prompt` and `<doc>` evidence blocks
  (recommended: one contract for local and global) vs a dedicated global system
  prompt.
- **Node failure policy**: skip + record + retry on re-run (recommended) vs
  failing the whole build or retrying inline.
- **Local flags with `--global`**: reject `--top-k`/`--chapter`/`--rerank` on
  `ask` and `--ragas` on `eval` (recommended) vs mapping `--top-k` to
  `map_top_k`.
- **Summary citation model**: new `SummaryCitation` + `Answer.summary_citations`
  (recommended: local `Citation` keeps its chunk contract) vs generalizing
  `Citation` with an optional node id.

## Outcomes

<!-- Filled on archival. -->

- **Implemented in**: <commit SHAs>
- **Spec archived**: <date>
- **Post-mortem notes**: <optional>
