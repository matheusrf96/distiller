# Thematic questions: whole-book answering

Local retrieval answers passage-level questions well — a fact, a definition, a
quote — but has no representation of the book above the chunk level, so
questions like *"What are the book's main themes?"* or *"How does the argument
develop across chapters?"* retrieve arbitrary passages and produce shallow
answers. The thematic layer (Phase 5) builds a hierarchical **summary tree**
from the book's own units and answers such questions by map-reducing over its
summaries.

Local `ask`/`eval` are unchanged; global answering is opt-in via `--global`.

## The summary tree

`distiller tree build <book>` summarizes, in one pass:

| Level | Nodes | Input |
|-------|-------|-------|
| 1 | one per chapter | chapter text (the only raw input) |
| 2 | windows of `window_size` consecutive chapters (only when the book has more chapters than `window_size`; a trailing window may be shorter) | the window's child summaries |
| 3 | one whole-book root | its child summaries (windows when level 2 exists, chapters otherwise) |

Node ids are deterministic: `<book-id>:chapter:<n>` (1-based),
`<book-id>:window:<start>-<end>` (inclusive) and `<book-id>:root`. Chapter
nodes cover exactly their chapter's chunks and the root covers every chunk.
Because higher levels only ever see already-distilled summaries, a build costs
`#chapters + #windows + 1` LLM calls and never re-reads the book.

```bash
# Build (or refresh) the tree
uv run distiller tree build the-adventures-of-sherlock-holmes

# Tune the level-2 window size for this run
uv run distiller tree build the-adventures-of-sherlock-holmes --window-size 2

# Force new summaries and/or print the manifest
uv run distiller tree build the-adventures-of-sherlock-holmes --regenerate
uv run distiller tree build the-adventures-of-sherlock-holmes --json
```

Artifacts live under `artifacts/<book>/thematic/`:

```
thematic/
├── tree.json        # nodes: ids, levels, titles, summaries, children, chunk coverage
├── manifest.json    # model, config, source/tree hashes, counts, failed node ids
└── summaries.jsonl  # append-only cache: cache key + node id + summary
```

### Caching and failures

- Summaries are cached per node in `summaries.jsonl`. The cache key embeds the
  node content, the model and the prompt version, so re-ingesting, re-chunking
  or switching models regenerates cleanly and a re-run with unchanged inputs
  makes **zero** LLM calls. `--regenerate` ignores the cache.
- A corrupt or truncated `summaries.jsonl` is ignored with a warning; the
  affected summaries are regenerated.
- One node whose LLM call fails (or whose completion is unusable) is skipped
  with a warning and recorded in `manifest.json` as a failed node; the build
  continues and the partial tree stays usable. Re-running `tree build` retries
  exactly the failed summaries and reuses the rest.
- If **every** chapter summary fails, the build aborts with an error naming the
  configured model and writes nothing.
- A book that is not ingested, has no chunks or has no chapters fails with an
  actionable message (`distiller ingest` / `distiller index`) and writes
  nothing.

## Asking whole-book questions

```bash
uv run distiller ask the-adventures-of-sherlock-holmes \
  "What are the book's main themes?" --global
```

Global answering:

1. **Selects summaries** — the top `map_top_k` summary-bearing nodes across all
   levels by embedding similarity when the index embedder is loadable; without
   a loadable index it maps every chapter summary in reading order.
2. **Maps** — one partial answer per selected summary, using the same system
   prompt, `<doc>` evidence blocks and refusal sentence as local answers.
3. **Reduces** — strips each partial's internal `[1]` marker (it cites its own
   single map document), combines the partials into one final answer and maps
   its `[n]` markers back to node ids and titles (`Answer.summary_citations`,
   `Answer.mode == "global"`).

The CLI prints a Sources table naming the cited nodes:

```
╭─ The Lantern Keeper ──────────────────────╮
│ According to the book: "..." [1]          │
╰───────────────────────────────────────────╯
       Sources
 #   Node          level
 1   Chapter Two   level 1
```

If no summary-bearing node is available (for example a tree whose nodes all
failed), the command returns the shared refusal sentence with `refused=True`
without calling the reduce step.

`--global` selects the mode and composes with `--adapter`/`--gguf`. The
local-only retrieval flags are rejected with a friendly error when combined
with `--global`: `--top-k`, `--chapter` and `--rerank`. `ask` without
`--global` is unchanged.

## Evaluating global answers

```bash
uv run distiller eval the-adventures-of-sherlock-holmes \
  --golden artifacts/the-adventures-of-sherlock-holmes/golden.yaml --global
```

The golden items are answered through the global pipeline and
`eval/report.json` gains a `retrieval` block next to the existing `index` and
`generator` blocks:

```json
{
  "retrieval": {
    "mode": "global",
    "tree": {
      "hash": "…",
      "model": "gpt-4o-mini",
      "chapter_count": 12,
      "window_count": 3,
      "node_count": 16
    }
  }
}
```

A local eval records `{"mode": "local", "tree": null}`. `--ragas` combined with
`--global` is rejected with a friendly error (global answers have no retrieved
chunks for the judge).

The deterministic metrics need no global-only additions: summary citations
count toward `citation_coverage`, cited node titles count as retrieved
chapters, expected snippets count toward `contains_rate` and refusals toward
`refusal_accuracy`.

## Writing thematic golden items

Thematic golden items are hand-written, exactly like local ones; the only
difference is that global answers cite summary nodes instead of chunks.

```yaml
items:
  - question: What are the book's main themes?
    expected_answer_contains:
      - memory
      - forgiveness
  - question: How does the argument develop between chapters 1 and 4?
    expected_chapters:
      - Chapters 1-4
    expected_answer_contains:
      - first
  - question: Who won the village sailing race?
    answerable: false
```

- `expected_answer_contains` checks snippets in the final answer; it works for
  global answers because the reduce step produces the same grounded text.
- `expected_chapters` is matched against the **titles of cited summary nodes**,
  so a chapter title ("Chapter Two") or a window title ("Chapters 1-4") both
  work; leave it out when any node may answer.
- `answerable: false` expects the shared refusal sentence, counted by
  `refusal_accuracy`.

Keep a local set for passage questions and a thematic set for whole-book
questions; run each with its mode.

## Configuration

| Setting | Default | Meaning |
|---------|---------|---------|
| `DISTILLER_THEMATIC__WINDOW_SIZE` | `4` | consecutive chapters per level-2 window (≥ 2) |
| `DISTILLER_THEMATIC__MAP_TOP_K` | `6` | summaries mapped over per global question (≥ 1) |
| `DISTILLER_THEMATIC__MAX_SOURCE_CHARS` | `6000` | cap for the text sent to a summarization call |
| `DISTILLER_THEMATIC__MAX_SUMMARY_CHARS` | `1200` | cap for one generated node summary |

The tree build uses the configured base LLM; `ask --global` and `eval --global`
accept `--adapter` and `--gguf` like their local counterparts. The whole flow —
`ingest → index → tree build → ask --global → eval --global` — runs fully
offline with `DISTILLER_LLM__MODEL=fake` and
`DISTILLER_EMBEDDING__BACKEND=hash` and adds no runtime dependency.
