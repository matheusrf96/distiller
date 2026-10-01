---
id: SDD-0003
status: done
supersedes:
owner: matheus
created: 2026-09-30
archived: 2026-09-30
---

# Synthetic QA + RAFT Dataset (Phase 2)

## Problem

Phase 3 will fine-tune a 3-4B model on this book, but no training data exists.
Hand-writing questions is slow and does not scale to hundreds of chunks. This
spec turns an indexed book into supervised data: a teacher LLM writes grounded
question-answer pairs, deterministic filters keep only pairs whose quotes are
verifiable against the source chunk, and a RAFT formatter emits training
examples with the golden chunk plus distractors, chain-of-thought answers that
cite verbatim, and explicit "not in the book" negatives.

## Context

- **Chunks**: `chunks.jsonl` written by `distiller index`
  (`src/distiller/indexing/bundle.py`). Synthesis consumes chunks, not vectors,
  so it needs no embedder and runs on a plain install.
- **LLM abstraction**: `src/distiller/llm/` (any OpenAI-compatible endpoint,
  plus the deterministic `FakeLLM`).
- **Refusal sentence**: previously private to `rag/prompts.py`; the RAFT
  negatives must target exactly the sentence the RAG prompt enforces, so it
  moved to `src/distiller/models.py` and both layers import it from there.
- **Related specs**: SDD-0001 records `index` identity in reports, which this
  spec copies into the dataset manifest; Phase 3 (training) consumes the
  dataset produced here.

### Data model deltas

| Model / field | Kind | Notes |
|---|---|---|
| `QAPair` | pydantic | `id`, `book_id`, `chunk_id`, `chapter`, `heading_path`, `question`, `answer`, `quotes`, `model` |
| `RejectedPair` / `FilterOutcome` | pydantic | Rejection bookkeeping with `rejection_counts()` |
| `RaftContext` / `RaftExample` | pydantic | Contexts with `is_golden`; examples with `answerable` |
| `DatasetManifest` | pydantic | Provenance: model, source, index identity, config, all counts |
| `SynthesisRun` | dataclass (runtime only) | Manifest + pairs + filter outcome + examples |

### Module deltas

```
src/distiller/synthesis/
├── __init__.py       # exports
├── qa.py             # prompts, JSON parsing, generation
├── filtering.py      # deterministic filters and rejection reasons
├── raft.py           # RAFT example builder (positives + negatives)
└── dataset.py        # sampling, manifest, orchestration, cache reuse
```

### Configuration

`SynthesisSettings(max_chunks=100, questions_per_chunk=2, distractors=4,
negative_ratio=0.15, seed=13)` under `Settings.synthesis`, with validated
bounds. Environment: `DISTILLER_SYNTHESIS__MAX_CHUNKS=…`.

### CLI

`distiller synth <book> [--max-chunks] [--questions-per-chunk] [--distractors]
[--negative-ratio] [--seed] [--regenerate] [--json]` — writes
`dataset/{qa,rejected,raft}.jsonl` + `dataset/manifest.json`.

### Failure modes

| Failure | Behaviour |
|---|---|
| Chunks missing (book not indexed) | Friendly error: run `distiller index` first |
| One chunk's completion unparseable | Skipped with a warning; the run continues |
| LLM error for one chunk | Skipped with a warning; the run continues |
| `qa.jsonl` corrupt | Treated as absent (regenerate) with a warning |
| No pairs survive filtering | Files written with zero examples; the table says so |

### Design decisions

- **Deterministic filters, not LLM judges.** Filters must be cheap, explainable
  and CI-stable; RAGAS-scored filtering is a later iteration.
- **Reuse `qa.jsonl`.** Generation costs money; filtering and RAFT layers are
  free to iterate on without re-paying the teacher.
- **One refusal sentence for prompt and training.** Training data must match
  inference behaviour, hence the shared `refusal_text`.
- **Verbatim quotes as the grounding check.** A quote that is not a substring of
  the source chunk (normalized whitespace) is dropped; a pair with no verified
  quotes is rejected.

## Requirements

1. **REQ-SQ-001** — WHEN `distiller synth` runs, THEN the system SHALL generate
   grounded question-answer pairs for sampled chunks with the configured LLM
   and write them to `dataset/qa.jsonl`.
2. **REQ-SQ-002** — WHEN generating pairs, THEN the system SHALL require
   verbatim quotes from the source chunk in each pair and SHALL persist them
   with the pair.
3. **REQ-SQ-003** — WHEN `questions_per_chunk` is N, THEN the system SHALL
   request exactly N questions per sampled chunk in a single LLM call.
4. **REQ-SQ-004** — WHEN a completion is not valid JSON, or an item is
   malformed, THEN the system SHALL skip that chunk or item, log a warning and
   continue.
5. **REQ-SQ-005** — WHEN filtering pairs, THEN the system SHALL drop quotes that
   do not appear verbatim in the source chunk, reject a pair whose quotes all
   fail, reject duplicate questions and reject questions that nearly echo their
   answer, counting every rejection by reason.
6. **REQ-SQ-006** — WHEN building RAFT examples, THEN the system SHALL place
   the golden chunk among `distractors` other chunks from the same book in a
   seeded random order.
7. **REQ-SQ-007** — WHEN a pair is answerable, THEN the system SHALL format the
   target answer as the primary verbatim quote with the golden context's
   citation index, followed by the answer text.
8. **REQ-SQ-008** — WHEN a negative example is built, THEN the system SHALL
   provide distractor contexts only and target the refusal sentence shared with
   the RAG prompt.
9. **REQ-SQ-009** — WHEN the dataset completes, THEN the system SHALL write
   `dataset/raft.jsonl`, `dataset/rejected.jsonl` and `dataset/manifest.json`
   containing counts, configuration, model and index identity.
10. **REQ-SQ-010** — WHEN `dataset/qa.jsonl` already exists, THEN the system
    SHALL reuse it instead of calling the LLM again unless `--regenerate` is
    given, recording the source in the manifest.
11. **REQ-SQ-011** — WHEN sampling chunks, THEN the system SHALL do so
    deterministically from the seed, capped by `--max-chunks`, in reading order.
12. **REQ-SQ-012** — WHEN the command finishes, THEN the system SHALL print a
    summary table with generated/kept/rejected/example counts and the output
    paths.
13. **REQ-SQ-013** — WHEN the offline fake LLM is configured, THEN the system
    SHALL complete the whole synth pipeline.
14. **REQ-SQ-014** — WHEN `--json` is given, THEN the system SHALL print the
    manifest as JSON instead of the table.
15. **REQ-SQ-015** — WHEN the book has not been indexed, THEN the system SHALL
    fail with an actionable message telling the user to run `distiller index`
    first.

## Acceptance Criteria

- **AC1** (Req 1) A synth run over sampled chunks produces `qa.jsonl` with
  provenance (`chunk_id`, `chapter`, `model`) and consistent manifest counts.
- **AC2** (Req 2) Parsed pairs carry their quotes; the fake LLM's quotes are
  verbatim substrings of the source chunk.
- **AC3** (Req 3) The generation prompt contains "exactly N question" and the
  chunk in `<chunk>` tags.
- **AC4** (Req 4) Prose-wrapped and fenced JSON is recovered; garbage yields no
  pairs with a warning; malformed items are dropped individually; a failing
  chunk does not abort the run.
- **AC5** (Req 5) Unverified quotes are dropped, quote-less pairs, short pairs,
  duplicate questions and answer-echoing questions are rejected with reasons.
- **AC6** (Req 6) A positive example has exactly one golden context among
  `distractors + 1` contexts, chosen reproducibly from the seed.
- **AC7** (Req 7) The positive answer starts with the quoted evidence, carries
  the golden context's `[n]` index, and ends with the answer text.
- **AC8** (Req 8) Negative examples have no golden context and target the
  refusal sentence containing the book title.
- **AC9** (Req 9) The manifest records counts, config, model and index identity;
  `raft.jsonl` and `rejected.jsonl` are written.
- **AC10** (Req 10) A second run reports `source: cache` and performs no LLM
  calls while producing identical examples.
- **AC11** (Req 11) Sampling is deterministic for a seed, respects the cap and
  returns chunks in reading order.
- **AC12** (Req 12) The CLI prints the "Synthesized" summary with counts and the
  dataset path.
- **AC13** (Req 13) The offline end-to-end synth test completes with the fake
  LLM.
- **AC14** (Req 14) `--json` prints the manifest and it matches the written
  file.
- **AC15** (Req 15) An unindexed book exits non-zero with "distiller index" in
  the message and no traceback.

## Non-Goals

- Multi-hop / cross-chapter question generation.
- LLM-judged (RAGAS-scored) dataset filtering.
- Concurrency or batching of generation calls.
- Embedding-based near-duplicate detection across the dataset.
- Training itself (Phase 3) and dataset mixing with general instruction data.

## File-change Plan

| Action | Path | Purpose |
|--------|------|---------|
| create | `src/distiller/synthesis/__init__.py` | Package exports |
| create | `src/distiller/synthesis/qa.py` | Prompts, JSON parsing, pair generation |
| create | `src/distiller/synthesis/filtering.py` | Deterministic filters and reasons |
| create | `src/distiller/synthesis/raft.py` | RAFT example builder |
| create | `src/distiller/synthesis/dataset.py` | Sampling, manifest, orchestration |
| modify | `src/distiller/models.py` | Move `refusal_text` here (shared prompt/training target) |
| modify | `src/distiller/rag/prompts.py` | Import the shared refusal text |
| modify | `src/distiller/config.py` | `SynthesisSettings` + `Settings.synthesis` |
| modify | `src/distiller/paths.py` | Dataset artifact paths |
| modify | `src/distiller/llm/fake.py` | Deterministic `<chunk>` QA branch |
| modify | `src/distiller/cli/context.py` | `load_book_and_chunks` |
| modify | `src/distiller/cli/main.py` | The `synth` command |
| modify | `src/distiller/cli/render.py` | `render_synthesis` summary table |
| create | `tests/unit/test_synthesis_{qa,filtering,raft,dataset}.py` | Unit tests |
| modify | `tests/conftest.py` | `corpus_factory` fixture |
| modify | `tests/integration/test_cli_end_to_end.py` | CLI synth coverage |
| modify | `README.md`, `AGENTS.md` | Document the pipeline and roadmap status |

## Test Plan

| AC | Test file | Test name |
|----|-----------|-----------|
| AC1 | `tests/unit/test_synthesis_dataset.py` | `test_synthesize_generates_filters_and_formats` |
| AC2 | `tests/unit/test_synthesis_qa.py` | `test_parse_pairs_accepts_plain_json` |
| AC2 | `tests/unit/test_synthesis_qa.py` | `test_generate_pairs_uses_the_fake_llm_and_records_provenance` |
| AC3 | `tests/unit/test_synthesis_qa.py` | `test_generate_pairs_prompt_requests_the_exact_count` |
| AC4 | `tests/unit/test_synthesis_qa.py` | `test_parse_pairs_recovers_json_from_prose_and_fences` |
| AC4 | `tests/unit/test_synthesis_qa.py` | `test_parse_pairs_returns_empty_on_garbage` |
| AC4 | `tests/unit/test_synthesis_qa.py` | `test_parse_pairs_drops_malformed_items` |
| AC4 | `tests/unit/test_synthesis_qa.py` | `test_generate_pairs_tolerates_llm_errors_and_bad_json` |
| AC5 | `tests/unit/test_synthesis_filtering.py` | `test_keeps_a_grounded_pair` |
| AC5 | `tests/unit/test_synthesis_filtering.py` | `test_rejects_a_pair_whose_quotes_are_not_in_the_chunk` |
| AC5 | `tests/unit/test_synthesis_filtering.py` | `test_drops_failing_quotes_but_keeps_the_pair` |
| AC5 | `tests/unit/test_synthesis_filtering.py` | `test_rejects_pairs_with_only_short_quotes` |
| AC5 | `tests/unit/test_synthesis_filtering.py` | `test_rejects_duplicate_questions` |
| AC5 | `tests/unit/test_synthesis_filtering.py` | `test_rejects_questions_that_echo_their_answer` |
| AC5 | `tests/unit/test_synthesis_filtering.py` | `test_rejects_pairs_that_are_too_short` |
| AC6 | `tests/unit/test_synthesis_raft.py` | `test_positive_example_includes_golden_and_distractors` |
| AC6 | `tests/unit/test_synthesis_raft.py` | `test_same_seed_is_reproducible` |
| AC7 | `tests/unit/test_synthesis_raft.py` | `test_positive_example_includes_golden_and_distractors` |
| AC7 | `tests/unit/test_synthesis_raft.py` | `test_zero_distractors_keeps_only_the_golden_context` |
| AC8 | `tests/unit/test_synthesis_raft.py` | `test_negative_examples_omit_the_golden_chunk` |
| AC9 | `tests/unit/test_synthesis_dataset.py` | `test_manifest_records_index_identity_and_config` |
| AC9 | `tests/integration/test_cli_end_to_end.py` | `test_synth_command_end_to_end` |
| AC10 | `tests/unit/test_synthesis_dataset.py` | `test_synthesize_reuses_cached_pairs_without_calling_the_llm` |
| AC10 | `tests/integration/test_cli_end_to_end.py` | `test_synth_command_end_to_end` |
| AC11 | `tests/unit/test_synthesis_dataset.py` | `test_sample_chunks_is_deterministic_and_in_reading_order` |
| AC11 | `tests/unit/test_synthesis_dataset.py` | `test_sample_chunks_returns_everything_when_cap_exceeds_corpus` |
| AC12 | `tests/integration/test_cli_end_to_end.py` | `test_synth_command_end_to_end` |
| AC13 | `tests/unit/test_synthesis_dataset.py` | `test_synthesize_generates_filters_and_formats` |
| AC14 | `tests/integration/test_cli_end_to_end.py` | `test_synth_command_end_to_end` |
| AC15 | `tests/integration/test_cli_end_to_end.py` | `test_synth_requires_an_index` |

## Open Questions

- Decision (2026-09-30): deterministic filters only; RAGAS-scored filtering is
  deferred until the raw yield from a real teacher is understood.
- Decision (2026-09-30): negatives are sampled per pair with probability
  `negative_ratio`, keeping the dataset deterministic for a given seed.

## Outcomes

- **Implemented in**: `824ef9d` (spec), `c0e170e` (refusal refactor), `7a6d245` (synthesis), `7aa2cfc` (CLI), `fb7933e` (docs)
- **Spec archived**: 2026-09-30
- **Post-mortem notes**: pydantic resolves field annotations at runtime, so
  `QAPair` had to be a runtime import in `filtering.py`. The fake LLM learned a
  `<chunk>` branch, which made the whole pipeline testable offline; the smoke
  test confirmed a second run costs zero LLM calls.
