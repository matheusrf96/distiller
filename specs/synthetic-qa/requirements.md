# Requirements — synthetic-qa

## Context

Phase 2 is the bridge between retrieval and training: turn an indexed book into
supervised fine-tuning data. A teacher LLM writes grounded question-answer pairs
from the book's own chunks; deterministic filters keep only pairs whose quotes
are verifiable against the source chunk; a RAFT formatter then produces training
examples with the golden chunk plus distractor chunks, chain-of-thought answers
that cite verbatim, and explicit "not in the book" negatives. Phase 3 trains on
this dataset.

---

## Functional Requirements

Each requirement uses EARS (Easy Approach to Requirements Syntax):

    WHEN <trigger>
    THE SYSTEM SHALL <response>
    WHILE <precondition / invariant>

### SYNTHETIC-QA

- **REQ-SQ-001**: WHEN `distiller synth` runs THE SYSTEM SHALL generate grounded question-answer pairs for sampled chunks with the configured LLM and write them to `dataset/qa.jsonl`.
- **REQ-SQ-002**: WHEN generating pairs THE SYSTEM SHALL require verbatim quotes from the source chunk in each pair and SHALL persist them with the pair.
- **REQ-SQ-003**: WHEN `questions_per_chunk` is N THE SYSTEM SHALL request exactly N questions per sampled chunk in a single LLM call.
- **REQ-SQ-004**: WHEN a completion is not valid JSON, or an item is malformed, THE SYSTEM SHALL skip that chunk or item, log a warning and continue.
- **REQ-SQ-005**: WHEN filtering pairs THE SYSTEM SHALL drop quotes that do not appear verbatim in the source chunk, reject a pair whose quotes all fail, reject duplicate questions and reject questions that nearly echo their answer, counting every rejection by reason.
- **REQ-SQ-006**: WHEN building RAFT examples THE SYSTEM SHALL place the golden chunk among `distractors` other chunks from the same book in a seeded random order.
- **REQ-SQ-007**: WHEN a pair is answerable THE SYSTEM SHALL format the target answer as the primary verbatim quote with the golden context's citation index, followed by the answer text.
- **REQ-SQ-008**: WHEN a negative example is built THE SYSTEM SHALL provide distractor contexts only and target the refusal sentence shared with the RAG prompt.
- **REQ-SQ-009**: WHEN the dataset completes THE SYSTEM SHALL write `dataset/raft.jsonl`, `dataset/rejected.jsonl` and `dataset/manifest.json` containing counts, configuration, model and index identity.
- **REQ-SQ-010**: WHEN `dataset/qa.jsonl` already exists THE SYSTEM SHALL reuse it instead of calling the LLM again unless `--regenerate` is given, recording the source in the manifest.
- **REQ-SQ-011**: WHEN sampling chunks THE SYSTEM SHALL do so deterministically from the seed, capped by `--max-chunks`, in reading order.
- **REQ-SQ-012**: WHEN the command finishes THE SYSTEM SHALL print a summary table with generated/kept/rejected/example counts and the output paths.
- **REQ-SQ-013**: WHEN the offline fake LLM is configured THE SYSTEM SHALL complete the whole synth pipeline (for tests and CI).
- **REQ-SQ-014**: WHEN `--json` is given THE SYSTEM SHALL print the manifest as JSON instead of the table.
- **REQ-SQ-015**: WHEN the book has not been indexed THE SYSTEM SHALL fail with an actionable message telling the user to run `distiller index` first.

---

## Constraints / Non-Functional

- **CON-SQ-001**: No new runtime dependencies; the teacher is any `LLMClient` (cloud API or local).
- **CON-SQ-002**: Synthesis needs no embedder: chunks come from `chunks.jsonl`, so it runs on a plain install.
- **CON-SQ-003**: Dataset artifacts live under `artifacts/<book>/dataset/` (qa, rejected, raft, manifest).
- **CON-SQ-004**: Given (seed, configuration, chunks) the dataset is byte-reproducible.
- **CON-SQ-005**: `mypy --strict` clean, 88-char lines, coverage gate (80%) stays green.
- **CON-SQ-006**: Fully testable offline with stub/fake LLMs — no network, no model downloads.

---

## Out of Scope

- Multi-hop / cross-chapter question generation.
- LLM-judged (RAGAS-scored) dataset filtering — deterministic filters only in this iteration.
- Concurrency or batching of generation calls.
- Embedding-based near-duplicate detection across the whole dataset.
- Training itself (Phase 3) and dataset mixing with general instruction data.
