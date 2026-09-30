# Requirements — contextual-retrieval

## Context

Retrieval quality on a lone book is limited by chunks that are ambiguous out of
context ("it was painted white in the spring" — what was? which book chapter?).
Contextual retrieval prepends a short LLM-generated situating context to each
chunk before embedding and lexical indexing, while the original text remains
what the model reads and cites. This is Phase 1 of the distiller roadmap.

---

## Functional Requirements

Each requirement uses EARS (Easy Approach to Requirements Syntax):

    WHEN <trigger>
    THE SYSTEM SHALL <response>
    WHILE <precondition / invariant>

### CONTEXTUAL-RETRIEVAL

- **REQ-CR-001**: WHEN contextual enrichment is enabled and an index is built THE SYSTEM SHALL generate a short situating context for every chunk using the configured LLM WHILE never modifying `Chunk.text`.
- **REQ-CR-002**: WHEN embedding and lexical indexing a book THE SYSTEM SHALL index `Chunk.index_text` (context prefix + text) for chunks that carry a context WHILE leaving chunks without context unchanged.
- **REQ-CR-003**: WHEN a context for a chunk id already exists in the enrichment cache THE SYSTEM SHALL reuse it and SHALL NOT call the LLM for that chunk.
- **REQ-CR-004**: WHEN the LLM call for a chunk fails THE SYSTEM SHALL continue indexing that chunk without context and log a warning WHILE the rest of the run completes.
- **REQ-CR-005**: WHEN enrichment is disabled (the default) THE SYSTEM SHALL produce the same index content as before this feature (no contexts, no LLM calls).
- **REQ-CR-006**: WHEN an answer is generated THE SYSTEM SHALL pass the original `Chunk.text` to the model, never the synthetic context.
- **REQ-CR-007**: WHEN an index build completes THE SYSTEM SHALL record in `index/metadata.json` whether enrichment was enabled and how many chunks received a context.
- **REQ-CR-008**: WHEN the CLI is used THE SYSTEM SHALL allow enabling or disabling enrichment for a single index run via `--contextual/--no-contextual`.
- **REQ-CR-009**: WHEN a generated context is empty, shorter than a minimum, refusal-like, or longer than the configured cap THE SYSTEM SHALL normalize it (collapse whitespace, cap length) or discard it.
- **REQ-CR-010**: WHEN generating context THE SYSTEM SHALL provide the model the book title, chapter title, heading path and a bounded excerpt of the chapter for situating.
- **REQ-CR-011**: WHEN `distiller eval` writes `report.json` THE SYSTEM SHALL include the index's embedder identity and contextual flag so enriched and baseline runs are comparable.
- **REQ-CR-012**: WHEN the configured LLM is the offline fake THE SYSTEM SHALL still complete an enriched index build end to end (for tests and CI).

---

## Constraints / Non-Functional

- **CON-CR-001**: Enrichment is opt-in; a default build performs no additional LLM calls (preserves v1 behaviour and cost).
- **CON-CR-002**: The context cache lives at `artifacts/<book>/enrichment.jsonl`, append-only, keyed by chunk id (which embeds the chunk content hash, so re-chunking invalidates safely).
- **CON-CR-003**: No new runtime dependencies; enrichment uses the existing `LLMClient` abstraction.
- **CON-CR-004**: Fully testable offline: tests inject a stub LLM, no network, no model downloads.
- **CON-CR-005**: Contexts are capped (default 500 characters) to bound index size and embedding cost.
- **CON-CR-006**: `mypy --strict` clean, line length 88, coverage gate (80%) stays green.

---

## Out of Scope

- Reranking ablations and experiment tracking (separate follow-up spec).
- Graph/RAPTOR indexes and whole-book thematic search (Phase 5).
- Concurrency / batching of enrichment LLM calls (sequential first; revisit if books grow).
- Fine-tuning phases (Phase 2–4).
