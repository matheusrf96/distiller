# Tasks — contextual-retrieval

## Spec Phase

- [x] Write `requirements.md` (EARS functional + constraints)
- [x] Write `design.md` (data model / module / config / CLI deltas)
- [x] Write this `tasks.md`

## Test Phase

- [x] Write failing unit test(s) covering REQ-CR-009 (normalization rules)
- [x] Write failing unit test(s) covering REQ-CR-003 (cache reuse, no repeat LLM calls)
- [x] Write failing unit test(s) covering REQ-CR-004 (LLM failure tolerance)
- [x] Write failing unit test(s) covering REQ-CR-010 (prompt carries book/chapter/section)
- [x] Write failing unit test(s) covering REQ-CR-002 (index_text used for embedding + BM25)
- [x] Write failing unit test(s) covering REQ-CR-005/REQ-CR-007 (disabled default, metadata flags)
- [x] Write failing integration test(s): CLI `index --contextual` end to end (REQ-CR-008/REQ-CR-012)
- [x] Write failing unit test covering REQ-CR-011 (eval report index block)

## Implementation Phase

- [x] Add `EnrichmentSettings` + `Settings.enrichment` (`config.py`)
- [x] Add `Chunk.context` + `Chunk.index_text` (`models.py`)
- [x] Add `BookPaths.enrichment_jsonl` (`paths.py`)
- [x] Implement `enrichment/contextual.py` (prompt, cache, normalize, tolerate failures)
- [x] Wire enrichment into `build_index`; index `index_text` on both dense and BM25 sides
- [x] Record `contextual` / `enriched_chunks` in `index/metadata.json`
- [x] Add `--contextual/--no-contextual` to `distiller index`
- [x] Add index block to `eval/report.json` (REQ-CR-011)
- [x] Extend `FakeLLM` to answer `<excerpt>` prompts (REQ-CR-012)
- [x] Update `README.md` config section and `AGENTS.md` roadmap status

## Verification Phase

- [x] All tests pass: `make test` (96 passed)
- [x] Type check passes: `uv run mypy` (41 files, strict)
- [x] Format/lint/security pass: `make format`
- [x] Coverage gate passes (80%): `make coverage` (85.5%)
- [x] Manual smoke: `distiller index --contextual` on the demo fixture, then `ask`
- [x] Spec docs updated to reflect any changes during implementation
- [x] Code reviewed against `docs/CONTRIBUTING.md`

## Notes from implementation

- Contexts are cached by chunk id; the id embeds a content hash, so re-chunking
  invalidates safely and a re-index costs zero LLM calls (verified in smoke test).
- The prompt contains the book title, so LLM stubs in tests must branch on chunk
  content, not on words that also appear in the title (learned the hard way).
- `FakeLLM` strips trailing punctuation before appending its sentence to avoid
  double periods in generated contexts.
