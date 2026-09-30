# Tasks — synthetic-qa

## Spec Phase

- [x] Write `requirements.md` (EARS functional + constraints)
- [x] Write `design.md` (models / prompts / filters / RAFT / CLI deltas)
- [x] Write this `tasks.md`

## Test Phase

- [x] Write failing unit test(s) covering REQ-SQ-004 (JSON parsing: prose, fences, garbage)
- [x] Write failing unit test(s) covering REQ-SQ-002/REQ-SQ-003 (generation prompt + pairs)
- [x] Write failing unit test(s) covering REQ-SQ-005 (filter rules and reasons)
- [x] Write failing unit test(s) covering REQ-SQ-006/REQ-SQ-007/REQ-SQ-008 (RAFT contexts, citation index, negatives)
- [x] Write failing unit test(s) covering CON-SQ-004 (seeded reproducibility)
- [x] Write failing unit test(s) covering REQ-SQ-011 and manifest counts (REQ-SQ-009)
- [x] Write failing integration test(s): CLI `synth` end to end incl. reuse and `--json` (REQ-SQ-010/REQ-SQ-013/REQ-SQ-014)
- [x] Write failing integration test(s): unindexed book error (REQ-SQ-015)

## Implementation Phase

- [x] Move `refusal_text` to `models.py` and re-export from `rag.prompts`
- [x] Add `SynthesisSettings` + `Settings.synthesis` (`config.py`)
- [x] Add dataset paths to `BookPaths` (`paths.py`)
- [x] Implement `synthesis/qa.py` (prompt, parse, generate)
- [x] Implement `synthesis/filtering.py`
- [x] Implement `synthesis/raft.py`
- [x] Implement `synthesis/dataset.py` (sampling, manifest)
- [x] Add `FakeLLM` synth branch (REQ-SQ-013)
- [x] Add `context.load_book_and_chunks` and the `synth` command (REQ-SQ-012/014/015)
- [x] Implement `render.render_synthesis`
- [x] Update `README.md`, `AGENTS.md`, `specs/README.md`

## Verification Phase

- [x] All tests pass: `make test` (132 passed)
- [x] Type check passes: `uv run mypy` (47 files, strict)
- [x] Format/lint/security pass: `make format`
- [x] Coverage gate passes (80%): `make coverage` (87.6%)
- [x] Manual smoke: `distiller synth` on the demo fixture (fake LLM), reuse on second run
- [x] Spec docs updated to reflect any changes during implementation
- [x] Code reviewed against `docs/CONTRIBUTING.md`

## Notes from implementation

- Pydantic resolves model annotations at runtime, so `QAPair` had to be a runtime
  import in `filtering.py` (ruff's `runtime-evaluated-base-classes` config did not
  cover the locally imported `DomainModel` alias). Lesson: when a pydantic model
  has a field typed with another local model, import it at runtime.
- The fake LLM's QA branch derives its questions from the chunk text (`text[:30]`)
  and its quote from `text[:60]`; the question/answer echo filter (0.7 ratio) is
  why the question keyword is deliberately shorter than the quoted span.
- `random.Random(seed)` is flagged by bandit (`S311`); the `# noqa` documents that
  reproducibility, not cryptography, is the goal.
- `FakeLLM` now has four deterministic branches (`<excerpt>`, `<chunk>`, `<doc>`,
  refusal) — one per pipeline stage that talks to an LLM.
