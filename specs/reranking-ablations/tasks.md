# Tasks — reranking-ablations

## Spec Phase

- [x] Write `requirements.md` (EARS functional + constraints)
- [x] Write `design.md` (models / runner / CLI deltas)
- [x] Write this `tasks.md`

## Test Phase

- [x] Write failing unit test(s) covering REQ-RA-002/REQ-RA-003 (variant construction)
- [x] Write failing unit test(s) covering REQ-RA-004/REQ-RA-011 (skips and failures recorded)
- [x] Write failing unit test(s) covering REQ-RA-006/REQ-RA-012 (report shape + shared metrics)
- [x] Write failing unit test(s) covering REQ-RA-007 (deltas versus baseline)
- [x] Write failing unit test(s) covering REQ-RA-009/`--top-k` parsing (CLI helpers)
- [x] Write failing integration test(s): `distiller ablation` end to end, skip path (REQ-RA-004)
- [x] Write failing integration test(s): rerank path with a stub reranker (REQ-RA-001)

## Implementation Phase

- [x] Implement `evaluation/ablation.py` (models, `build_variants`, `run_ablation`, `metric_deltas`)
- [x] Export the new symbols from `evaluation/__init__.py`
- [x] Implement `context.parse_top_k_values` and `context.build_ablation_runs`
- [x] Implement `render.render_ablation`
- [x] Add the `distiller ablation` command (REQ-RA-008/REQ-RA-010)
- [x] Update `README.md` (command, quickstart) and `AGENTS.md` (roadmap, test count)

## Verification Phase

- [x] All tests pass: `make test` (106 passed)
- [x] Type check passes: `uv run mypy` (42 files, strict)
- [x] Format/lint/security pass: `make format`
- [x] Coverage gate passes (80%): `make coverage` (86.5%)
- [x] Manual smoke: `distiller ablation` on the demo fixture (skip path) and with a stubbed reranker
- [x] Spec docs updated to reflect any changes during implementation
- [x] Code reviewed against `docs/CONTRIBUTING.md`

## Notes from implementation

- The report is written to `eval/ablation.json` on every run (including `--json`
  mode), mirroring how `eval` always writes `eval/report.json`.
- Metric column labels are short (`hit rate`, `contains`, `citations`, `refusal`)
  because the full metric names truncate badly on 80-column terminals.
- Rerank availability is decided by `optional_deps.is_available`, so the skip
  path costs nothing and needs no import attempt.
- `build_ablation_runs` binds each variant's pipeline with
  `functools.partial(pipeline.ask)` (no late-binding closure bugs).
