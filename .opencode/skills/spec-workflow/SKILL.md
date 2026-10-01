---
description: End-to-end spec-driven development workflow — author, review, approve, implement, archive
---

# Spec Workflow Skill

Orchestrates the full SDD lifecycle from initial draft to archived, done spec.
Use this skill when starting a new feature that requires a formal spec.

See `specs/README.md` for the workflow guide and `specs/TEMPLATE.md` for the
canonical spec format.

## Workflow

### Phase 1: Author

1. Ask the user for the feature intent (problem to solve, expected outcome)
2. Use the `spec-author` agent to produce a draft in `specs/drafts/` following
   `specs/TEMPLATE.md` (assign the next `SDD-NNNN`)
3. Read the draft back to confirm the user agrees with the direction

### Phase 2: Review

1. Use the `spec-reviewer` agent to validate the draft
2. Present the review report to the user
3. If FAIL: work with the user to fix issues → re-run the reviewer
4. Repeat until `spec-reviewer` passes all sections

### Phase 3: Approve

1. Ask the user explicitly: "Approve this spec for implementation?"
2. On approval:
   - Change `status` from `draft` to `approved` in the front-matter
   - Move the file from `specs/drafts/` to `specs/active/`
   - Confirm the move with the user

### Phase 4: Implement

1. Pass the active spec ID to the `feature-implementer` agent
2. The implementer sets the spec front-matter `status` to `implementing`
3. The implementer reads the spec and produces code that satisfies AC1..ACn,
   test-first (failing test → implement → green)
4. After each commit context, the implementer prints an **AC coverage report**:

   ```
   AC1: covered by tests/unit/test_synthesis_qa.py::test_parse_pairs_accepts_plain_json
   AC2: covered by tests/integration/test_cli_end_to_end.py::test_synth_command_end_to_end
   AC3: not yet covered — pending test
   ```

5. Run validation: `make format`, `make test`, `make spec-check`
6. If validation fails, fix and re-run
7. On successful validation, restore the spec front-matter `status` to
   `approved` (until the archive phase)

### Phase 5: Archive

1. Ensure all ACs are covered (no pending remaining)
2. All validation passes
3. Update front-matter:
   - `status` -> `done` (or `superseded` if replaced)
   - `archived` -> the archive date
4. Fill the `Outcomes` section:
   - Commit SHAs that implement each AC
   - Archive date
   - Optional post-mortem notes
5. Move the file from `specs/active/` to `specs/archived/`
6. Announce completion to the user

## Rules

- Never skip a phase — always go draft -> review -> approve -> implement -> archive
- The `spec-reviewer` must pass all sections before moving to the approve phase
- Do not start implementation without explicit user approval
- If the user cancels mid-workflow, leave the spec in its current state (e.g., still in `drafts/`)
- `make spec-check` must stay green: it enforces front-matter, lifecycle
  placement, sequential ACs tied to requirements, test-plan coverage and
  filled Outcomes for done specs
