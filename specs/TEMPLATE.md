---
id: SDD-NNNN
status: draft
supersedes:
owner:
created:
---

# Feature Title

<!-- One-line summary of the feature. -->

## Problem

<!-- What user or technical problem does this spec solve? 2-4 sentences. -->

## Context

<!--
- Links to related code (AGENTS.md sections, key files, modules)
- Related specs (SDD-XXXX) this builds on or conflicts with
- Existing behavior summary
- Design substance: data model / module / config / CLI deltas, failure modes,
  decisions and their rationale (as subsections)
-->

## Requirements

<!--
Use EARS notation. Canonical trigger keywords:
  WHEN       — event-driven trigger ("WHEN x, THEN the system SHALL y")
  IF / UNLESS — conditional
  WHILE      — concurrency/overlap
  WHERE      — feature-conditional
  AFTER      — sequence trigger
  BEFORE     — pre-condition trigger
  UNTIL      — temporal bound

Modal keywords:
  SHALL      — mandatory behaviour
  MUST       — hard requirement
  SHOULD     — nice-to-have

THEN is always allowed after any trigger keyword to express the response.

Number the requirements 1..N. This repository also gives each requirement a
stable id (`**REQ-<FEATURE>-NNN**`) so tests can reference it; keep it on the
requirement line.
-->

1. **REQ-XXXX-001** — WHEN <trigger>, THEN the system SHALL <response>.
2. **REQ-XXXX-002** — IF <condition>, THEN the system SHALL <response>.

## Acceptance Criteria

<!--
Numbered AC1..ACn. Each MUST be objectively testable and reference the
requirement it proves: "AC1 (Req 1) …".
-->

- **AC1** (Req 1) <description of verifiable outcome>
- **AC2** (Req 2) <description of verifiable outcome>

## Non-Goals

<!-- Explicitly out of scope. Prevents scope creep. -->

- <out of scope item>

## File-change Plan

| Action | Path | Purpose |
|--------|------|---------|
| modify | `src/distiller/…` | <purpose> |
| create | `src/distiller/…` | <purpose> |
| create | `tests/unit/…` | <purpose> |

## Test Plan

| AC | Test file | Test name |
|----|-----------|-----------|
| AC1 | `tests/unit/test_….py` | `test_…` |
| AC2 | `tests/integration/test_cli_end_to_end.py` | `test_…` |

## Open Questions

<!--
- <question>
- Decision (<date>): <answer>
-->

## Outcomes

<!-- Filled on archival. -->

- **Implemented in**: <commit SHAs>
- **Spec archived**: <date>
- **Post-mortem notes**: <optional>
