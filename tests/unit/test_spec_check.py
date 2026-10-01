"""Unit tests for the SDD spec-tree validator (`make spec-check`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from distiller.spec_check import check_specs, main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

REQUIREMENTS = (
    "1. **REQ-TX-001** — WHEN a thing happens, THEN the system SHALL do the thing.\n"
    "2. **REQ-TX-002** — IF another thing happens, THEN the system SHALL do "
    "the other thing."
)

ACCEPTANCE = """- **AC1** (Req 1) The thing happens.
- **AC2** (Req 2) The other thing happens."""

TEST_PLAN = """| AC | Test file | Test name |
|----|-----------|-----------|
| AC1 | `tests/unit/test_x.py` | `test_thing` |
| AC2 | `tests/unit/test_x.py` | `test_other_thing` |"""

OUTCOMES_BODY = """- **Implemented in**: `abc1234`
- **Spec archived**: 2026-09-30
- **Post-mortem notes**: none"""

DIRECTORY_BY_STATUS = {
    "draft": "drafts",
    "approved": "active",
    "implementing": "active",
    "done": "archived",
    "superseded": "archived",
}


def spec_text(
    *,
    spec_id: str = "SDD-0001",
    status: str = "done",
    sections: dict[str, str | None] | None = None,
    outcomes: bool = True,
) -> str:
    """Build a valid spec document, with optional section overrides/removals."""
    body: dict[str, str | None] = {
        "Problem": "Something is broken.",
        "Context": "Some context.",
        "Requirements": REQUIREMENTS,
        "Acceptance Criteria": ACCEPTANCE,
        "Non-Goals": "- Nothing else.",
        "File-change Plan": (
            "| Action | Path | Purpose |\n|---|---|---|\n| create | `x.py` | Thing |"
        ),
        "Test Plan": TEST_PLAN,
        "Open Questions": "- none",
    }
    for name, value in (sections or {}).items():
        if value is None:
            body.pop(name, None)
        else:
            body[name] = value
    if outcomes:
        body["Outcomes"] = OUTCOMES_BODY

    parts = [
        "---",
        f"id: {spec_id}",
        f"status: {status}",
        "supersedes:",
        "owner: matheus",
        "created: 2026-09-30",
        "---",
        "",
        "# Test Feature",
        "",
    ]
    for name, text in body.items():
        parts.append(f"## {name}")
        parts.append("")
        parts.append(text or "")
        parts.append("")
    return "\n".join(parts)


def build_spec_tree(
    tmp_path: Path,
    *,
    spec_id: str = "SDD-0001",
    status: str = "done",
    directory: str | None = None,
    text: str | None = None,
    readme: bool = True,
) -> Path:
    """Create a specs/ tree with one spec, valid by default."""
    specs = tmp_path / "specs"
    for name in ("drafts", "active", "archived"):
        (specs / name).mkdir(parents=True, exist_ok=True)
    (specs / "TEMPLATE.md").write_text("# Template\n", encoding="utf-8")

    directory = directory or DIRECTORY_BY_STATUS[status]
    path = specs / directory / f"{spec_id}-test-feature.md"
    if readme:
        (specs / "README.md").write_text(
            f"# Specs\n\n- [{spec_id}]({directory}/{path.name})\n", encoding="utf-8"
        )

    path.write_text(
        text if text is not None else spec_text(spec_id=spec_id, status=status),
        encoding="utf-8",
    )
    return path


def test_valid_archived_spec_passes(tmp_path: Path) -> None:
    """A complete, done spec produces no findings."""
    build_spec_tree(tmp_path)

    assert check_specs(tmp_path / "specs") == []


def test_valid_draft_spec_passes(tmp_path: Path) -> None:
    """A draft in drafts/ with open work is fine."""
    build_spec_tree(
        tmp_path,
        status="draft",
        text=spec_text(status="draft", outcomes=False),
    )

    assert check_specs(tmp_path / "specs") == []


def test_missing_lifecycle_directory_is_reported(tmp_path: Path) -> None:
    """The drafts/active/archived scaffolding is required."""
    build_spec_tree(tmp_path)
    (tmp_path / "specs" / "drafts").rmdir()

    findings = check_specs(tmp_path / "specs")

    assert any("missing drafts" in finding.message for finding in findings)


def test_bad_file_name_is_reported(tmp_path: Path) -> None:
    """Spec files must be named SDD-NNNN-kebab-description.md."""
    path = build_spec_tree(tmp_path)
    path.rename(path.parent / "not-a-spec.md")

    findings = check_specs(tmp_path / "specs")

    assert any("file name must match" in finding.message for finding in findings)


def test_id_must_match_file_name(tmp_path: Path) -> None:
    """Front-matter id and file name prefix must agree."""
    build_spec_tree(tmp_path, text=spec_text(spec_id="SDD-0042", status="done"))

    findings = check_specs(tmp_path / "specs")

    assert any("does not match file name" in finding.message for finding in findings)


def test_duplicate_ids_are_reported(tmp_path: Path) -> None:
    """Two specs cannot share an id."""
    build_spec_tree(tmp_path)
    second = tmp_path / "specs" / "archived" / "SDD-0001-other-feature.md"
    second.write_text(spec_text(spec_id="SDD-0001", status="done"), encoding="utf-8")

    findings = check_specs(tmp_path / "specs")

    assert any("duplicate id" in finding.message for finding in findings)


def test_status_must_match_directory(tmp_path: Path) -> None:
    """A done spec belongs in archived/, not drafts/."""
    build_spec_tree(tmp_path, status="done", directory="drafts")

    findings = check_specs(tmp_path / "specs")

    assert any(
        "must live in specs/archived/" in finding.message for finding in findings
    )


def test_missing_section_is_reported(tmp_path: Path) -> None:
    """Every template section is required."""
    build_spec_tree(tmp_path, text=spec_text(sections={"Non-Goals": None}))

    findings = check_specs(tmp_path / "specs")

    assert any("missing '## Non-Goals'" in finding.message for finding in findings)


def test_requirements_must_be_numbered_sequentially(tmp_path: Path) -> None:
    """Requirement numbering must run 1..N."""
    broken = (
        "1. **REQ-TX-001** — WHEN x, THEN the system SHALL y.\n"
        "3. **REQ-TX-003** — WHEN z, THEN the system SHALL w."
    )
    build_spec_tree(tmp_path, text=spec_text(sections={"Requirements": broken}))

    findings = check_specs(tmp_path / "specs")

    assert any("not numbered 1..N" in finding.message for finding in findings)


def test_ac_referencing_missing_requirement_is_reported(tmp_path: Path) -> None:
    """Every AC must point at an existing requirement number."""
    broken = "- **AC1** (Req 1) Fine.\n- **AC2** (Req 9) Dangling."
    build_spec_tree(tmp_path, text=spec_text(sections={"Acceptance Criteria": broken}))

    findings = check_specs(tmp_path / "specs")

    assert any("missing requirement 9" in finding.message for finding in findings)


def test_every_ac_needs_a_test_plan_row(tmp_path: Path) -> None:
    """AC2 without a Test Plan row is a finding."""
    plan = (
        "| AC | Test file | Test name |\n"
        "|----|-----------|-----------|\n"
        "| AC1 | `t.py` | `test_x` |"
    )
    build_spec_tree(tmp_path, text=spec_text(sections={"Test Plan": plan}))

    findings = check_specs(tmp_path / "specs")

    assert any(
        "AC2 has no row in the Test Plan" in finding.message for finding in findings
    )


def test_unknown_ac_in_test_plan_is_reported(tmp_path: Path) -> None:
    """Test Plan rows for non-existent ACs are findings."""
    plan = TEST_PLAN + "\n| AC9 | `t.py` | `test_ghost` |"
    build_spec_tree(tmp_path, text=spec_text(sections={"Test Plan": plan}))

    findings = check_specs(tmp_path / "specs")

    assert any("unknown AC9" in finding.message for finding in findings)


def test_done_spec_requires_filled_outcomes(tmp_path: Path) -> None:
    """Done specs need Outcomes with real commit SHAs."""
    build_spec_tree(tmp_path, text=spec_text(outcomes=False))

    findings = check_specs(tmp_path / "specs")

    assert any("Outcomes section is empty" in finding.message for finding in findings)


def test_placeholder_outcomes_are_reported(tmp_path: Path) -> None:
    """The template placeholder must be replaced on archival."""
    placeholder = (
        "## Outcomes\n\n"
        "- **Implemented in**: <commit SHAs>\n"
        "- **Spec archived**: <date>"
    )
    build_spec_tree(
        tmp_path,
        text=spec_text(sections={"Outcomes": None}) + placeholder,
    )

    findings = check_specs(tmp_path / "specs")

    assert any("placeholder" in finding.message for finding in findings)


def test_missing_readme_entry_is_reported(tmp_path: Path) -> None:
    """Every spec must be linked from specs/README.md."""
    build_spec_tree(tmp_path, readme=False)
    (tmp_path / "specs" / "README.md").write_text("# Specs\n", encoding="utf-8")

    findings = check_specs(tmp_path / "specs")

    assert any(
        "not linked from specs/README.md" in finding.message for finding in findings
    )


def test_main_exit_codes(tmp_path: Path, capsys: pytest.LogCaptureFixture) -> None:
    """main() returns 0 for a valid tree and 1 with findings, printing a report."""
    path = build_spec_tree(tmp_path)
    assert main([str(tmp_path / "specs")]) == 0
    assert "spec-check: OK (1 spec)" in capsys.readouterr().out

    path.write_text(spec_text(sections={"Non-Goals": None}), encoding="utf-8")
    assert main([str(tmp_path / "specs")]) == 1
    assert "spec-check: FAILED" in capsys.readouterr().out


def test_missing_specs_directory_is_reported(tmp_path: Path) -> None:
    """A missing specs/ directory is a finding, not a crash."""
    findings = check_specs(tmp_path / "missing")

    assert any("specs directory not found" in finding.message for finding in findings)


def test_missing_front_matter_is_reported(tmp_path: Path) -> None:
    """Specs without YAML front-matter are findings."""
    build_spec_tree(tmp_path, text="# No front matter\n\n## Problem\n")

    findings = check_specs(tmp_path / "specs")

    assert any("missing YAML front-matter" in finding.message for finding in findings)


def test_invalid_status_and_missing_fields_are_reported(tmp_path: Path) -> None:
    """Invalid statuses and missing owner/created fields are findings."""
    text = spec_text(status="bogus").replace("owner: matheus\n", "")
    build_spec_tree(tmp_path, status="bogus", directory="drafts", text=text)

    findings = check_specs(tmp_path / "specs")
    messages = [finding.message for finding in findings]

    assert any("invalid or missing status" in message for message in messages)
    assert any("missing owner" in message for message in messages)


def test_empty_requirements_and_acceptance_are_reported(tmp_path: Path) -> None:
    """Empty Requirements and Acceptance Criteria sections are findings."""
    build_spec_tree(
        tmp_path,
        text=spec_text(
            sections={"Requirements": "Nothing.", "Acceptance Criteria": "Nothing."}
        ),
    )

    findings = check_specs(tmp_path / "specs")
    messages = [finding.message for finding in findings]

    assert any("no numbered requirements" in message for message in messages)
    assert any("no AC items" in message for message in messages)


def test_non_sequential_acceptance_is_reported(tmp_path: Path) -> None:
    """AC labels must run AC1..ACn."""
    broken = "- **AC1** (Req 1) Fine.\n- **AC3** (Req 2) Skips a label."
    build_spec_tree(tmp_path, text=spec_text(sections={"Acceptance Criteria": broken}))

    findings = check_specs(tmp_path / "specs")

    assert any("not sequential" in finding.message for finding in findings)


def test_done_spec_needs_the_implemented_line(tmp_path: Path) -> None:
    """Done specs need the 'Implemented in' line in Outcomes."""
    text = spec_text(outcomes=False) + "\n## Outcomes\n\n- Something else.\n"
    build_spec_tree(tmp_path, text=text)

    findings = check_specs(tmp_path / "specs")

    assert any("'Implemented in'" in finding.message for finding in findings)


def test_dir_for_unknown_status_falls_back_to_drafts() -> None:
    """The defensive status lookup falls back to drafts/."""
    from distiller.spec_check import _dir_for

    assert _dir_for("nonexistent") == "drafts"
