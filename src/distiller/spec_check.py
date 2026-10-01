"""Repository tooling: validate the SDD spec tree under ``specs/``.

Not part of the shipped pipeline. It enforces the spec-driven workflow
(``specs/README.md``) and runs as ``make spec-check`` / ``make check``.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LIFECYCLE_DIRS = ("drafts", "active", "archived")
STATUS_BY_DIR = {
    "drafts": frozenset({"draft"}),
    "active": frozenset({"approved", "implementing"}),
    "archived": frozenset({"done", "superseded"}),
}
VALID_STATUSES = frozenset({"draft", "approved", "implementing", "done", "superseded"})
REQUIRED_SECTIONS = (
    "Problem",
    "Context",
    "Requirements",
    "Acceptance Criteria",
    "Non-Goals",
    "File-change Plan",
    "Test Plan",
)

_SPEC_FILE = re.compile(r"^SDD-\d{4}-[a-z0-9-]+\.md$")
_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_REQUIREMENT = re.compile(r"^(\d+)\.\s+\*\*(REQ-[A-Z0-9]+-\d{3})\*\*", re.MULTILINE)
_ACCEPTANCE = re.compile(r"^- \*\*(AC\d+)\*\*\s*\(Req (\d+)\)", re.MULTILINE)
_TEST_PLAN_ROW = re.compile(r"^\|\s*(AC\d+)\s*\|", re.MULTILINE)
_SECTION = re.compile(r"^## (.+)$", re.MULTILINE)
_PLACEHOLDER = "<commit SHAs>"


@dataclass(frozen=True, slots=True)
class Finding:
    """One validation problem.

    Attributes:
        spec: Spec id, or ``specs`` for repo-level findings.
        message: Human-readable description of the problem.
    """

    spec: str
    message: str


def check_specs(specs_dir: Path) -> list[Finding]:
    """Validate the SDD spec tree.

    Args:
        specs_dir: Repository ``specs/`` directory.

    Returns:
        Findings; an empty list means the tree is valid.
    """
    specs_dir = Path(specs_dir)
    if not specs_dir.is_dir():
        return [Finding("specs", f"specs directory not found: {specs_dir}")]

    findings = _check_scaffolding(specs_dir)
    readme = (specs_dir / "README.md").read_text(encoding="utf-8")
    seen_ids: dict[str, str] = {}

    for directory in LIFECYCLE_DIRS:
        for path in sorted((specs_dir / directory).glob("*.md")):
            findings.extend(_check_spec(path, directory, readme, seen_ids))
    return findings


def _check_scaffolding(specs_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    for required in ("README.md", "TEMPLATE.md", *LIFECYCLE_DIRS):
        if not (specs_dir / required).exists():
            findings.append(Finding("specs", f"missing {required}"))
    return findings


def _check_spec(
    path: Path, directory: str, readme: str, seen_ids: dict[str, str]
) -> list[Finding]:
    name = path.name
    if not _SPEC_FILE.match(name):
        return [Finding(name, "file name must match SDD-NNNN-kebab-description.md")]

    text = path.read_text(encoding="utf-8")
    front_matter = _parse_front_matter(text)
    if front_matter is None:
        return [Finding(name, "missing YAML front-matter")]

    sections = _sections(text)
    findings = _check_identity(name, front_matter, directory, seen_ids)
    findings.extend(_check_sections(name, sections))
    findings.extend(_check_requirements_and_acs(name, sections))
    if front_matter.get("status") == "done":
        findings.extend(_check_outcomes(name, sections))

    spec_id = str(front_matter.get("id", ""))
    if name not in readme:
        findings.append(
            Finding(name, f"{spec_id or name} is not linked from specs/README.md")
        )
    return findings


def _check_identity(
    name: str, front_matter: dict[str, Any], directory: str, seen_ids: dict[str, str]
) -> list[Finding]:
    findings: list[Finding] = []
    spec_id = str(front_matter.get("id", ""))
    expected_id = name[:8]
    if spec_id != expected_id:
        findings.append(
            Finding(name, f"front-matter id {spec_id!r} does not match file name")
        )
    elif spec_id in seen_ids:
        findings.append(
            Finding(name, f"duplicate id {spec_id} (also {seen_ids[spec_id]})")
        )
    else:
        seen_ids[spec_id] = name

    status = front_matter.get("status")
    if status not in VALID_STATUSES:
        findings.append(Finding(name, f"invalid or missing status: {status!r}"))
    elif status not in STATUS_BY_DIR[directory]:
        findings.append(
            Finding(name, f"status {status!r} must live in specs/{_dir_for(status)}/")
        )
    for field in ("owner", "created"):
        if not front_matter.get(field):
            findings.append(Finding(name, f"missing {field} in front-matter"))
    return findings


def _check_sections(name: str, sections: dict[str, str]) -> list[Finding]:
    return [
        Finding(name, f"missing '## {required}' section")
        for required in REQUIRED_SECTIONS
        if required not in sections
    ]


def _check_requirements_and_acs(name: str, sections: dict[str, str]) -> list[Finding]:
    findings: list[Finding] = []
    requirements = _REQUIREMENT.findall(sections.get("Requirements", ""))
    numbers = [int(number) for number, _ in requirements]
    if not requirements:
        findings.append(Finding(name, "no numbered requirements with REQ-* ids"))
    elif numbers != list(range(1, len(numbers) + 1)):
        findings.append(Finding(name, f"requirements are not numbered 1..N: {numbers}"))

    acceptance = _ACCEPTANCE.findall(sections.get("Acceptance Criteria", ""))
    ac_labels = [label for label, _ in acceptance]
    expected_labels = [f"AC{index}" for index in range(1, len(ac_labels) + 1)]
    if not acceptance:
        findings.append(Finding(name, "no AC items with '(Req N)' references"))
    elif ac_labels != expected_labels:
        findings.append(
            Finding(name, f"acceptance criteria are not sequential: {ac_labels}")
        )
    for _, requirement_number in acceptance:
        if int(requirement_number) not in numbers:
            findings.append(
                Finding(name, f"AC references missing requirement {requirement_number}")
            )

    planned = set(_TEST_PLAN_ROW.findall(sections.get("Test Plan", "")))
    for label in ac_labels:
        if label not in planned:
            findings.append(Finding(name, f"{label} has no row in the Test Plan"))
    for label in sorted(planned - set(ac_labels)):
        findings.append(Finding(name, f"Test Plan references unknown {label}"))
    return findings


def _check_outcomes(name: str, sections: dict[str, str]) -> list[Finding]:
    outcomes = sections.get("Outcomes", "")
    if not outcomes.strip():
        return [Finding(name, "status is done but the Outcomes section is empty")]
    if "**Implemented in**" not in outcomes:
        return [Finding(name, "Outcomes is missing the 'Implemented in' line")]
    if _PLACEHOLDER in outcomes:
        return [Finding(name, "Outcomes still contains the template placeholder")]
    return []


def _dir_for(status: str) -> str:
    for directory, statuses in STATUS_BY_DIR.items():
        if status in statuses:
            return directory
    return "drafts"


def _parse_front_matter(text: str) -> dict[str, Any] | None:
    match = _FRONT_MATTER.match(text)
    if match is None:
        return None
    parsed = yaml.safe_load(match.group(1))
    return parsed if isinstance(parsed, dict) else None


def _sections(text: str) -> dict[str, str]:
    """Split the document into ``## Section`` bodies (code fences stripped)."""
    body = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    matches = list(_SECTION.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[match.group(1).strip()] = body[match.end() : end]
    return sections


def main(argv: list[str] | None = None) -> int:
    """Validate the spec tree and print a report.

    Args:
        argv: Optional arguments (a single optional ``specs/`` path override).

    Returns:
        Process exit code: 0 when valid, 1 when findings exist.
    """
    arguments = sys.argv[1:] if argv is None else argv
    specs_dir = Path(arguments[0]) if arguments else Path("specs")
    findings = check_specs(specs_dir)
    if not findings:
        count = sum(
            len(list((specs_dir / directory).glob("*.md")))
            for directory in LIFECYCLE_DIRS
        )
        label = "spec" if count == 1 else "specs"
        print(f"spec-check: OK ({count} {label})")
        return 0
    for finding in findings:
        print(f"spec-check: {finding.spec}: {finding.message}")
    print(f"spec-check: FAILED ({len(findings)} finding(s))")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
