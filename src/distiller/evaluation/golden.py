"""Golden question sets: the measuring stick for every iteration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from ..models import DomainModel


class GoldenItem(DomainModel):
    """One evaluation question with provenance expectations.

    Attributes:
        question: The question to ask the pipeline.
        answerable: True when the book contains the answer (False expects a refusal).
        expected_chapters: Chapter titles retrieval should surface.
        expected_answer_contains: Snippets the answer should contain.
        notes: Free-form rationale for reviewers.
    """

    question: str
    answerable: bool = True
    expected_chapters: list[str] = Field(default_factory=list)
    expected_answer_contains: list[str] = Field(default_factory=list)
    notes: str | None = None


def load_golden(path: Path | str) -> list[GoldenItem]:
    """Load a golden question set from YAML, JSON or JSONL.

    Args:
        path: File containing a list of questions, or a mapping with
            ``items``/``questions`` keys.

    Returns:
        Validated golden items in file order.

    Raises:
        ValueError: If the file format is unsupported or the set is empty.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".jsonl":
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported golden set format: {suffix}")

    if isinstance(data, dict):
        data = data.get("items", data.get("questions", []))
    if not isinstance(data, list):
        raise ValueError(
            f"Golden set must be a list of items, got {type(data).__name__}"
        )

    items = [GoldenItem.model_validate(entry) for entry in data]
    if not items:
        raise ValueError(f"Golden set is empty: {path}")
    return items


def save_golden(path: Path | str, items: list[GoldenItem]) -> None:
    """Write a golden question set as YAML or JSON (by file suffix)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "items": [item.model_dump(exclude_none=True) for item in items]
    }
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    else:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
