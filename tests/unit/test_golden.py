"""Unit tests for golden set loading and saving."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from distiller.evaluation import GoldenItem, load_golden, save_golden

if TYPE_CHECKING:
    from pathlib import Path


def test_load_golden_accepts_json_and_jsonl(tmp_path: Path) -> None:
    """JSON lists and JSON Lines files load in file order."""
    json_path = tmp_path / "golden.json"
    json_path.write_text(json.dumps([{"question": "q1"}]), encoding="utf-8")
    jsonl_path = tmp_path / "golden.jsonl"
    jsonl_path.write_text(
        '{"question": "q1"}\n\n{"question": "q2"}\n', encoding="utf-8"
    )

    assert load_golden(json_path) == [GoldenItem(question="q1")]
    assert [item.question for item in load_golden(jsonl_path)] == ["q1", "q2"]


def test_load_golden_accepts_items_and_questions_wrappers(tmp_path: Path) -> None:
    """Mapping files may wrap the list under ``items`` or ``questions``."""
    items_path = tmp_path / "items.yaml"
    items_path.write_text("items:\n  - question: q1\n", encoding="utf-8")
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text("questions:\n  - question: q2\n", encoding="utf-8")

    assert load_golden(items_path) == [GoldenItem(question="q1")]
    assert load_golden(questions_path) == [GoldenItem(question="q2")]


def test_load_golden_rejects_unsupported_and_non_list_formats(tmp_path: Path) -> None:
    """Unsupported suffixes and non-list payloads raise ValueError."""
    txt_path = tmp_path / "golden.txt"
    txt_path.write_text("q1", encoding="utf-8")
    scalar_path = tmp_path / "golden.json"
    scalar_path.write_text("42", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported golden set format"):
        load_golden(txt_path)
    with pytest.raises(ValueError, match="must be a list"):
        load_golden(scalar_path)


def test_load_golden_rejects_empty_sets(tmp_path: Path) -> None:
    """Empty lists and mappings without a known wrapper raise ValueError."""
    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("items: []\n", encoding="utf-8")
    unknown_path = tmp_path / "unknown.yaml"
    unknown_path.write_text("other:\n  - question: q1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Golden set is empty"):
        load_golden(empty_path)
    with pytest.raises(ValueError, match="Golden set is empty"):
        load_golden(unknown_path)


def test_save_golden_writes_json_by_suffix(tmp_path: Path) -> None:
    """A non-YAML suffix writes JSON with the same items payload."""
    path = tmp_path / "golden.json"
    item = GoldenItem(question="q1", expected_chapters=["One"])
    save_golden(path, [item])

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "items": [item.model_dump(exclude_none=True)]
    }
