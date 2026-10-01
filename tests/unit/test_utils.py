"""Unit tests for shared helpers (slugs, hashing, JSON I/O, wrapping)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from distiller.models import DomainModel
from distiller.utils import read_jsonl, slugify, stable_hash_hex, wrap_text, write_jsonl

if TYPE_CHECKING:
    from pathlib import Path


class SampleRow(DomainModel):
    """Minimal model used to exercise model-aware JSONL writing."""

    value: int


def test_wrap_text_empty_input_returns_one_empty_line() -> None:
    """Empty text yields a single empty line instead of an empty list."""
    assert wrap_text("") == [""]
    assert wrap_text("one two", width=3) == ["one", "two"]


def test_write_jsonl_serializes_models_and_plain_rows(tmp_path: Path) -> None:
    """Models dump themselves; plain mappings are written as-is."""
    path = tmp_path / "rows.jsonl"

    count = write_jsonl(path, [SampleRow(value=1), {"value": 2}])

    assert count == 2
    assert [row["value"] for row in read_jsonl(path)] == [1, 2]


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    """Blank lines never yield rows."""
    path = tmp_path / "rows.jsonl"
    path.write_text('{"value": 1}\n\n   \n{"value": 2}\n', encoding="utf-8")

    assert [row["value"] for row in read_jsonl(path)] == [1, 2]


def test_slug_and_hash_are_stable() -> None:
    """Slugs never come back empty and hashes keep their length."""
    assert slugify("!!!") == "book"
    assert slugify("The Storm!") == "the-storm"
    assert len(stable_hash_hex("x")) == 40
