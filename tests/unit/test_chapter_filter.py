"""Unit tests for the chapter filter matching rules."""

from __future__ import annotations

import pytest

from distiller.rag.retriever import _chapter_matches


@pytest.mark.parametrize(
    ("chunk_chapter", "wanted", "expected"),
    [
        ("Chapter One", "Chapter One", True),
        ("Chapter One", "chapter one", True),
        ("Chapter One", "one", True),
        ("Chapter One", None, True),
        ("Chapter One", "   ", True),
        ("Chapter 1: The Storm", "Chapter 1", True),
        ("Chapter 1 - The Storm", "Chapter 1", True),
        ("Chapter 10", "Chapter 1", False),
        ("Chapter 11 - The Return", "Chapter 1", False),
        ("Part One", "part", True),
        ("The Storm", "storm", True),
        ("Chapter One", "Two", False),
        ("Chapter One", "1", False),
    ],
)
def test_chapter_matching_uses_word_boundaries(
    chunk_chapter: str,
    wanted: str | None,
    expected: bool,
) -> None:
    """Filters match whole words only, so ``Chapter 1`` never matches ``Chapter 10``."""
    assert _chapter_matches(chunk_chapter, wanted) is expected


def test_chapter_matching_treats_regex_metacharacters_as_text() -> None:
    """User input is escaped, not interpreted as a regular expression."""
    assert _chapter_matches("Chapter 1 (Revised)", "1 (Revised)") is True
    assert _chapter_matches("Chapter A.B", "A.B") is True
