"""Unit tests for domain model helpers."""

from __future__ import annotations

from distiller.models import format_location


def test_format_location_includes_heading_and_page_range() -> None:
    """A distinct heading and a page range are appended to the label."""
    label = format_location("Chapter One", ["Part I", "The Storm"], 12, 13)

    assert label == "Chapter One — The Storm (pp. 12-13)"


def test_format_location_skips_duplicate_headings_and_pages() -> None:
    """A heading equal to the chapter and unknown pages add nothing."""
    assert format_location("Chapter One", ["Chapter One"], None, None) == "Chapter One"
