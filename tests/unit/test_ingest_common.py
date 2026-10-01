"""Unit tests for the shared block -> chapter helpers."""

from __future__ import annotations

from distiller.ingest.common import (
    blocks_to_chapters,
    render_book_markdown,
    strip_duplicate_leading_heading,
)
from distiller.models import Block, BookDocument, Chapter


def test_blocks_to_chapters_handles_empty_and_headingless_documents() -> None:
    """No blocks yield no chapters; a headingless document is one chapter."""
    assert blocks_to_chapters([], "Fallback") == []

    blocks = [Block(type="paragraph", text="Body.")]
    chapters = blocks_to_chapters(blocks, "Fallback")

    assert [chapter.title for chapter in chapters] == ["Fallback"]
    assert chapters[0].blocks == blocks


def test_blocks_to_chapters_uses_h2_when_no_h1_exists() -> None:
    """A document with only h2+ breaks chapters at level 2."""
    blocks = [
        Block(type="heading", text="One", level=2),
        Block(type="paragraph", text="Body one."),
        Block(type="heading", text="Two", level=2),
        Block(type="paragraph", text="Body two."),
    ]

    chapters = blocks_to_chapters(blocks, "Fallback")

    assert [chapter.title for chapter in chapters] == ["One", "Two"]


def test_blocks_to_chapters_prepends_content_before_the_first_heading() -> None:
    """Blocks before the first heading start a fallback-titled chapter."""
    blocks = [
        Block(type="paragraph", text="Preface."),
        Block(type="heading", text="One", level=1),
        Block(type="paragraph", text="Body."),
    ]

    chapters = blocks_to_chapters(blocks, "The Book")

    assert [chapter.title for chapter in chapters] == ["The Book", "One"]


def test_strip_duplicate_leading_heading_drops_repeats_only() -> None:
    """A leading heading matching the title is dropped; others are kept."""
    heading = Block(type="heading", text="Chapter One", level=1)
    body = Block(type="paragraph", text="Body.")

    assert strip_duplicate_leading_heading([], "Chapter One") == []
    assert strip_duplicate_leading_heading([heading, body], "chapter one") == [body]
    assert strip_duplicate_leading_heading([heading, body], "Other") == [heading, body]


def test_render_book_markdown_skips_authors_and_empty_chapters() -> None:
    """Books without authors and chapters without blocks render cleanly."""
    book = BookDocument(
        book_id="b",
        title="The Book",
        source_path="b.txt",
        source_format="txt",
        chapters=[
            Chapter(title="Empty", blocks=[]),
            Chapter(title="One", blocks=[Block(type="paragraph", text="Body.")]),
        ],
    )

    rendered = render_book_markdown(book)

    assert rendered.startswith("# The Book")
    assert "by " not in rendered
    assert "# Empty" in rendered
    assert "Body." in rendered
