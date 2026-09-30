"""Unit tests for EPUB ingestion edge cases."""

from __future__ import annotations

from typing import TYPE_CHECKING

from distiller.ingest import ingest_book
from distiller.ingest.epub import _MIN_CHAPTER_CHARS

if TYPE_CHECKING:
    from pathlib import Path

    from distiller.config import Settings


def build_epub_with_junk(path: Path) -> Path:
    """EPUB with a navigation-ish stub, a tiny titled chapter and a big chapter.

    The stub has no heading and fewer than ``_MIN_CHAPTER_CHARS`` characters,
    so ingestion must skip it; the tiny chapter carries a heading and must
    survive.
    """
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("edge-cases")
    book.set_title("Edge Cases")
    book.set_language("en")

    stub = epub.EpubHtml(title="Cover", file_name="cover.xhtml", lang="en")
    stub.content = "<p>navigation stub</p>"

    tiny = epub.EpubHtml(title="The Tiny Chapter", file_name="tiny.xhtml", lang="en")
    tiny.content = "<h1>The Tiny Chapter</h1><p>One short line.</p>"

    big = epub.EpubHtml(title="The Long Chapter", file_name="big.xhtml", lang="en")
    big.content = (
        "<h1>The Long Chapter</h1><p>"
        + ("Endless prose about lanterns. " * 20)
        + "</p>"
    )

    for item in (stub, tiny, big):
        book.add_item(item)
    book.toc = (
        epub.Link(tiny.file_name, tiny.title, "tiny"),
        epub.Link(big.file_name, big.title, "big"),
    )
    book.spine = ["nav", stub, tiny, big]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)
    return path


def test_short_titled_chapter_survives_but_nav_stub_is_skipped(
    settings: Settings, tmp_path: Path
) -> None:
    """Tiny chapters are kept when they have a heading; heading-less stubs are not."""
    epub_path = build_epub_with_junk(tmp_path / "edge.epub")
    book = ingest_book(epub_path, settings=settings)

    titles = [chapter.title for chapter in book.chapters]
    assert titles == ["The Tiny Chapter", "The Long Chapter"]
    assert len("One short line.") < _MIN_CHAPTER_CHARS  # the point of the fixture
    assert "navigation stub" not in book.full_text()
