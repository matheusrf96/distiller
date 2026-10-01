"""Unit tests for EPUB ingestion edge cases."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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


def test_ingest_rejects_an_epub_without_readable_chapters(
    settings: Settings, tmp_path: Path
) -> None:
    """An EPUB with only a stub and a stylesheet yields no chapters."""
    from ebooklib import epub

    from distiller.exceptions import IngestError

    book = epub.EpubBook()
    book.set_identifier("empty")
    book.set_title("Empty")
    book.set_language("en")
    stub = epub.EpubHtml(title="Cover", file_name="cover.xhtml", lang="en")
    stub.content = "<p>tiny stub</p>"
    style = epub.EpubItem(
        uid="style", file_name="style.css", media_type="text/css", content=b"body{}"
    )
    for item in (stub, style):
        book.add_item(item)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", stub, style]
    epub.write_epub(str(tmp_path / "empty.epub"), book)

    with pytest.raises(IngestError, match="No readable chapters"):
        ingest_book(tmp_path / "empty.epub", settings=settings)


def test_document_to_chapter_skips_heading_only_documents() -> None:
    """A document whose only block repeats the title yields no chapter."""
    from distiller.ingest.epub import _document_to_chapter

    class StubItem:
        def get_name(self) -> str:
            return "tiny.xhtml"

        def get_content(self) -> bytes:
            return b"<h1>Tiny</h1>"

    assert _document_to_chapter(StubItem(), {}) is None


def test_document_body_falls_back_to_the_fragment() -> None:
    """A fragment without a body tag parses as-is, with junk tags removed."""
    from distiller.ingest.epub import _document_body

    class StubItem:
        def get_content(self) -> bytes:
            return b"<nav>junk</nav><p>fragment text</p>"

    body = _document_body(StubItem())

    assert "fragment text" in body.get_text()
    assert "junk" not in body.get_text()


def test_toc_titles_skip_untitled_links() -> None:
    """TOC entries without a title are not mapped."""
    from distiller.ingest.epub import _toc_titles

    class Untitled:
        href = "empty.xhtml"
        title = None
        children: tuple[object, ...] = ()

    assert _toc_titles([Untitled()]) == {}


def test_flatten_toc_handles_none_links_sections_and_children() -> None:
    """TOC flattening recurses through sections and skips empty entries."""
    from distiller.ingest.epub import _flatten_toc

    class Link:
        def __init__(self, href: str, title: str) -> None:
            self.href = href
            self.title = title

    class Section:
        href = None
        title = "Part I"
        children: tuple[object, ...] = (Link("child.xhtml", "Child"),)

    assert _flatten_toc(None) == []
    assert _flatten_toc([Section()]) == [("child.xhtml", "Child")]


def test_spine_entry_normalizes_ids_linear_flags_and_dicts() -> None:
    """Spine entries may be ids, pairs or dicts."""
    from distiller.ingest.epub import _spine_entry

    assert _spine_entry("chap1") == ("chap1", True)
    assert _spine_entry(("chap1", "no")) == ("chap1", False)
    assert _spine_entry(("chap1", {"linear": "no"})) == ("chap1", False)
    assert _spine_entry(("chap1",)) == ("chap1", True)


def test_metadata_first_value_returns_none_without_values() -> None:
    """Books without a metadata entry fall back to None."""
    from distiller.ingest.epub import _metadata_first_value

    class EmptyBook:
        def get_metadata(self, namespace: str, name: str) -> list[object]:
            return []

    assert _metadata_first_value(EmptyBook(), "title") is None


def test_first_heading_skips_empty_tags() -> None:
    """Empty heading tags are skipped in favour of the next one."""
    from bs4 import BeautifulSoup

    from distiller.ingest.epub import _first_heading

    soup = BeautifulSoup("<h1>   </h1><h2>Real</h2>", "html.parser")

    assert _first_heading(soup) == "Real"
