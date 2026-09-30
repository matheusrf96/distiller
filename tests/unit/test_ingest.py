"""Unit tests for EPUB, PDF and plain-text ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from distiller.ingest import ingest_book

if TYPE_CHECKING:
    from pathlib import Path

    from distiller.config import Settings


def test_epub_ingest_structure(settings: Settings, sample_epub: Path) -> None:
    """EPUB parsing resolves TOC chapters, metadata and heading blocks."""
    book = ingest_book(sample_epub, settings=settings)

    assert book.book_id == "the-lantern-keeper"
    assert book.title == "The Lantern Keeper"
    assert book.authors == ["Ada Fixture"]
    assert book.language == "en"
    assert book.source_format == "epub"
    assert [chapter.title for chapter in book.chapters] == [
        "Chapter One",
        "Chapter Two",
        "Chapter Three",
    ]
    assert "three hundred steps" in book.full_text()
    assert any(
        block.type == "heading" for chapter in book.chapters for block in chapter.blocks
    )


def test_pdf_ingest_structure(settings: Settings, sample_pdf: Path) -> None:
    """Plain-PyMuPDF parsing keeps page numbers and metadata."""
    settings.ingest.pdf_backend = "pymupdf"
    book = ingest_book(sample_pdf, settings=settings)

    assert book.source_format == "pdf"
    assert book.metadata["backend"] == "pymupdf"
    assert book.metadata["pages"] == 3
    assert book.title == "The Lantern Keeper"
    assert book.book_id == "the-lantern-keeper"
    assert "three hundred steps" in book.full_text()
    assert "lantern cracked" in book.full_text()
    assert any(block.page for chapter in book.chapters for block in chapter.blocks)


def test_markdown_ingest_with_title_override(
    settings: Settings, tmp_path: Path
) -> None:
    """A title override wins over the file stem for id and title."""
    source = tmp_path / "notes.md"
    source.write_text(
        "# Alpha\n\nFirst body paragraph about the sea.\n\n"
        "# Beta\n\nSecond body paragraph.\n",
        encoding="utf-8",
    )
    book = ingest_book(source, settings=settings, title="My Notes")

    assert book.book_id == "my-notes"
    assert book.title == "My Notes"
    assert [chapter.title for chapter in book.chapters] == ["Alpha", "Beta"]
    assert book.block_count > 0
