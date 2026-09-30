"""Shared pytest fixtures.

Everything here generates books locally: no network, no binary fixtures, no
model downloads. The offline pipeline uses the hashing embedder and the fake LLM.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from distiller.config import Settings, load_settings
from distiller.utils import wrap_text

FIXTURE_CHAPTERS: list[tuple[str, str]] = [
    (
        "Chapter One",
        "The lighthouse keeper counted three hundred steps "
        "to the top of the tower every morning.",
    ),
    (
        "Chapter Two",
        "During the great storm of 1887 the lantern cracked, "
        "and the keeper rowed to the village for a new one.",
    ),
    (
        "Chapter Three",
        "The keeper returned in the spring and painted the tower white, "
        "a colour the fishermen trusted.",
    ),
]

EpubFactory = Callable[..., Path]
PdfFactory = Callable[..., Path]


@pytest.fixture()
def offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the pipeline at a temp artifacts dir and offline backends.

    Returns:
        The temporary artifacts directory.
    """
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("DISTILLER_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("DISTILLER_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("DISTILLER_LLM__MODEL", "fake")
    return artifacts


@pytest.fixture()
def settings(offline_env: Path) -> Settings:
    """Settings wired to the offline environment."""
    return load_settings()


@pytest.fixture()
def epub_factory() -> EpubFactory:
    """Factory fixture building small EPUB files with EbookLib."""

    def _make(path: Path, chapters: list[tuple[str, str]] | None = None) -> Path:
        from ebooklib import epub

        book = epub.EpubBook()
        book.set_identifier("fixture-lantern-keeper")
        book.set_title("The Lantern Keeper")
        book.set_language("en")
        book.add_author("Ada Fixture")

        items = []
        for index, (title, text) in enumerate(chapters or FIXTURE_CHAPTERS, start=1):
            item = epub.EpubHtml(
                title=title, file_name=f"chap_{index}.xhtml", lang="en"
            )
            item.content = f"<h1>{title}</h1><h2>Section {index}</h2><p>{text}</p>"
            book.add_item(item)
            items.append(item)

        book.toc = tuple(
            epub.Link(item.file_name, item.title, f"chap-{index}")
            for index, item in enumerate(items, start=1)
        )
        book.spine = ["nav", *items]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        epub.write_epub(str(path), book)
        return path

    return _make


@pytest.fixture()
def pdf_factory() -> PdfFactory:
    """Factory fixture building small text-based PDF files with PyMuPDF."""

    def _make(path: Path, chapters: list[tuple[str, str]] | None = None) -> Path:
        import pymupdf

        document = pymupdf.open()
        for title, text in chapters or FIXTURE_CHAPTERS:
            page = document.new_page()
            page.insert_text((72, 90), title, fontsize=20)
            y_position = 130
            for line in wrap_text(text, 78):
                page.insert_text((72, y_position), line, fontsize=11)
                y_position += 18
        document.set_metadata({"title": "The Lantern Keeper", "author": "Ada Fixture"})
        document.save(str(path))
        document.close()
        return path

    return _make


@pytest.fixture()
def sample_epub(epub_factory: EpubFactory, tmp_path: Path) -> Path:
    """A three-chapter EPUB fixture."""
    return epub_factory(tmp_path / "lantern.epub")


@pytest.fixture()
def sample_pdf(pdf_factory: PdfFactory, tmp_path: Path) -> Path:
    """A three-page PDF fixture."""
    return pdf_factory(tmp_path / "lantern.pdf")
