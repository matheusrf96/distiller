"""Unit tests for PDF backend selection, fallback and extraction."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from distiller.exceptions import IngestError
from distiller.ingest import pdf
from distiller.ingest.pdf import parse_pdf

if TYPE_CHECKING:
    from pathlib import Path

    from distiller.config import Settings


def test_resolve_backends_prefers_installed_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto mode lists installed backends; explicit mode trusts the request."""
    monkeypatch.setattr(pdf, "is_available", lambda name: False)

    assert pdf._resolve_backends("auto") == ["pymupdf"]
    assert pdf._resolve_backends("docling") == ["docling"]


def test_pdf_backend_failures_fall_back_to_the_next(
    settings: Settings,
    sample_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing or empty backend is recorded and the next one is tried."""

    def failing(path: Path) -> tuple[list, dict]:
        raise RuntimeError("converter exploded")

    def empty(path: Path) -> tuple[list, dict]:
        return [], {}

    monkeypatch.setattr(
        pdf,
        "_resolve_backends",
        lambda requested: ["docling", "pymupdf4llm", "pymupdf"],
    )
    monkeypatch.setitem(pdf._EXTRACTORS, "docling", failing)
    monkeypatch.setitem(pdf._EXTRACTORS, "pymupdf4llm", empty)

    with caplog.at_level(logging.WARNING):
        book = parse_pdf(sample_pdf, "fallback-book", settings.ingest)

    assert book.chapters
    assert book.metadata["backend"] == "pymupdf"
    assert any("converter exploded" in record.message for record in caplog.records)


def test_pdf_reports_all_backend_failures(
    settings: Settings,
    sample_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every backend fails the error lists what was tried."""

    def failing(path: Path) -> tuple[list, dict]:
        raise RuntimeError("boom")

    monkeypatch.setattr(pdf, "_resolve_backends", lambda requested: ["docling"])
    monkeypatch.setitem(pdf._EXTRACTORS, "docling", failing)

    with pytest.raises(IngestError, match="Could not parse PDF"):
        parse_pdf(sample_pdf, "broken-book", settings.ingest)


def test_optional_pdf_extractors_use_required_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Docling and PyMuPDF4LLM extraction go through require() and parse blocks."""
    markdown = "# One\n\nBody."

    class FakeConverter:
        def convert(self, path: str) -> object:
            document = SimpleNamespace(export_to_markdown=lambda: markdown)
            return SimpleNamespace(document=document)

    def fake_require(name: str, **kwargs: object) -> object:
        if name == "docling.document_converter":
            return SimpleNamespace(DocumentConverter=FakeConverter)
        return SimpleNamespace(to_markdown=lambda path: markdown)

    monkeypatch.setattr(pdf, "require", fake_require)

    docling_blocks, docling_meta = pdf._extract_docling(tmp_path / "x.pdf")
    pymupdf4llm_blocks, pymupdf4llm_meta = pdf._extract_pymupdf4llm(tmp_path / "x.pdf")

    assert docling_blocks[0].text == "One"
    assert docling_meta["markdown_chars"] == len(markdown)
    assert pymupdf4llm_blocks[0].text == "One"
    assert pymupdf4llm_meta["markdown_chars"] == len(markdown)


def test_extract_pymupdf_skips_empty_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty text blocks are skipped and pages are recorded."""

    class FakePage:
        def get_text(self, kind: str) -> list[list[object]]:
            return [
                [0, 0, 0, 0, "   ", 0, 0],
                [0, 0, 0, 0, "Real text.", 0, 0],
            ]

    class FakeDocument:
        page_count = 1

        def load_page(self, index: int) -> FakePage:
            return FakePage()

        def __enter__(self) -> FakeDocument:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    monkeypatch.setattr(pdf.pymupdf, "open", lambda path: FakeDocument())

    blocks, metadata = pdf._extract_pymupdf(tmp_path / "x.pdf")

    assert [block.text for block in blocks] == ["Real text."]
    assert metadata == {"pages": 1}


def test_clean_block_text_rejoins_wrapped_lines() -> None:
    """Hard line wraps are rejoined until sentence-ending punctuation."""
    assert pdf._clean_block_text("Hello\nworld.\nNext") == "Hello world.\nNext"
