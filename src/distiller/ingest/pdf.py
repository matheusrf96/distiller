"""PDF ingestion with pluggable backends.

Backend order for ``auto``: Docling (best structure, heaviest) -> PyMuPDF4LLM
(fast Markdown) -> plain PyMuPDF page text (always available). Docling and
PyMuPDF4LLM come from the ``pdf-ai`` extra and are loaded dynamically;
availability is checked without importing so backend selection is not
exception-driven.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pymupdf

from ..exceptions import IngestError, MissingDependencyError
from ..models import Block, BookDocument
from ..optional_deps import is_available, require
from .common import blocks_to_chapters, drop_empty_blocks
from .markdown import markdown_to_blocks

if TYPE_CHECKING:
    from ..config import IngestSettings

logger = logging.getLogger(__name__)

PdfBackend = Literal["auto", "docling", "pymupdf4llm", "pymupdf"]

_BACKEND_PRIORITY: tuple[str, ...] = ("docling", "pymupdf4llm", "pymupdf")
_BACKEND_MODULES: dict[str, str] = {
    "docling": "docling",
    "pymupdf4llm": "pymupdf4llm",
    "pymupdf": "pymupdf",
}

_Extractor = Callable[[Path], tuple[list[Block], dict[str, Any]]]


def parse_pdf(path: Path, book_id: str, settings: IngestSettings) -> BookDocument:
    """Parse a PDF into a normalized book document.

    Tries the configured backend (or the best available one for ``auto``) and
    falls back to the next backend when one fails.

    Args:
        path: Path to the ``.pdf`` file.
        book_id: Provisional book slug (refined by the caller from the title).
        settings: Ingest settings selecting the backend.

    Returns:
        Parsed book with chapters derived from the extractor's headings.

    Raises:
        MissingDependencyError: If an explicitly requested backend's extra is absent.
        IngestError: If every backend fails or produces no content.
    """
    errors: list[str] = []
    for backend in _resolve_backends(settings.pdf_backend):
        try:
            blocks, metadata = _EXTRACTORS[backend](path)
        except MissingDependencyError:
            raise  # already actionable: names the module and the extra to install
        except Exception as exc:
            message = f"{backend}: {type(exc).__name__}: {exc}"
            logger.warning("PDF backend failed: %s", message)
            errors.append(message)
            continue

        blocks = drop_empty_blocks(blocks)
        if not blocks:
            errors.append(f"{backend}: produced no content")
            continue

        title, authors = _pdf_metadata(path)
        return BookDocument(
            book_id=book_id,
            title=title,
            authors=authors,
            source_path=str(path),
            source_format="pdf",
            chapters=blocks_to_chapters(blocks, fallback_title=title),
            metadata={"backend": backend, **metadata},
        )

    raise IngestError(
        "Could not parse PDF with any backend. Tried: " + " | ".join(errors)
    )


def _resolve_backends(requested: PdfBackend) -> list[str]:
    if requested != "auto":
        return [
            requested
        ]  # an explicitly requested backend fails loudly if unavailable
    available = [
        name for name in _BACKEND_PRIORITY if is_available(_BACKEND_MODULES[name])
    ]
    return available or ["pymupdf"]


def _extract_docling(path: Path) -> tuple[list[Block], dict[str, Any]]:
    converter_module = require(
        "docling.document_converter", extra="pdf-ai", purpose="Docling PDF parsing"
    )
    result = converter_module.DocumentConverter().convert(str(path))
    markdown = result.document.export_to_markdown()
    return markdown_to_blocks(markdown), {"markdown_chars": len(markdown)}


def _extract_pymupdf4llm(path: Path) -> tuple[list[Block], dict[str, Any]]:
    module = require("pymupdf4llm", extra="pdf-ai", purpose="PyMuPDF4LLM PDF parsing")
    markdown = module.to_markdown(str(path))
    return markdown_to_blocks(markdown), {"markdown_chars": len(markdown)}


def _extract_pymupdf(path: Path) -> tuple[list[Block], dict[str, Any]]:
    blocks: list[Block] = []
    with pymupdf.open(str(path)) as document:
        page_count = int(document.page_count)
        for page_index in range(1, page_count + 1):
            page = document.load_page(page_index - 1)
            for raw_block in page.get_text("blocks"):
                text = str(raw_block[4]).strip()
                if text:
                    blocks.append(
                        Block(
                            type="paragraph",
                            text=_clean_block_text(text),
                            page=page_index,
                        )
                    )
    return blocks, {"pages": page_count}


def _pdf_metadata(path: Path) -> tuple[str, list[str]]:
    with pymupdf.open(str(path)) as document:
        metadata = dict(document.metadata or {})
    title = str(metadata.get("title") or "").strip() or path.stem
    author = str(metadata.get("author") or "").strip()
    authors = [
        name.strip() for name in author.replace(";", ",").split(",") if name.strip()
    ]
    return title, authors


def _clean_block_text(text: str) -> str:
    """Rejoin obvious hard line wraps that PyMuPDF introduces."""
    out: list[str] = []
    for line in (raw.rstrip() for raw in text.splitlines()):
        if (
            out
            and out[-1]
            and not out[-1].endswith((".", "!", "?", ":", ";", '"', "”"))
        ):
            out[-1] = f"{out[-1]} {line.strip()}"
        else:
            out.append(line.strip())
    return "\n".join(line for line in out if line)


_EXTRACTORS: dict[str, _Extractor] = {
    "docling": _extract_docling,
    "pymupdf4llm": _extract_pymupdf4llm,
    "pymupdf": _extract_pymupdf,
}
