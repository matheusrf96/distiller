"""Plain text / Markdown ingestion (useful for testing and copied excerpts)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..exceptions import IngestError
from ..models import BookDocument
from .common import blocks_to_chapters, drop_empty_blocks
from .markdown import markdown_to_blocks

if TYPE_CHECKING:
    from pathlib import Path


def parse_text(path: Path, book_id: str, *, source_format: str = "md") -> BookDocument:
    """Parse a Markdown or plain-text file into a normalized book document.

    Args:
        path: Path to the file.
        book_id: Provisional book slug (refined by the caller from the title).
        source_format: ``"md"`` or ``"txt"``, recorded on the document.

    Returns:
        Parsed book; headings become chapters when present.

    Raises:
        IngestError: If the file contains no content.
    """
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    blocks = drop_empty_blocks(markdown_to_blocks(raw_text))
    if not blocks:
        raise IngestError(f"No content found in {path}")
    chapters = blocks_to_chapters(blocks, fallback_title=path.stem)
    return BookDocument(
        book_id=book_id,
        title=path.stem,
        source_path=str(path),
        source_format="md" if source_format != "txt" else "txt",
        chapters=chapters,
        metadata={"backend": "text"},
    )
