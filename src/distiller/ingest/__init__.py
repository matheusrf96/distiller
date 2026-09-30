"""Book ingestion: dispatch by file type into a normalized ``BookDocument``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..exceptions import BookNotFoundError, IngestError
from ..utils import slugify
from .epub import parse_epub
from .pdf import parse_pdf
from .text import parse_text

if TYPE_CHECKING:
    from ..config import Settings
    from ..models import BookDocument

SUPPORTED_SUFFIXES = frozenset({".pdf", ".epub", ".md", ".markdown", ".txt"})


def ingest_book(
    path: Path | str,
    *,
    settings: Settings,
    title: str | None = None,
) -> BookDocument:
    """Parse a book file into a normalized document.

    The ``book_id`` is derived from the title (override or parsed), so the same
    book always lands in the same artifact directory.

    Args:
        path: Path to a ``.pdf``, ``.epub``, ``.md`` or ``.txt`` file.
        settings: Pipeline settings (PDF backend selection lives here).
        title: Optional title override; also determines the book id.

    Returns:
        Parsed book with chapters, blocks and provenance metadata.

    Raises:
        BookNotFoundError: If ``path`` does not exist.
        IngestError: If the file type is unsupported or parsing fails.
    """
    path = Path(path)
    if not path.exists():
        raise BookNotFoundError(f"Book not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestError(
            f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    provisional_id = slugify(path.stem)
    if suffix == ".epub":
        book = parse_epub(path, provisional_id)
    elif suffix == ".pdf":
        book = parse_pdf(path, provisional_id, settings.ingest)
    else:
        book = parse_text(path, provisional_id, source_format=suffix.lstrip("."))

    book.book_id = slugify(title or book.title or path.stem)
    if title:
        book.title = title
    return book
