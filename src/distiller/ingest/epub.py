"""EPUB ingestion: EbookLib (spine + TOC) -> BeautifulSoup -> markdownify -> blocks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from markdownify import markdownify

from ..exceptions import IngestError
from ..models import Block, BookDocument, Chapter
from .common import drop_empty_blocks, strip_duplicate_leading_heading
from .markdown import markdown_to_blocks

if TYPE_CHECKING:
    from collections.abc import Iterator

    from bs4.element import Tag

_MIN_CHAPTER_CHARS = 80
_SKIPPED_SPINE_IDS = frozenset({"nav", "toc"})


def parse_epub(path: Path, book_id: str) -> BookDocument:
    """Parse an EPUB into a normalized book document.

    Walks the spine in reading order, resolves chapter titles from the TOC and
    converts each document to Markdown before block extraction.

    Args:
        path: Path to the ``.epub`` file.
        book_id: Provisional book slug (refined by the caller from the title).

    Returns:
        Parsed book with TOC-aware chapters.

    Raises:
        IngestError: If the EPUB contains no readable chapters.
    """
    book = epub.read_epub(str(path))
    title, authors, language = _metadata(book, fallback_title=path.stem)
    toc_titles = _toc_titles(book.toc)

    chapters = [
        chapter
        for chapter in (
            _document_to_chapter(item, toc_titles) for item in _spine_documents(book)
        )
        if chapter is not None
    ]
    if not chapters:
        raise IngestError(f"No readable chapters found in {path}")

    return BookDocument(
        book_id=book_id,
        title=title,
        authors=authors,
        language=language,
        source_path=str(path),
        source_format="epub",
        chapters=chapters,
        metadata={"toc_entries": len(toc_titles), "backend": "ebooklib"},
    )


def _metadata(
    book: epub.EpubBook, *, fallback_title: str
) -> tuple[str, list[str], str | None]:
    title = _metadata_first_value(book, "title") or fallback_title
    authors = _metadata_values(book, "creator")
    language = _metadata_first_value(book, "language")
    return title, authors, language


def _document_to_chapter(item: Any, toc_titles: dict[str, str]) -> Chapter | None:
    href = item.get_name()
    body = _document_body(item)
    title = toc_titles.get(href) or _first_heading(body) or _title_from_href(href)

    # Skip navigation pages and covers, but never drop a short document that
    # carries a heading: it is a legitimate (tiny) chapter.
    is_short = len(body.get_text(" ", strip=True)) < _MIN_CHAPTER_CHARS
    if is_short and _first_heading(body) is None:
        return None
    markdown = markdownify(str(body), heading_style="ATX")
    blocks: list[Block] = drop_empty_blocks(
        strip_duplicate_leading_heading(markdown_to_blocks(markdown), title)
    )
    if not blocks:
        return None
    return Chapter(title=title, level=1, blocks=blocks)


def _document_body(item: Any) -> Tag:
    soup = BeautifulSoup(
        item.get_content().decode("utf-8", errors="replace"), "html.parser"
    )
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    return soup.body or soup


def _spine_documents(book: epub.EpubBook) -> Iterator[Any]:
    """Yield readable documents in spine order, skipping nav and non-linear items."""
    for entry in book.spine:
        idref, linear = _spine_entry(entry)
        if not linear or idref in _SKIPPED_SPINE_IDS:
            continue
        item = book.get_item_with_id(idref)
        if item is not None and item.get_type() == ebooklib.ITEM_DOCUMENT:
            yield item


def _spine_entry(entry: Any) -> tuple[str, bool]:
    if isinstance(entry, (tuple, list)):
        idref = str(entry[0])
        linear = entry[1] if len(entry) > 1 else "yes"
    else:
        idref, linear = str(entry), "yes"
    if isinstance(linear, dict):
        linear = linear.get("linear", "yes")
    return idref, str(linear).lower() not in {"no", "false"}


def _toc_titles(toc: Any) -> dict[str, str]:
    """Flatten ebooklib's nested TOC into an ``href -> title`` mapping."""
    titles: dict[str, str] = {}
    for href, title in _flatten_toc(toc):
        if href and title:
            titles.setdefault(href, title)
    return titles


def _flatten_toc(node: Any, level: int = 0) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if node is None:
        return out
    if isinstance(node, (list, tuple)):
        for child in node:
            out.extend(_flatten_toc(child, level))
        return out

    href = getattr(node, "href", None)
    title = getattr(node, "title", None)
    if href:
        out.append((str(href).split("#")[0], str(title or "").strip()))
    for child in getattr(node, "children", None) or []:
        out.extend(_flatten_toc(child, level + 1))
    return out


def _metadata_first_value(book: epub.EpubBook, name: str) -> str | None:
    values = book.get_metadata("DC", name)
    if not values:
        return None
    value = str(values[0][0]).strip()
    return value or None


def _metadata_values(book: epub.EpubBook, name: str) -> list[str]:
    return [
        str(value[0]).strip()
        for value in book.get_metadata("DC", name)
        if str(value[0]).strip()
    ]


def _first_heading(soup: Tag) -> str | None:
    for tag_name in ("h1", "h2", "h3"):
        tag = soup.find(tag_name)
        if tag:
            text = tag.get_text(" ", strip=True)
            if text:
                return text
    return None


def _title_from_href(href: str) -> str:
    stem = Path(href).stem
    cleaned = stem.replace("_", " ").replace("-", " ").strip()
    return cleaned.title() or href
