"""Shared helpers for turning a flat block list into chapters."""

from __future__ import annotations

from ..models import Block, BookDocument, Chapter
from .markdown import blocks_to_markdown


def blocks_to_chapters(blocks: list[Block], fallback_title: str) -> list[Chapter]:
    """Group blocks into chapters using the shallowest heading level present.

    Heading level 1 is preferred as the chapter break; if the document only uses
    h2+, that becomes the break. Documents without headings become a single
    chapter titled ``fallback_title``.

    Args:
        blocks: Flat, ordered content blocks.
        fallback_title: Title for chapters that start without a heading
            (usually the book title).

    Returns:
        Chapters in reading order, each with at least one non-empty block.
    """
    if not blocks:
        return []

    levels = {
        block.level for block in blocks if block.type == "heading" and block.level
    }
    if 1 in levels:
        break_level = 1
    elif 2 in levels:
        break_level = 2
    else:
        break_level = 0

    if break_level == 0:
        return [Chapter(title=fallback_title, level=1, blocks=list(blocks))]

    chapters: list[Chapter] = []
    current: Chapter | None = None
    for block in blocks:
        if block.type == "heading" and block.level == break_level:
            current = Chapter(title=block.text.strip(), level=break_level, blocks=[])
            chapters.append(current)
            continue
        if current is None:
            current = Chapter(title=fallback_title, level=1, blocks=[])
            chapters.append(current)
        current.blocks.append(block)

    return [
        chapter
        for chapter in chapters
        if any(block.text.strip() for block in chapter.blocks)
    ]


def strip_duplicate_leading_heading(blocks: list[Block], title: str) -> list[Block]:
    """Drop a leading heading that merely repeats the chapter title."""
    if not blocks:
        return blocks
    first = blocks[0]
    if first.type == "heading" and _normalized(first.text) == _normalized(title):
        return blocks[1:]
    return blocks


def drop_empty_blocks(blocks: list[Block]) -> list[Block]:
    """Return only blocks that contain non-whitespace text."""
    return [block for block in blocks if block.text.strip()]


def render_book_markdown(book: BookDocument) -> str:
    """Reconstruct a readable Markdown rendering of a parsed book.

    Args:
        book: Parsed book document.

    Returns:
        Markdown text with the book title, authors and each chapter.
    """
    parts: list[str] = [f"# {book.title}"]
    if book.authors:
        parts.append("by " + ", ".join(book.authors))
    for chapter in book.chapters:
        parts.append(f"# {chapter.title}")
        rendered = blocks_to_markdown(chapter.blocks).strip()
        if rendered:
            parts.append(rendered)
    return "\n\n".join(part for part in parts if part.strip()) + "\n"


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())
