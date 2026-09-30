"""Structure-aware chunking: chapter-bounded, heading-aware, paragraph-aligned."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..models import BookDocument, Chapter, Chunk

if TYPE_CHECKING:
    from ..config import ChunkingSettings


@dataclass(frozen=True, slots=True)
class _Piece:
    """Internal packing unit (created once per block/paragraph, hence a dataclass)."""

    text: str
    heading_path: tuple[str, ...]
    page: int | None
    start: int
    end: int
    is_heading: bool = False


def chunk_document(book: BookDocument, settings: ChunkingSettings) -> list[Chunk]:
    """Split a parsed book into retrieval chunks.

    Guarantees:

    * chunks never cross chapter boundaries;
    * every chunk carries its chapter, heading path and page range;
    * a chunk never exceeds ``settings.max_chars``;
    * overlap is whole-paragraph only (never mid-sentence).

    Args:
        book: Parsed book document.
        settings: Chunk sizing settings (validated bounds).

    Returns:
        Chunks in reading order with sequential ordinals.
    """
    chunks: list[Chunk] = []
    offset = 0
    for chapter in book.chapters:
        pieces = _chapter_pieces(
            chapter, start_offset=offset, max_chars=settings.max_chars
        )
        if not pieces:
            continue
        offset = pieces[-1].end + 2

        packer = _ChapterPacker(book, chapter, settings, start_ordinal=len(chunks))
        for piece in pieces:
            packer.feed(piece)
        chunks.extend(packer.finish())
    return chunks


class _ChapterPacker:
    """Accumulates pieces into chunks; keeps all packing rules in one place."""

    def __init__(
        self,
        book: BookDocument,
        chapter: Chapter,
        settings: ChunkingSettings,
        *,
        start_ordinal: int,
    ) -> None:
        self._book = book
        self._chapter = chapter
        self._settings = settings
        self._chunks: list[Chunk] = []
        self._buffer: list[_Piece] = []
        self._buffer_len = 0
        self._next_ordinal = start_ordinal

    def feed(self, piece: _Piece) -> None:
        piece_len = len(piece.text)
        # Never exceed max_chars: flush early and shrink the carry-over to fit.
        # Headings alone are not content, so they stay attached to what follows.
        has_content = any(not buffered.is_heading for buffered in self._buffer)
        if has_content and self._buffer_len + piece_len + 2 > self._settings.max_chars:
            self._flush(carry_budget=self._settings.max_chars - piece_len - 2)

        self._buffer.append(piece)
        self._buffer_len += piece_len + 2
        if self._buffer_len >= self._settings.target_chars:
            self._flush()

    def finish(self) -> list[Chunk]:
        self._flush()
        return self._chunks

    def _flush(self, carry_budget: int | None = None) -> None:
        if not self._buffer:
            return
        text = "\n\n".join(piece.text for piece in self._buffer).strip()
        if text:
            pages = [piece.page for piece in self._buffer if piece.page]
            anchor = next(
                (p for p in self._buffer if not p.is_heading), self._buffer[0]
            )
            self._chunks.append(
                Chunk(
                    id=Chunk.make_id(self._book.book_id, self._next_ordinal, text),
                    book_id=self._book.book_id,
                    ordinal=self._next_ordinal,
                    text=text,
                    chapter=self._chapter.title,
                    heading_path=list(anchor.heading_path),
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    char_start=self._buffer[0].start,
                    char_end=self._buffer[-1].end,
                    char_count=len(text),
                )
            )
            self._next_ordinal += 1
        self._buffer, self._buffer_len = self._carry_overlap(carry_budget)

    def _carry_overlap(self, budget: int | None) -> tuple[list[_Piece], int]:
        """Keep whole trailing pieces (<= limit) as overlap for the next chunk.

        Overlap is best-effort: if a single trailing piece is larger than the
        limit, the next chunk starts fresh rather than duplicating a big paragraph.
        """
        limit = (
            self._settings.overlap_chars
            if budget is None
            else max(0, min(self._settings.overlap_chars, budget))
        )
        if limit <= 0:
            return [], 0
        keep: list[_Piece] = []
        total = 0
        for piece in reversed(self._buffer):
            piece_len = len(piece.text) + 2
            if total + piece_len > limit:
                break
            keep.insert(0, piece)
            total += piece_len
        return keep, total


def _chapter_pieces(
    chapter: Chapter, *, start_offset: int, max_chars: int
) -> list[_Piece]:
    pieces: list[_Piece] = []
    heading_stack: list[tuple[int, str]] = []
    offset = start_offset

    for block in chapter.blocks:
        text = block.text.strip()
        if not text:
            continue
        level = block.level if block.type == "heading" else None
        if level:
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))

        heading_path = tuple(title for _, title in heading_stack)
        for part in _split_oversized(text, max_chars):
            pieces.append(
                _Piece(
                    text=part,
                    heading_path=heading_path,
                    page=block.page,
                    start=offset,
                    end=offset + len(part),
                    is_heading=bool(level),
                )
            )
            offset += len(part) + 2
    return pieces


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Hard-split a block that cannot fit in one chunk (at newline/word boundaries)."""
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        cut = remaining.rfind("\n", 0, max_chars)
        if cut < max_chars // 2:
            cut = remaining.rfind(" ", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        parts.append(remaining)
    return [part for part in parts if part]
