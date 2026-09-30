"""Unit tests for structure-aware chunking."""

from __future__ import annotations

from distiller.chunking import chunk_document
from distiller.config import ChunkingSettings
from distiller.models import Block, BookDocument, Chapter


def build_book() -> BookDocument:
    """Build a three-chapter synthetic book with paragraphs."""
    chapters = []
    for chapter_index in range(1, 4):
        blocks: list[Block] = [
            Block(type="heading", text=f"Chapter {chapter_index}", level=1)
        ]
        for paragraph in range(1, 8):
            blocks.append(
                Block(
                    type="paragraph",
                    text=(
                        f"Paragraph {paragraph} of chapter {chapter_index} "
                        + "tells a story about the sea, the wind and the lantern. " * 4
                    ),
                    page=chapter_index,
                )
            )
        chapters.append(Chapter(title=f"Chapter {chapter_index}", blocks=blocks))
    return BookDocument(
        book_id="test-book",
        title="Test Book",
        source_path="test.epub",
        source_format="epub",
        chapters=chapters,
    )


def test_chunks_cover_all_chapters_and_are_well_formed() -> None:
    """Chunks are non-empty, uniquely identified and chapter-tagged."""
    book = build_book()
    chunks = chunk_document(
        book, ChunkingSettings(target_chars=600, overlap_chars=80, max_chars=1200)
    )

    assert chunks, "expected chunks"
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(
        chunk.chapter in {"Chapter 1", "Chapter 2", "Chapter 3"} for chunk in chunks
    )
    assert all(chunk.char_count == len(chunk.text) for chunk in chunks)
    assert {chunk.chapter for chunk in chunks} == {
        "Chapter 1",
        "Chapter 2",
        "Chapter 3",
    }


def test_chunks_stay_within_size_bounds() -> None:
    """Chunks never exceed max_chars."""
    book = build_book()
    chunks = chunk_document(
        book, ChunkingSettings(target_chars=600, overlap_chars=0, max_chars=1200)
    )

    assert all(chunk.char_count <= 1200 for chunk in chunks)
    assert len(chunks) >= book.char_count // 1200


def test_overlap_does_not_reduce_chunk_count() -> None:
    """Overlap repeats boundary text rather than merging content."""
    book = build_book()
    without = chunk_document(
        book, ChunkingSettings(target_chars=600, overlap_chars=0, max_chars=1200)
    )
    with_overlap = chunk_document(
        book, ChunkingSettings(target_chars=600, overlap_chars=200, max_chars=1200)
    )

    assert len(with_overlap) >= len(without)


def test_oversized_paragraph_is_split() -> None:
    """A block larger than max_chars is hard-split without exceeding the limit."""
    book = BookDocument(
        book_id="big",
        title="Big",
        source_path="big.txt",
        source_format="txt",
        chapters=[
            Chapter(title="Only", blocks=[Block(type="paragraph", text="word " * 2000)])
        ],
    )
    chunks = chunk_document(
        book, ChunkingSettings(target_chars=500, overlap_chars=50, max_chars=800)
    )

    assert len(chunks) > 1
    assert all(chunk.char_count <= 800 for chunk in chunks)


def test_heading_path_is_tracked() -> None:
    """Chunks carry the heading stack of their anchor content block."""
    book = BookDocument(
        book_id="paths",
        title="Paths",
        source_path="p.md",
        source_format="md",
        chapters=[
            Chapter(
                title="One",
                blocks=[
                    Block(type="heading", text="One", level=1),
                    Block(type="heading", text="Deep Section", level=2),
                    Block(type="paragraph", text="Content under a deep section. " * 30),
                ],
            )
        ],
    )
    chunks = chunk_document(
        book, ChunkingSettings(target_chars=300, overlap_chars=0, max_chars=600)
    )

    assert chunks
    assert "Deep Section" in chunks[0].heading_path
