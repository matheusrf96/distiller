"""Domain models shared across the pipeline.

Boundary data (parsed books, chunks, answers) uses pydantic models so it is
validated and serializable; internal hot-path values use dataclasses in their
own modules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .utils import stable_hash_hex

BlockType = Literal["heading", "paragraph", "list", "code", "table", "quote", "caption"]
SourceFormat = Literal["pdf", "epub", "md", "txt"]


class DomainModel(BaseModel):
    """Base for boundary models.

    Unknown fields are rejected so that typos and stale field names fail loudly
    instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")


class Block(DomainModel):
    """An atomic content block extracted from a book.

    Attributes:
        type: Kind of content (paragraph, heading, list, table, code, quote).
        text: Extracted text content.
        level: Heading level (1-6) when ``type`` is ``"heading"``.
        page: 1-based source page, when the parser provides it (PDF backends).
    """

    type: BlockType = "paragraph"
    text: str
    level: int | None = None
    page: int | None = None


class Chapter(DomainModel):
    """A top-level book chapter; headings inside ``blocks`` form sub-sections.

    Attributes:
        title: Chapter title, from the TOC or the first heading.
        level: Heading level that defined the chapter break (1 or 2).
        blocks: Ordered content blocks of the chapter.
    """

    title: str
    level: int = 1
    blocks: list[Block] = Field(default_factory=list)

    @property
    def char_count(self) -> int:
        """Total number of characters across the chapter's blocks."""
        return sum(len(block.text) for block in self.blocks)


class BookDocument(DomainModel):
    """Normalized representation of a parsed book.

    Attributes:
        book_id: Slug used for the artifact directory.
        title: Book title, from metadata or a CLI override.
        authors: Author names, when available.
        language: ISO language code, when available.
        source_path: Path of the original file.
        source_format: Original format (pdf, epub, md, txt).
        chapters: Parsed chapters in reading order.
        metadata: Parser-specific provenance (backend name, page count, ...).
        created_at: UTC timestamp of the parse.
    """

    book_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    language: str | None = None
    source_path: str
    source_format: SourceFormat
    chapters: list[Chapter] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def char_count(self) -> int:
        """Total number of characters across all chapters."""
        return sum(chapter.char_count for chapter in self.chapters)

    @property
    def block_count(self) -> int:
        """Total number of content blocks across all chapters."""
        return sum(len(chapter.blocks) for chapter in self.chapters)

    def full_text(self) -> str:
        """Return the whole book as plain text (titles plus block text)."""
        parts: list[str] = []
        for chapter in self.chapters:
            parts.append(chapter.title)
            parts.extend(block.text for block in chapter.blocks)
        return "\n\n".join(part for part in parts if part.strip())


class Chunk(DomainModel):
    """A retrieval unit with full provenance back to the book.

    Attributes:
        id: Stable content-addressed id (``book:ordinal:hash``).
        book_id: Owning book.
        ordinal: Position of the chunk within the book index.
        text: Chunk text as shown to the model; never synthetically modified.
        context: LLM-generated situating context used only for indexing.
        chapter: Chapter title the chunk belongs to.
        heading_path: Heading stack at the chunk's anchor content block.
        page_start: First source page covered, when known.
        page_end: Last source page covered, when known.
        char_start: Approximate start offset within the book text.
        char_end: Approximate end offset within the book text.
        char_count: Cached length of ``text``.
    """

    id: str
    book_id: str
    ordinal: int
    text: str
    context: str | None = None
    chapter: str
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    char_start: int = 0
    char_end: int = 0
    char_count: int = 0

    @staticmethod
    def make_id(book_id: str, ordinal: int, text: str) -> str:
        """Build a stable chunk id from its position and content.

        Args:
            book_id: Owning book slug.
            ordinal: Position within the book index.
            text: Chunk text (its hash disambiguates re-chunking).

        Returns:
            Id of the form ``<book_id>:<ordinal>:<content-hash>``.
        """
        return f"{book_id}:{ordinal:04d}:{stable_hash_hex(text, 10)}"

    @property
    def index_text(self) -> str:
        """Text used for embedding and lexical indexing (context prefix + text)."""
        return f"{self.context}\n\n{self.text}" if self.context else self.text


class RetrievedChunk(DomainModel):
    """A chunk returned by retrieval, with its score and provenance.

    Attributes:
        chunk: The retrieved chunk.
        score: Fusion (or rerank) score; higher is better.
        source: Retrieval stage that produced the score.
    """

    chunk: Chunk
    score: float = 0.0
    source: Literal["dense", "sparse", "hybrid", "rerank"] = "hybrid"


class Citation(DomainModel):
    """A machine-checkable reference from an answer back to a chunk.

    Attributes:
        index: The ``[n]`` marker used in the answer text.
        chunk_id: Chunk the marker refers to.
        chapter: Chapter title for display.
        heading_path: Heading stack at the chunk anchor.
        page_start: First source page covered, when known.
        page_end: Last source page covered, when known.
    """

    index: int
    chunk_id: str
    chapter: str
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None

    @property
    def label(self) -> str:
        """Human-readable location, e.g. ``Chapter Two — The Storm (pp. 12-13)``."""
        return format_location(
            self.chapter, self.heading_path, self.page_start, self.page_end
        )


class Answer(DomainModel):
    """A grounded answer with citations, contexts and model provenance.

    Attributes:
        question: The question that was asked.
        text: The model's (or refusal) text.
        citations: Citations extracted from ``text``.
        contexts: Chunks that were provided to the model.
        model: Identifier of the generating model.
        refused: True when the book did not contain the answer.
    """

    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    model: str | None = None
    refused: bool = False


def format_location(
    chapter: str,
    heading_path: list[str],
    page_start: int | None,
    page_end: int | None,
) -> str:
    """Render a human-readable provenance label.

    Args:
        chapter: Chapter title.
        heading_path: Heading stack at the chunk anchor.
        page_start: First source page covered, when known.
        page_end: Last source page covered, when known.

    Returns:
        Label such as ``Chapter Two — The Storm (pp. 12-13)``.
    """
    parts = [chapter]
    if heading_path and heading_path[-1] != chapter:
        parts.append(heading_path[-1])
    label = " — ".join(parts)
    if page_start:
        pages = (
            f"p. {page_start}"
            if page_start == page_end
            else f"pp. {page_start}-{page_end}"
        )
        label += f" ({pages})"
    return label


REFUSAL_TEMPLATE = "I couldn't find that in {title}."


def refusal_text(book_title: str) -> str:
    """Return the canonical refusal sentence for a book.

    Shared by the RAG prompt and the RAFT training targets so that training
    data matches inference behaviour.

    Args:
        book_title: Title of the book being queried.

    Returns:
        The refusal sentence, e.g. ``I couldn't find that in The Book.``
    """
    return REFUSAL_TEMPLATE.format(title=book_title)
