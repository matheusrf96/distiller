"""Unit tests for contextual enrichment wired into index builds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from distiller.indexing import build_index, load_index
from distiller.indexing.embedder import HashingEmbedder
from distiller.ingest import ingest_book
from distiller.paths import BookPaths
from distiller.utils import read_jsonl, write_json

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from distiller.config import Settings


class RecordingEmbedder(HashingEmbedder):
    """Hashing embedder that records every document batch it receives."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim=dim)
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.batches.append(list(texts))
        return super().embed_documents(texts)


class BranchingLLM:
    """Returns a context that names the chunk's own subject."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        """Stable identifier for the stub."""
        return "branching"

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.prompts.append(user)
        # Branch on the chunk/chapter content, not on the book title, which
        # appears in every prompt.
        if "three hundred steps" in user:
            return "A keeper scene about counting steps uniqueword beta."
        if "cracked" in user:
            return "A storm scene about the cracked lantern uniqueword alpha."
        return "A generic passage context for indexing."


def prepare_book(sample_epub: Path, settings: Settings) -> str:
    """Ingest the fixture book, returning its id."""
    book = ingest_book(sample_epub, settings=settings)
    paths = BookPaths.for_book(settings.artifacts_dir, book.book_id).ensure()
    write_json(paths.book_json, book.model_dump())
    return book.book_id


def test_contextual_index_embeds_index_text(
    settings: Settings, sample_epub: Path, monkeypatch
) -> None:
    """Contexts reach the dense side and persist in chunks.jsonl.

    Covers REQ-CR-001, REQ-CR-002 and REQ-CR-007.
    """
    book_id = prepare_book(sample_epub, settings)
    settings.enrichment.enabled = True
    monkeypatch.setattr(
        "distiller.indexing.bundle.get_llm", lambda _settings: BranchingLLM()
    )

    embedder = RecordingEmbedder(dim=64)
    metadata = build_index(book_id, settings, embedder=embedder)

    embedded = [text for batch in embedder.batches for text in batch]
    assert metadata["contextual"] is True
    assert metadata["enriched_chunks"] == len(embedded) > 0
    assert all(text.startswith("A ") for text in embedded)  # context prefix present

    rows = list(
        read_jsonl(BookPaths.for_book(settings.artifacts_dir, book_id).chunks_jsonl)
    )
    assert all(row["context"] for row in rows)


def test_contextual_bm25_matches_context_only_tokens(
    settings: Settings, sample_epub: Path, monkeypatch
) -> None:
    """The lexical side indexes index_text, so context-only terms are searchable."""
    book_id = prepare_book(sample_epub, settings)
    settings.enrichment.enabled = True
    monkeypatch.setattr(
        "distiller.indexing.bundle.get_llm", lambda _settings: BranchingLLM()
    )
    build_index(book_id, settings, embedder=RecordingEmbedder(dim=64))

    bundle = load_index(book_id, settings, embedder=RecordingEmbedder(dim=64))
    hits = bundle.bm25.search("uniqueword alpha", k=3)

    assert hits
    top_chunk = bundle.chunk_by_id[hits[0].id]
    assert top_chunk.context is not None
    assert "alpha" in top_chunk.context
    assert "uniqueword alpha" not in top_chunk.text  # proves the context did the work


def test_disabled_enrichment_indexes_original_text(
    settings: Settings, sample_epub: Path
) -> None:
    """The default build performs no enrichment and changes no text (REQ-CR-005)."""
    book_id = prepare_book(sample_epub, settings)

    embedder = RecordingEmbedder(dim=64)
    metadata = build_index(book_id, settings, embedder=embedder)

    embedded = [text for batch in embedder.batches for text in batch]
    assert metadata["contextual"] is False
    assert metadata["enriched_chunks"] == 0
    rows = list(
        read_jsonl(BookPaths.for_book(settings.artifacts_dir, book_id).chunks_jsonl)
    )
    assert all(row["context"] is None for row in rows)
    assert all(not text.startswith("A ") for text in embedded)


def test_fake_llm_answers_excerpt_prompts() -> None:
    """The offline client returns usable contexts for enrichment prompts.

    Covers REQ-CR-012.
    """
    from distiller.llm.fake import FakeLLM

    answer: Any = FakeLLM().complete(
        system="s", user="<excerpt>\nThe lantern cracked.\n</excerpt>"
    )

    assert answer == "A passage about: The lantern cracked."
