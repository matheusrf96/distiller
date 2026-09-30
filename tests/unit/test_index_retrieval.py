"""Unit tests for index building and hybrid retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from distiller.exceptions import IndexBuildError, IndexNotFoundError
from distiller.indexing import build_index, load_index
from distiller.ingest import ingest_book
from distiller.paths import BookPaths
from distiller.rag import Retriever
from distiller.utils import write_json

if TYPE_CHECKING:
    from distiller.config import Settings


def ingest_and_index(sample_epub: Path, settings: Settings) -> tuple[str, dict]:
    """Ingest the fixture book and build its index, returning id and metadata."""
    book = ingest_book(sample_epub, settings=settings)
    paths = BookPaths.for_book(settings.artifacts_dir, book.book_id).ensure()
    write_json(paths.book_json, book.model_dump())
    metadata = build_index(book.book_id, settings)
    return book.book_id, metadata


def test_build_index_and_retrieve(settings: Settings, sample_epub: Path) -> None:
    """Indexing persists artifacts and retrieval finds the right chapter."""
    book_id, metadata = ingest_and_index(sample_epub, settings)

    assert metadata["chunk_count"] >= 2
    assert metadata["store"] == "numpy"
    assert Path(settings.artifacts_dir, book_id, "chunks.jsonl").exists()

    bundle = load_index(book_id, settings)
    assert bundle.chunk_by_id

    retriever = Retriever(bundle, settings.retrieval)
    hits = retriever.search("What happened to the lantern during the storm?")

    assert hits
    assert "Chapter Two" in [hit.chunk.chapter for hit in hits[:3]]


def test_chapter_filter_restricts_results(
    settings: Settings, sample_epub: Path
) -> None:
    """The chapter filter applies to dense and sparse candidates alike."""
    book_id, _ = ingest_and_index(sample_epub, settings)
    bundle = load_index(book_id, settings)
    retriever = Retriever(bundle, settings.retrieval)

    hits = retriever.search("colour of the tower", chapter="Chapter Three")

    assert hits
    assert all(hit.chunk.chapter == "Chapter Three" for hit in hits)


def test_index_records_embedder_identity(settings: Settings, sample_epub: Path) -> None:
    """A different embedder is rejected instead of returning garbage."""
    book_id, metadata = ingest_and_index(sample_epub, settings)
    assert metadata["embedder"] == "hash:512"

    settings.embedding.hash_dim = 256
    with pytest.raises(IndexBuildError, match="built with embedder"):
        load_index(book_id, settings)


def test_missing_index_raises_helpful_error(settings: Settings) -> None:
    """Loading an unknown book id raises IndexNotFoundError."""
    with pytest.raises(IndexNotFoundError, match="distiller index"):
        load_index("does-not-exist", settings)


def test_building_index_without_ingest_raises(settings: Settings) -> None:
    """Building an index before ingesting raises BookNotFoundError."""
    from distiller.exceptions import BookNotFoundError

    with pytest.raises(BookNotFoundError, match="distiller ingest"):
        build_index("never-ingested", settings)
