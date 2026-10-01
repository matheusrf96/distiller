"""Unit tests for index building and hybrid retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from distiller.exceptions import IndexBuildError, IndexNotFoundError
from distiller.indexing import build_index, list_books, load_index
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


def test_corrupt_store_raises_actionable_error(
    settings: Settings, sample_epub: Path
) -> None:
    """A deleted store directory is reported as an index problem, not a crash."""
    import shutil

    book_id, _ = ingest_and_index(sample_epub, settings)
    store_dir = BookPaths.for_book(settings.artifacts_dir, book_id).store_dir
    shutil.rmtree(store_dir)

    with pytest.raises(IndexBuildError, match="Vector store"):
        load_index(book_id, settings)


def test_missing_book_json_raises_actionable_error(
    settings: Settings, sample_epub: Path
) -> None:
    """A deleted book.json is reported as corruption with a recovery hint."""
    book_id, _ = ingest_and_index(sample_epub, settings)
    Path(settings.artifacts_dir, book_id, "book.json").unlink()

    with pytest.raises(IndexBuildError, match="incomplete or corrupt"):
        load_index(book_id, settings)


def test_list_books_skips_corrupt_entries(
    settings: Settings, sample_epub: Path
) -> None:
    """One unreadable book.json cannot break the listing."""
    book_id, _ = ingest_and_index(sample_epub, settings)
    Path(settings.artifacts_dir, book_id, "book.json").write_text(
        "{not json", encoding="utf-8"
    )

    assert list_books(settings.artifacts_dir) == []


def test_list_books_skips_directories_without_book_json(
    settings: Settings,
) -> None:
    """Directories without a book.json are skipped by the listing."""
    (settings.artifacts_dir / "not-a-book").mkdir(parents=True)

    assert list_books(settings.artifacts_dir) == []


def test_build_index_rejects_a_book_without_chunks(settings: Settings) -> None:
    """An ingested book with no chapters fails with an actionable error."""
    paths = BookPaths.for_book(settings.artifacts_dir, "empty-book").ensure()
    write_json(
        paths.book_json,
        {
            "book_id": "empty-book",
            "title": "Empty",
            "source_path": "empty.txt",
            "source_format": "txt",
        },
    )

    with pytest.raises(IndexBuildError, match="no chunks"):
        build_index("empty-book", settings)


def test_bm25_index_validates_lengths_and_empty_searches() -> None:
    """BM25 rejects mismatched inputs and returns no hits when empty."""
    from distiller.indexing.bm25 import BM25Index

    with pytest.raises(ValueError, match="same length"):
        BM25Index(["a"], [])
    assert BM25Index([], []).search("anything", k=3) == []


def test_index_bundle_reuses_a_provided_chunk_lookup() -> None:
    """An explicit chunk_by_id skips the automatic lookup construction."""
    from distiller.indexing import IndexBundle
    from distiller.indexing.bm25 import BM25Index
    from distiller.indexing.embedder import HashingEmbedder
    from distiller.indexing.store import NumpyStore
    from distiller.models import BookDocument, Chunk

    book = BookDocument(
        book_id="b", title="T", source_path="x.txt", source_format="txt"
    )
    chunk = Chunk(id="a", book_id="b", ordinal=0, text="t", chapter="c")
    bundle = IndexBundle(
        book=book,
        chunks=[],
        store=NumpyStore(dim=4),
        bm25=BM25Index([], []),
        embedder=HashingEmbedder(dim=4),
        chunk_by_id={"a": chunk},
    )

    assert set(bundle.chunk_by_id) == {"a"}
