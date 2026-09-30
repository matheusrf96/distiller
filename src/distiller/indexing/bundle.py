"""Build and load a book index (chunks + dense vectors + BM25)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..chunking import chunk_document
from ..exceptions import BookNotFoundError, IndexBuildError, IndexNotFoundError
from ..models import BookDocument, Chunk
from ..paths import BookPaths
from ..utils import read_json, read_jsonl, write_json, write_jsonl
from .bm25 import BM25Index
from .embedder import get_embedder
from .store import create_store, open_store

if TYPE_CHECKING:
    from ..config import Settings
    from .embedder import Embedder
    from .store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexBundle:
    """Everything needed to query one indexed book.

    A runtime container for live objects (store, embedder), not a serializable
    data model, so it stays a dataclass; the records inside it are pydantic.

    Attributes:
        book: Parsed book document.
        chunks: All chunks in reading order.
        store: Dense vector store.
        bm25: Sparse lexical index.
        embedder: Embedder that produced the query/document vectors.
        chunk_by_id: Lookup table from chunk id to chunk.
    """

    book: BookDocument
    chunks: list[Chunk]
    store: VectorStore
    bm25: BM25Index
    embedder: Embedder
    chunk_by_id: dict[str, Chunk] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_by_id:
            self.chunk_by_id = {chunk.id: chunk for chunk in self.chunks}


def build_index(
    book_id: str, settings: Settings, *, embedder: Embedder | None = None
) -> dict[str, Any]:
    """Chunk, embed and persist a retrieval index for one ingested book.

    Args:
        book_id: Slug of a previously ingested book.
        settings: Pipeline settings (chunking, embedding, store backend).
        embedder: Optional pre-built embedder (used by tests to stay offline).

    Returns:
        Index metadata written to ``index/metadata.json``.

    Raises:
        BookNotFoundError: If the book was never ingested.
        IndexBuildError: If chunking produced no chunks.
    """
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    if not paths.book_json.exists():
        raise BookNotFoundError(
            f"No ingested book for id '{book_id}' at {paths.root}. "
            f"Run `distiller ingest` first."
        )

    book = BookDocument.model_validate(read_json(paths.book_json))
    chunks = chunk_document(book, settings.chunking)
    if not chunks:
        raise IndexBuildError(
            f"Book '{book_id}' produced no chunks; check the parsed content."
        )
    write_jsonl(paths.chunks_jsonl, chunks)

    embedder = embedder or get_embedder(settings.embedding)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])

    store = create_store(paths.store_dir, embedder.dim, settings.store.backend)
    store.upsert(
        [chunk.id for chunk in chunks],
        vectors,
        payloads=[
            {
                "chunk_id": chunk.id,
                "chapter": chunk.chapter,
                "ordinal": chunk.ordinal,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            }
            for chunk in chunks
        ],
    )
    store.persist()

    metadata: dict[str, Any] = {
        "embedder": embedder.name,
        "dim": embedder.dim,
        "store": settings.store.backend,
        "chunk_count": len(chunks),
        "average_chunk_chars": round(
            sum(chunk.char_count for chunk in chunks) / len(chunks)
        ),
        "chunking": settings.chunking.model_dump(),
    }
    write_json(paths.index_metadata, metadata)
    return metadata


def load_index(
    book_id: str,
    settings: Settings,
    *,
    embedder: Embedder | None = None,
) -> IndexBundle:
    """Load a previously built index, ready for retrieval.

    Args:
        book_id: Slug of an indexed book.
        settings: Pipeline settings; the configured embedder must match the index.
        embedder: Optional pre-built embedder (used by tests to stay offline).

    Returns:
        Bundle with book, chunks, stores and embedder.

    Raises:
        IndexNotFoundError: If the book has not been indexed yet.
        IndexBuildError: If artifacts are corrupt or the configured embedder
            differs from the indexed one.
    """
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    if not paths.index_metadata.exists():
        raise IndexNotFoundError(
            f"No index for book '{book_id}' at {paths.index_dir}. "
            f"Run `distiller index {book_id}` first."
        )

    metadata = read_json(paths.index_metadata)
    embedder = embedder or get_embedder(settings.embedding)
    if metadata.get("embedder") != embedder.name:
        raise IndexBuildError(
            f"Index '{book_id}' was built with embedder '{metadata.get('embedder')}', "
            f"but the current configuration uses '{embedder.name}'. "
            f"Re-run `distiller index {book_id}` or fix the embedding settings."
        )

    try:
        book = BookDocument.model_validate(read_json(paths.book_json))
        chunks = [Chunk.model_validate(row) for row in read_jsonl(paths.chunks_jsonl)]
    except (OSError, ValueError) as exc:
        raise IndexBuildError(
            f"Index artifacts for '{book_id}' are incomplete or corrupt. "
            f"Re-run `distiller ingest {book_id}` and `distiller index {book_id}`."
        ) from exc

    try:
        store = open_store(
            paths.store_dir, int(metadata["dim"]), str(metadata.get("store", "numpy"))
        )
    except (OSError, ValueError, KeyError) as exc:
        raise IndexBuildError(
            f"Vector store for '{book_id}' is incomplete or corrupt. "
            f"Re-run `distiller index {book_id}`."
        ) from exc

    bm25 = BM25Index([chunk.id for chunk in chunks], [chunk.text for chunk in chunks])
    return IndexBundle(
        book=book, chunks=chunks, store=store, bm25=bm25, embedder=embedder
    )


def list_books(artifacts_dir: Path | str) -> list[dict[str, Any]]:
    """Summarize every ingested book, for the CLI.

    Unreadable or corrupt book artifacts are skipped with a warning so that one
    bad directory cannot break the listing.

    Args:
        artifacts_dir: Root artifacts directory.

    Returns:
        One summary dict per ingested book, sorted by directory name.
    """
    root = Path(artifacts_dir)
    if not root.exists():
        return []
    books: list[dict[str, Any]] = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        book_json = child / "book.json"
        if not book_json.exists():
            continue
        try:
            data = read_json(book_json)
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable book artifact %s: %s", book_json, exc)
            continue
        books.append(
            {
                "book_id": data.get("book_id", child.name),
                "title": data.get("title", child.name),
                "format": data.get("source_format"),
                "chapters": len(data.get("chapters", [])),
                "indexed": (child / "index" / "metadata.json").exists(),
            }
        )
    return books
