"""Hybrid retrieval: dense (vector store) + sparse (BM25), fused with RRF."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import RetrievedChunk

if TYPE_CHECKING:
    from ..config import RetrievalSettings
    from ..indexing.bundle import IndexBundle


class Retriever:
    """Hybrid retriever over one indexed book.

    Attributes:
        bundle: Loaded index (chunks, vector store, BM25, embedder).
        settings: Retrieval and fusion settings.
    """

    def __init__(self, bundle: IndexBundle, settings: RetrievalSettings) -> None:
        """Prepare the retriever for one book.

        Args:
            bundle: Loaded index bundle.
            settings: Retrieval settings (top-k values and RRF constant).
        """
        self.bundle = bundle
        self.settings = settings

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        chapter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query.

        Dense and BM25 candidate lists are fused with reciprocal rank fusion.

        Args:
            query: Natural-language question or keywords.
            top_k: Number of chunks to return; defaults to ``settings.top_k_final``.
            chapter: Optional chapter filter (case-insensitive substring match).

        Returns:
            Chunks ordered by fused score, best first.
        """
        top_k = top_k or self.settings.top_k_final

        query_vector = self.bundle.embedder.embed_query(query)
        dense_hits = self.bundle.store.search(query_vector, k=self.settings.top_k_dense)
        sparse_hits = self.bundle.bm25.search(query, k=self.settings.top_k_sparse)

        fused: dict[str, float] = {}
        for hits in (dense_hits, sparse_hits):
            for rank, hit in enumerate(hits):
                chunk = self.bundle.chunk_by_id.get(hit.id)
                if chunk is None or not _chapter_matches(chunk.chapter, chapter):
                    continue
                fused[hit.id] = fused.get(hit.id, 0.0) + 1.0 / (
                    self.settings.rrf_k + rank + 1
                )

        ranked = sorted(fused.items(), key=lambda item: -item[1])[:top_k]
        return [
            RetrievedChunk(
                chunk=self.bundle.chunk_by_id[chunk_id], score=score, source="hybrid"
            )
            for chunk_id, score in ranked
        ]


def _chapter_matches(chunk_chapter: str, wanted: str | None) -> bool:
    if not wanted:
        return True
    return wanted.strip().lower() in chunk_chapter.lower()
