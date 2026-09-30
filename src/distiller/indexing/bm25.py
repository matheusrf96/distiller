"""Sparse lexical retrieval with BM25 (per-book scale: rebuilt from chunks.jsonl)."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .hits import SearchHit

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase a text and split it into word tokens."""
    return _TOKEN.findall(text.lower())


class BM25Index:
    """BM25 lexical index over a book's chunks.

    Attributes:
        ids: Chunk ids in index order.
    """

    def __init__(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Build the index.

        Args:
            chunk_ids: Chunk ids, aligned with ``texts``.
            texts: Chunk texts.

        Raises:
            ValueError: When the two lists have different lengths.
        """
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must have the same length")
        self.ids = list(chunk_ids)
        self._bm25 = BM25Okapi([tokenize(text) for text in texts]) if texts else None

    def search(self, query: str, k: int) -> list[SearchHit]:
        """Return the top ``k`` chunks by BM25 score.

        Args:
            query: Query text.
            k: Maximum number of hits.

        Returns:
            Hits with positive scores, best first.
        """
        if self._bm25 is None or not self.ids or k <= 0:
            return []
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            SearchHit(self.ids[int(i)], float(scores[int(i)]))
            for i in top
            if scores[int(i)] > 0
        ]
