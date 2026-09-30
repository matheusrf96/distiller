"""Optional cross-encoder reranking of retrieved candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ..optional_deps import require

if TYPE_CHECKING:
    from ..config import RetrievalSettings
    from ..models import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def rerank(
        self, query: str, items: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class CrossEncoderReranker:
    """Local cross-encoder (default: BAAI/bge-reranker-v2-m3, multilingual)."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        """Load the cross-encoder.

        Args:
            model_name: Hugging Face cross-encoder model name.
            device: Torch device override (e.g. ``"cpu"``, ``"cuda"``).
            batch_size: Scoring batch size.
        """
        module = require(
            "sentence_transformers", extra="embed", purpose="Cross-encoder reranking"
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = module.CrossEncoder(model_name, device=device)

    @property
    def name(self) -> str:
        """Stable identifier, recorded on reranked results."""
        return f"cross-encoder:{self.model_name}"

    def rerank(
        self, query: str, items: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Rescore candidates with the cross-encoder.

        Args:
            query: Original user question.
            items: Candidate chunks from hybrid retrieval.
            top_k: Number of items to keep.

        Returns:
            The ``top_k`` highest-scoring chunks, best first.
        """
        if not items:
            return []
        scores = self._model.predict(
            [(query, item.chunk.text) for item in items],
            batch_size=self.batch_size,
        )
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        order = np.argsort(-scores)[:top_k]
        return [
            items[int(i)].model_copy(
                update={"score": float(scores[int(i)]), "source": "rerank"}
            )
            for i in order
        ]


def get_reranker(settings: RetrievalSettings) -> Reranker | None:
    """Build the configured reranker, or None when reranking is disabled.

    Args:
        settings: Retrieval settings.

    Returns:
        A cross-encoder reranker when enabled, otherwise None.
    """
    if not settings.rerank:
        return None
    return CrossEncoderReranker(settings.rerank_model)
