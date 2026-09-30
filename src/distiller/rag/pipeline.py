"""End-to-end QA pipeline: retrieve -> (optionally rerank) -> generate."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import RetrievalSettings
    from ..models import Answer
    from .generator import Generator
    from .reranker import Reranker
    from .retriever import Retriever


class QAPipeline:
    """Answer questions about one book with citations.

    Attributes:
        retriever: Hybrid retriever for the book.
        generator: Grounded answer generator.
        reranker: Optional cross-encoder reranker.
        settings: Retrieval settings.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        *,
        reranker: Reranker | None = None,
        settings: RetrievalSettings | None = None,
    ) -> None:
        """Assemble the pipeline.

        Args:
            retriever: Hybrid retriever for the book.
            generator: Grounded answer generator.
            reranker: Optional cross-encoder reranker.
            settings: Retrieval settings; defaults to the retriever's.
        """
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.settings = settings or retriever.settings

    def ask(
        self,
        question: str,
        *,
        chapter: str | None = None,
        top_k: int | None = None,
    ) -> Answer:
        """Ask one question about the book.

        Args:
            question: Natural-language question.
            chapter: Optional chapter filter for retrieval.
            top_k: Number of contexts handed to the generator.

        Returns:
            Cited answer (or refusal) with the retrieved contexts attached.
        """
        final_k = top_k or self.settings.top_k_final
        pool_k = max(final_k, self.settings.rerank_pool) if self.reranker else final_k

        contexts = self.retriever.search(question, top_k=pool_k, chapter=chapter)
        if self.reranker and contexts:
            contexts = self.reranker.rerank(question, contexts, top_k=final_k)
        return self.generator.answer(question, contexts)
