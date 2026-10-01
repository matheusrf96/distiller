"""Unit tests for the QA pipeline flow and reranker wiring."""

from __future__ import annotations

from typing import Any

import pytest

from distiller.config import RetrievalSettings
from distiller.exceptions import MissingDependencyError
from distiller.llm.fake import FakeLLM
from distiller.models import Chunk, RetrievedChunk
from distiller.optional_deps import is_available
from distiller.rag import Generator, QAPipeline, get_reranker


def build_contexts(count: int) -> list[RetrievedChunk]:
    """Build ``count`` contexts with descending scores."""
    return [
        RetrievedChunk(
            chunk=Chunk(
                id=f"b:{index:04d}:x",
                book_id="b",
                ordinal=index,
                text=f"passage {index} about the lantern and the storm",
                chapter=f"Chapter {index}",
                heading_path=[f"Chapter {index}"],
            ),
            score=1.0 - index / 100,
        )
        for index in range(count)
    ]


class StubRetriever:
    """Records search calls and returns the given contexts."""

    def __init__(
        self, settings: RetrievalSettings, contexts: list[RetrievedChunk]
    ) -> None:
        self.settings = settings
        self.contexts = contexts
        self.calls: list[tuple[str, int | None, str | None]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        chapter: str | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, top_k, chapter))
        return self.contexts[:top_k] if top_k else self.contexts


class StubReranker:
    """Records rerank calls and reverses the candidate order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    @property
    def name(self) -> str:
        """Identifier of the stub reranker."""
        return "stub"

    def rerank(
        self, query: str, items: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        self.calls.append((query, len(items), top_k))
        return list(reversed(items))[:top_k]


def test_pipeline_without_reranker_uses_final_top_k() -> None:
    """Without a reranker the retriever is asked for exactly top_k_final."""
    settings = RetrievalSettings(top_k_final=3, rerank=False)
    retriever = StubRetriever(settings, build_contexts(10))
    pipeline = QAPipeline(retriever, Generator(FakeLLM(), "Book"), settings=settings)

    answer = pipeline.ask("what happened?")

    assert retriever.calls == [("what happened?", 3, None)]
    assert len(answer.contexts) == 3


def test_pipeline_with_reranker_widens_pool_then_trims() -> None:
    """A reranker sees the configured pool and trims it back to top_k_final."""
    settings = RetrievalSettings(top_k_final=2, rerank_pool=6, rerank=True)
    retriever = StubRetriever(settings, build_contexts(10))
    reranker = StubReranker()
    pipeline = QAPipeline(
        retriever, Generator(FakeLLM(), "Book"), reranker=reranker, settings=settings
    )

    answer = pipeline.ask("what happened?", chapter="Chapter 1")

    assert retriever.calls == [("what happened?", 6, "Chapter 1")]
    assert reranker.calls == [("what happened?", 6, 2)]
    assert len(answer.contexts) == 2
    assert answer.contexts[0].score < answer.contexts[1].score  # reversed by the stub


def test_pipeline_short_circuits_refusal_on_empty_retrieval() -> None:
    """No retrieved contexts means refusal without calling the model."""
    settings = RetrievalSettings(top_k_final=3)
    retriever = StubRetriever(settings, [])
    pipeline = QAPipeline(retriever, Generator(FakeLLM(), "Book"), settings=settings)

    answer = pipeline.ask("unanswerable?")

    assert answer.refused
    assert answer.citations == []


def test_get_reranker_disabled_returns_none() -> None:
    """Reranking disabled yields no reranker."""
    assert get_reranker(RetrievalSettings(rerank=False)) is None


def test_get_reranker_requires_embed_extra() -> None:
    """Enabling reranking without sentence-transformers names the extra."""
    if is_available("sentence_transformers"):
        pytest.skip("sentence-transformers is installed in this environment")

    with pytest.raises(MissingDependencyError, match="--extra embed"):
        get_reranker(RetrievalSettings(rerank=True, rerank_model="some-model"))


def test_prompts_render_provenance_attributes() -> None:
    """Rendered documents carry chapter/section/page attributes and numbering."""
    from distiller.rag.prompts import (
        build_system_prompt,
        build_user_prompt,
        render_documents,
    )

    contexts = build_contexts(2)
    rendered = render_documents(contexts)
    prompt = build_user_prompt("what happened?", contexts)

    assert '<doc id="1" chapter="Chapter 0"' in rendered
    assert "what happened?" in prompt
    assert "couldn't find" in build_system_prompt("The Book")


def test_render_documents_skips_absent_provenance() -> None:
    """Chunks without a heading or page render only id and chapter."""
    from distiller.rag.prompts import render_documents

    context = RetrievedChunk(
        chunk=Chunk(id="c0", book_id="b", ordinal=0, text="text", chapter="One")
    )

    rendered = render_documents([context])

    assert 'chapter="One"' in rendered
    assert "section=" not in rendered
    assert "pages=" not in rendered


def test_answer_model_dump_is_json_serializable() -> None:
    """Answers serialize cleanly for the --json CLI mode."""
    settings = RetrievalSettings(top_k_final=1)
    retriever = StubRetriever(settings, build_contexts(2))
    pipeline = QAPipeline(retriever, Generator(FakeLLM(), "Book"), settings=settings)

    payload: dict[str, Any] = pipeline.ask("what happened?").model_dump()

    assert payload["question"] == "what happened?"
    assert payload["citations"]
