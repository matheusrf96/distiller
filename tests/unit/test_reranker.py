"""Unit tests for the optional cross-encoder reranker."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from distiller.models import Chunk, RetrievedChunk
from distiller.rag.reranker import CrossEncoderReranker

if TYPE_CHECKING:
    import pytest


class FakeCrossEncoder:
    """Deterministic stand-in for sentence_transformers.CrossEncoder."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def predict(
        self, pairs: list[tuple[str, str]], batch_size: int = 16
    ) -> list[float]:
        """Score each pair by the passage length."""
        return [float(len(passage)) for _, passage in pairs]


def fake_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the reranker's require() to the fake module."""
    module = SimpleNamespace(CrossEncoder=FakeCrossEncoder)
    monkeypatch.setattr("distiller.rag.reranker.require", lambda name, **kwargs: module)


def build_context(text: str) -> RetrievedChunk:
    """A retrieved chunk carrying ``text``."""
    return RetrievedChunk(
        chunk=Chunk(
            id=f"c:{text[:6]}",
            book_id="the-book",
            ordinal=0,
            text=text,
            chapter="One",
        ),
        score=0.1,
    )


def test_cross_encoder_reranker_scores_and_trims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rerank reorders by score, keeps top_k and tags the source."""
    fake_cross_encoder(monkeypatch)
    reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3", batch_size=2)
    items = [build_context("short"), build_context("a much longer passage")]

    reranked = reranker.rerank("q", items, top_k=1)

    assert reranker.name == "cross-encoder:BAAI/bge-reranker-v2-m3"
    assert [item.chunk.text for item in reranked] == ["a much longer passage"]
    assert reranked[0].source == "rerank"
    assert reranked[0].score == float(len("a much longer passage"))
    assert reranker.rerank("q", [], top_k=3) == []
