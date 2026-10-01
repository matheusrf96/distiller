"""Unit tests for global (map-reduce) answering over the summary tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from distiller.config import ThematicSettings
from distiller.indexing.embedder import HashingEmbedder
from distiller.llm.fake import FakeLLM
from distiller.models import refusal_text
from distiller.thematic import GlobalPipeline, extract_summary_citations

if TYPE_CHECKING:
    from collections.abc import Callable

    from distiller.models import BookDocument, Chunk, SummaryTree

    CorpusFactory = Callable[..., tuple[BookDocument, list[Chunk]]]
    TreeFactory = Callable[..., SummaryTree]

QUESTION = "What happened to the lantern during the storm?"


def test_global_ask_maps_selected_summaries_and_reduces(
    corpus_factory: CorpusFactory,
    tree_factory: TreeFactory,
) -> None:
    """ask() makes one map call per selected summary plus one reduce call (AC7)."""
    book, chunks = corpus_factory()
    tree = tree_factory(book, chunks, window_size=2)
    prompts: list[str] = []

    def respond(user: str) -> str:
        prompts.append(user)
        return "Partial answer [1]"

    pipeline = GlobalPipeline(
        tree,
        FakeLLM(response=respond),
        ThematicSettings(map_top_k=2),
        embedder=HashingEmbedder(dim=128),
    )
    answer = pipeline.ask(QUESTION)

    assert len(prompts) == 3
    assert all("Summary documents:" in prompt for prompt in prompts[:2])
    assert "Partial answers:" in prompts[2]
    assert answer.mode == "global"
    assert answer.text == "Partial answer [1]"
    assert len(answer.summary_citations) == 1


def test_global_selection_falls_back_to_chapter_summaries_without_an_index(
    corpus_factory: CorpusFactory,
    tree_factory: TreeFactory,
) -> None:
    """Without an embedder every chapter summary is mapped in reading order (AC7)."""
    book, chunks = corpus_factory()
    tree = tree_factory(book, chunks, window_size=2)

    pipeline = GlobalPipeline(tree, FakeLLM(), ThematicSettings(map_top_k=2))
    selected = pipeline.select("What are the main themes?")

    chapter_ids = [node.id for node in tree.nodes if node.level == 1]
    assert [node.id for node in selected] == chapter_ids
    assert len(selected) == 3 > pipeline.settings.map_top_k


def test_global_answer_cites_summary_nodes(
    corpus_factory: CorpusFactory,
    tree_factory: TreeFactory,
) -> None:
    """Summary citations map [n] markers back to node ids and titles (AC8)."""
    book, chunks = corpus_factory()
    tree = tree_factory(book, chunks, window_size=2)
    pipeline = GlobalPipeline(
        tree,
        FakeLLM(),
        ThematicSettings(map_top_k=1),
        embedder=HashingEmbedder(dim=128),
    )

    answer = pipeline.ask(QUESTION)
    selected = pipeline.select(QUESTION)

    assert answer.mode == "global"
    assert answer.refused is False
    assert answer.citations == []
    assert [citation.index for citation in answer.summary_citations] == [1]
    citation = answer.summary_citations[0]
    assert citation.node_id == selected[0].id
    assert citation.title == selected[0].title
    assert citation.level == selected[0].level


def test_global_answer_refuses_with_the_shared_sentence(
    corpus_factory: CorpusFactory,
    tree_factory: TreeFactory,
) -> None:
    """An unanswerable question returns the shared refusal sentence (AC8)."""
    book, chunks = corpus_factory()
    tree = tree_factory(book, chunks, window_size=2)
    refusal = refusal_text(book.title)
    pipeline = GlobalPipeline(
        tree,
        FakeLLM(response=lambda user: refusal),
        ThematicSettings(),
        embedder=HashingEmbedder(dim=128),
    )

    answer = pipeline.ask("Who won the village sailing race?")

    assert answer.refused is True
    assert answer.text == refusal
    assert answer.summary_citations == []
    assert answer.mode == "global"


def test_global_ask_refuses_when_no_summary_is_available(
    corpus_factory: CorpusFactory,
    tree_factory: TreeFactory,
) -> None:
    """A tree with no summary-bearing node refuses without a reduce call (AC8)."""
    book, chunks = corpus_factory()
    tree = tree_factory(book, chunks, failed_chapters=(1, 2, 3))
    assert all(node.summary is None for node in tree.nodes)

    def explode(user: str) -> str:
        pytest.fail("no LLM call expected without summaries")

    pipeline = GlobalPipeline(
        tree,
        FakeLLM(response=explode),
        ThematicSettings(),
        embedder=HashingEmbedder(dim=128),
    )

    answer = pipeline.ask("What are the book's main themes?")

    assert answer.refused is True
    assert answer.text == refusal_text(book.title)
    assert answer.summary_citations == []
    assert answer.mode == "global"


def test_reduce_strips_map_citation_markers(
    corpus_factory: CorpusFactory,
    tree_factory: TreeFactory,
) -> None:
    """Internal [1] markers from map partials never reach the reduce prompt (AC8)."""
    book, chunks = corpus_factory()
    tree = tree_factory(book, chunks, window_size=2)
    prompts: list[str] = []

    def respond(user: str) -> str:
        prompts.append(user)
        if "Partial answers:" in user:
            return "Final answer [2]"
        return "Partial from a summary [1]"

    pipeline = GlobalPipeline(
        tree,
        FakeLLM(response=respond),
        ThematicSettings(map_top_k=2),
        embedder=HashingEmbedder(dim=128),
    )
    answer = pipeline.ask(QUESTION)
    selected = pipeline.select(QUESTION)

    assert len(prompts) == 3
    assert "[1]" not in prompts[2]
    assert "Partial from a summary" in prompts[2]
    assert [citation.index for citation in answer.summary_citations] == [2]
    assert answer.summary_citations[0].node_id == selected[1].id


def test_extract_summary_citations_ignores_out_of_range_markers(
    corpus_factory: CorpusFactory,
    tree_factory: TreeFactory,
) -> None:
    """Markers beyond the selected nodes are dropped."""
    book, chunks = corpus_factory()
    tree = tree_factory(book, chunks)
    nodes = [node for node in tree.nodes if node.level == 1]

    citations = extract_summary_citations("Answer [1] and [9].", nodes)

    assert [citation.index for citation in citations] == [1]
