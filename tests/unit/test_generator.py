"""Unit tests for grounded answer generation and citation extraction."""

from __future__ import annotations

from distiller.llm.fake import FakeLLM
from distiller.models import Chunk, RetrievedChunk
from distiller.rag import Generator, extract_citations, is_refusal


def build_chunk(ordinal: int, chapter: str, text: str, page: int) -> Chunk:
    """Build a chunk with the given provenance."""
    return Chunk(
        id=f"book:{ordinal:04d}:abcd",
        book_id="book",
        ordinal=ordinal,
        text=text,
        chapter=chapter,
        heading_path=[chapter],
        page_start=page,
        page_end=page,
        char_count=len(text),
    )


def build_contexts() -> list[RetrievedChunk]:
    """Two retrieved chunks from different chapters."""
    return [
        RetrievedChunk(
            chunk=build_chunk(
                0, "Chapter Two", "The lantern cracked during the storm of 1887.", 2
            ),
            score=0.9,
        ),
        RetrievedChunk(
            chunk=build_chunk(
                1, "Chapter Three", "The tower was painted white in the spring.", 3
            ),
            score=0.8,
        ),
    ]


def test_generator_answers_with_citations() -> None:
    """The generator records citations mapping back to chunk provenance."""
    generator = Generator(FakeLLM(), "The Lantern Keeper")
    answer = generator.answer("What happened to the lantern?", build_contexts())

    assert not answer.refused
    assert answer.citations
    assert answer.citations[0].index == 1
    assert answer.citations[0].chapter == "Chapter Two"
    assert answer.model == "fake"


def test_generator_refuses_without_context() -> None:
    """No contexts means an explicit refusal without calling the model."""
    generator = Generator(FakeLLM(), "The Lantern Keeper")
    answer = generator.answer("Who won the race?", [])

    assert answer.refused
    assert "couldn't find" in answer.text.lower()
    assert answer.citations == []


def test_extract_citations_maps_and_ignores_out_of_range() -> None:
    """Out-of-range markers are ignored; valid ones keep their order."""
    text = "First claim [1]. Second claim [2]. Bogus [9]."
    citations = extract_citations(text, build_contexts())

    assert [citation.index for citation in citations] == [1, 2]
    assert citations[1].chunk_id == build_contexts()[1].chunk.id


def test_is_refusal_detection() -> None:
    """The refusal sentence is detected case-insensitively."""
    assert is_refusal("I couldn't find that in The Book.", "The Book")
    assert is_refusal("Sure. I couldn't find that in The Book.", "The Book")
    assert not is_refusal("The lantern cracked [1].", "The Book")


def test_generator_keeps_hedged_but_cited_answer_as_grounded() -> None:
    """An answer that hedges but cites the book is grounded, not a refusal."""
    hedged = FakeLLM(
        response=(
            "I couldn't find the exact wording, but the book says "
            "the lantern cracked during the storm [1]."
        )
    )
    generator = Generator(hedged, "The Lantern Keeper")
    answer = generator.answer("What happened to the lantern?", build_contexts())

    assert not answer.refused
    assert answer.citations


def test_generator_flags_hedged_answer_without_citation_as_refusal() -> None:
    """Refusal wording without any citation still counts as a refusal."""
    hedged = FakeLLM(response="I couldn't find that in The Lantern Keeper.")
    generator = Generator(hedged, "The Lantern Keeper")
    answer = generator.answer("Who won the race?", build_contexts())

    assert answer.refused
    assert answer.citations == []


def test_citation_label_includes_location() -> None:
    """Citation labels combine chapter, section and page range."""
    context = RetrievedChunk(
        chunk=Chunk(
            id="b:0000:x",
            book_id="b",
            ordinal=0,
            text="text",
            chapter="Chapter Two",
            heading_path=["Chapter Two", "The Storm"],
            page_start=12,
            page_end=13,
        )
    )
    answer = Generator(FakeLLM(), "Book").answer("q", [context])

    assert answer.citations[0].label == "Chapter Two — The Storm (pp. 12-13)"
