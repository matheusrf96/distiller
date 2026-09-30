"""Unit tests for RAFT example formatting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from distiller.synthesis.qa import QAPair
from distiller.synthesis.raft import build_examples

if TYPE_CHECKING:
    from collections.abc import Callable

    from distiller.models import BookDocument, Chunk

    CorpusFactory = Callable[..., tuple[BookDocument, list[Chunk]]]

QUOTE = "the lantern cracked, and the keeper rowed to the village"


def build_pair(chunk: Chunk) -> QAPair:
    """Build a pair targeting the given chunk."""
    return QAPair(
        id=f"{chunk.id}:test",
        book_id=chunk.book_id,
        chunk_id=chunk.id,
        chapter=chunk.chapter,
        heading_path=list(chunk.heading_path),
        question="What happened to the lantern during the storm?",
        answer="The lantern cracked and the keeper rowed to the village.",
        quotes=[QUOTE],
        model="stub",
    )


def test_positive_example_includes_golden_and_distractors(
    corpus_factory: CorpusFactory,
) -> None:
    """The golden chunk is one of the contexts and drives the citation.

    Covers REQ-SQ-006 and REQ-SQ-007.
    """
    book, chunks = corpus_factory()
    pair = build_pair(chunks[0])

    examples = build_examples(
        book.title, [pair], chunks, distractors=2, negative_ratio=0.0, seed=7
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.answerable is True
    assert example.book_id == book.book_id
    assert example.id == f"{pair.id}:raft"
    assert len(example.contexts) == 3

    golden_positions = [
        i for i, context in enumerate(example.contexts) if context.is_golden
    ]
    assert len(golden_positions) == 1
    assert example.contexts[golden_positions[0]].chunk_id == chunks[0].id

    assert example.answer.startswith('The relevant passage says: "')
    assert QUOTE in example.answer
    assert f"[{golden_positions[0] + 1}]" in example.answer
    assert pair.answer in example.answer


def test_negative_examples_omit_the_golden_chunk(corpus_factory: CorpusFactory) -> None:
    """Negatives provide distractors only and target the refusal (REQ-SQ-008)."""
    book, chunks = corpus_factory()

    examples = build_examples(
        book.title,
        [build_pair(chunks[0])],
        chunks,
        distractors=2,
        negative_ratio=1.0,
        seed=7,
    )

    example = examples[0]
    assert example.answerable is False
    assert example.contexts
    assert all(not context.is_golden for context in example.contexts)
    assert "couldn't find" in example.answer.lower()
    assert "The Lantern Keeper" in example.answer


def test_zero_distractors_keeps_only_the_golden_context(
    corpus_factory: CorpusFactory,
) -> None:
    """distractors=0 yields a single-context example with citation [1]."""
    book, chunks = corpus_factory()

    examples = build_examples(
        book.title,
        [build_pair(chunks[1])],
        chunks,
        distractors=0,
        negative_ratio=0.0,
        seed=3,
    )

    assert len(examples[0].contexts) == 1
    assert examples[0].contexts[0].is_golden
    assert "[1]" in examples[0].answer


def test_same_seed_is_reproducible(corpus_factory: CorpusFactory) -> None:
    """Identical inputs and seed produce identical datasets (CON-SQ-004)."""
    book, chunks = corpus_factory()
    pairs = [build_pair(chunk) for chunk in chunks]

    first = build_examples(
        book.title, pairs, chunks, distractors=2, negative_ratio=0.5, seed=13
    )
    second = build_examples(
        book.title, pairs, chunks, distractors=2, negative_ratio=0.5, seed=13
    )

    assert first == second
    assert [example.id for example in first] == [example.id for example in second]


def test_examples_cover_every_kept_pair(corpus_factory: CorpusFactory) -> None:
    """One example is emitted per pair, positives and negatives alike."""
    book, chunks = corpus_factory()
    pairs = [build_pair(chunk) for chunk in chunks]

    examples = build_examples(
        book.title, pairs, chunks, distractors=1, negative_ratio=0.5, seed=2
    )

    assert len(examples) == len(pairs)
    assert {example.id for example in examples} == {f"{pair.id}:raft" for pair in pairs}
