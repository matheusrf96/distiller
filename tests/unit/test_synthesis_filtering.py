"""Unit tests for deterministic QA filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from distiller.synthesis.filtering import filter_pairs
from distiller.synthesis.qa import QAPair

if TYPE_CHECKING:
    from collections.abc import Callable

    from distiller.models import BookDocument, Chunk

    CorpusFactory = Callable[..., tuple[BookDocument, list[Chunk]]]

GOOD_QUOTE = "the lantern cracked, and the keeper rowed to the village"


def build_pair(
    chunk: Chunk,
    *,
    question: str = "What happened to the lantern during the storm?",
    answer: str = "The lantern cracked and the keeper rowed to the village for a new one.",  # noqa: E501
    quotes: list[str] | None = None,
) -> QAPair:
    """Build a pair targeting the given chunk."""
    return QAPair(
        id=f"{chunk.id}:test",
        book_id=chunk.book_id,
        chunk_id=chunk.id,
        chapter=chunk.chapter,
        heading_path=list(chunk.heading_path),
        question=question,
        answer=answer,
        quotes=[GOOD_QUOTE] if quotes is None else quotes,
        model="stub",
    )


def test_keeps_a_grounded_pair(corpus_factory: CorpusFactory) -> None:
    """A pair with a verified quote survives filtering (REQ-SQ-005)."""
    _, chunks = corpus_factory()
    outcome = filter_pairs(
        [build_pair(chunks[1])], chunks_by_id={c.id: c for c in chunks}
    )

    assert [pair.id for pair in outcome.kept] == [build_pair(chunks[1]).id]
    assert outcome.rejected == []


def test_rejects_a_pair_whose_quotes_are_not_in_the_chunk(
    corpus_factory: CorpusFactory,
) -> None:
    """Fabricated quotes reject the pair with a clear reason (REQ-SQ-005)."""
    _, chunks = corpus_factory()
    pair = build_pair(chunks[1], quotes=["a sentence that is not in the book at all"])

    outcome = filter_pairs([pair], chunks_by_id={c.id: c for c in chunks})

    assert outcome.kept == []
    assert outcome.rejected[0].reason == "no verified quotes"
    assert outcome.rejection_counts() == {"no verified quotes": 1}


def test_drops_failing_quotes_but_keeps_the_pair(corpus_factory: CorpusFactory) -> None:
    """Only the verified quotes survive on a partially grounded pair (REQ-SQ-005)."""
    _, chunks = corpus_factory()
    pair = build_pair(chunks[1], quotes=["not in the book", GOOD_QUOTE])

    outcome = filter_pairs([pair], chunks_by_id={c.id: c for c in chunks})

    assert len(outcome.kept) == 1
    assert outcome.kept[0].quotes == [GOOD_QUOTE]


def test_rejects_pairs_with_only_short_quotes(corpus_factory: CorpusFactory) -> None:
    """Quotes under the minimum length are noise and reject the pair (REQ-SQ-005)."""
    _, chunks = corpus_factory()
    pair = build_pair(chunks[1], quotes=["the lantern"])

    outcome = filter_pairs([pair], chunks_by_id={c.id: c for c in chunks})

    assert outcome.rejected[0].reason == "no verified quotes"


def test_rejects_duplicate_questions(corpus_factory: CorpusFactory) -> None:
    """The second occurrence of a normalized question is rejected (REQ-SQ-005)."""
    _, chunks = corpus_factory()
    first = build_pair(chunks[1])
    duplicate = build_pair(
        chunks[2],
        question="  WHAT   happened to the lantern during the storm?  ",
        quotes=["painted the tower white, a colour the fishermen trusted"],
    )

    outcome = filter_pairs([first, duplicate], chunks_by_id={c.id: c for c in chunks})

    assert len(outcome.kept) == 1
    assert outcome.rejected[0].reason == "duplicate question"


def test_rejects_questions_that_echo_their_answer(
    corpus_factory: CorpusFactory,
) -> None:
    """Near-copies of the answer are not questions (REQ-SQ-005)."""
    _, chunks = corpus_factory()
    answer = "The lantern cracked during the great storm of 1887."
    pair = build_pair(chunks[1], question=answer, answer=answer)

    outcome = filter_pairs([pair], chunks_by_id={c.id: c for c in chunks})

    assert outcome.rejected[0].reason == "question echoes answer"


def test_rejects_pairs_that_are_too_short(corpus_factory: CorpusFactory) -> None:
    """Tiny questions and answers are rejected as too short (REQ-SQ-005)."""
    _, chunks = corpus_factory()
    pair = build_pair(chunks[1], question="Why?", answer="Because.")

    outcome = filter_pairs([pair], chunks_by_id={c.id: c for c in chunks})

    assert outcome.rejected[0].reason == "too short"
