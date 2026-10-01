"""Unit tests for RAFT -> chat formatting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from distiller.exceptions import TrainingError
from distiller.models import RetrievedChunk, refusal_text
from distiller.rag import build_system_prompt, build_user_prompt
from distiller.training.chat import format_example, prompt_contract_hash

if TYPE_CHECKING:
    from collections.abc import Callable

    from distiller.models import BookDocument, Chunk
    from distiller.synthesis import RaftExample

    CorpusFactory = Callable[..., tuple[BookDocument, list[Chunk]]]
    RaftFactory = Callable[..., RaftExample]


def test_format_example_matches_the_rag_prompt_contract(
    corpus_factory: CorpusFactory, raft_factory: RaftFactory
) -> None:
    """System and user messages come from the shared RAG contract (REQ-TR-001)."""
    book, chunks = corpus_factory()
    example = raft_factory(chunks, distractor_count=2)

    formatted = format_example(
        book.title, example, {chunk.id: chunk for chunk in chunks}
    )

    assert formatted.id == example.id
    assert formatted.book_id == example.book_id
    assert formatted.question == example.question
    assert formatted.answerable is True
    assert [message.role for message in formatted.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert formatted.messages[0].content == build_system_prompt(book.title)
    assert formatted.messages[2].content == example.answer

    expected_contexts = [RetrievedChunk(chunk=chunk, score=0.0) for chunk in chunks[:3]]
    assert formatted.messages[1].content == build_user_prompt(
        example.question, expected_contexts
    )

    user = formatted.messages[1].content
    assert user.index('id="1"') < user.index('id="2"') < user.index('id="3"')
    assert "[1]" in formatted.messages[2].content


def test_format_negative_targets_the_refusal_sentence(
    corpus_factory: CorpusFactory, raft_factory: RaftFactory
) -> None:
    """Negatives keep distractor contexts and target the refusal (REQ-TR-001)."""
    book, chunks = corpus_factory()
    example = raft_factory(chunks, answerable=False)

    formatted = format_example(
        book.title, example, {chunk.id: chunk for chunk in chunks}
    )

    assert formatted.answerable is False
    assert formatted.messages[2].content == refusal_text(book.title)
    assert "couldn't find" in formatted.messages[2].content
    assert "<doc " in formatted.messages[1].content


def test_format_example_renders_context_provenance(
    corpus_factory: CorpusFactory, raft_factory: RaftFactory
) -> None:
    """``<doc>`` blocks carry chapter/section/page provenance (REQ-TR-002)."""
    book, chunks = corpus_factory()
    chunks[0].heading_path = [chunks[0].chapter, "The Tower"]
    chunks[0].page_start = 3
    chunks[0].page_end = 3
    example = raft_factory(chunks, distractor_count=0)

    formatted = format_example(
        book.title, example, {chunk.id: chunk for chunk in chunks}
    )
    user = formatted.messages[1].content

    assert 'chapter="Chapter One"' in user
    assert 'section="The Tower"' in user
    assert 'pages="3"' in user
    assert chunks[0].text in user


def test_format_example_requires_known_chunks(
    corpus_factory: CorpusFactory, raft_factory: RaftFactory
) -> None:
    """An unknown context chunk id fails with the re-index/re-synth message.

    Covers REQ-TR-002.
    """
    book, chunks = corpus_factory()
    example = raft_factory(chunks)

    with pytest.raises(TrainingError) as excinfo:
        format_example(book.title, example, {})

    message = str(excinfo.value)
    assert "distiller index" in message
    assert "distiller synth" in message


def test_prompt_contract_hash_tracks_the_shared_prompt(
    corpus_factory: CorpusFactory,
) -> None:
    """The manifest prompt hash derives from the shared RAG contract (REQ-TR-005)."""
    book, _ = corpus_factory()

    assert prompt_contract_hash(book.title) == prompt_contract_hash(book.title)
    assert prompt_contract_hash(book.title) != prompt_contract_hash("Another Book")
