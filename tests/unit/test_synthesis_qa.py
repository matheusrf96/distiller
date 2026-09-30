"""Unit tests for synthetic QA generation (parsing and prompts)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from distiller.llm.fake import FakeLLM
from distiller.synthesis.qa import generate_pairs, parse_pairs

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

    from distiller.models import BookDocument, Chunk

    CorpusFactory = Callable[..., tuple[BookDocument, list[Chunk]]]

QUOTE = "the lantern cracked, and the keeper rowed to the village"


def payload(question: str = "What cracked during the storm?") -> str:
    """A well-formed one-item completion."""
    return json.dumps(
        [{"question": question, "answer": "The lantern cracked.", "quotes": [QUOTE]}]
    )


def test_parse_pairs_accepts_plain_json() -> None:
    """A bare JSON array parses into pair dictionaries (REQ-SQ-002)."""
    pairs = parse_pairs(payload())

    assert len(pairs) == 1
    assert pairs[0]["question"] == "What cracked during the storm?"
    assert pairs[0]["quotes"] == [QUOTE]


def test_parse_pairs_recovers_json_from_prose_and_fences() -> None:
    """JSON wrapped in prose or markdown fences is still recovered (REQ-SQ-004)."""
    wrapped = f"Sure! Here you go:\n```json\n{payload()}\n```\nHope that helps."

    assert len(parse_pairs(wrapped)) == 1


def test_parse_pairs_returns_empty_on_garbage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unparseable completions yield no pairs and a warning (REQ-SQ-004)."""
    with caplog.at_level(logging.WARNING):
        pairs = parse_pairs("not json at all")

    assert pairs == []


def test_parse_pairs_drops_malformed_items() -> None:
    """Items missing question or answer are dropped individually (REQ-SQ-004)."""
    text = json.dumps(
        [
            {"question": "ok?", "answer": "yes", "quotes": []},
            {"answer": "missing question"},
            "nonsense",
        ]
    )

    pairs = parse_pairs(text)

    assert [pair["question"] for pair in pairs] == ["ok?"]


def test_generate_pairs_uses_the_fake_llm_and_records_provenance(
    corpus_factory: CorpusFactory,
) -> None:
    """The offline client produces grounded, distinct pairs (REQ-SQ-013)."""
    book, chunks = corpus_factory()
    pairs = generate_pairs(FakeLLM(), book, chunks[:1], questions_per_chunk=2)

    assert len(pairs) == 2
    assert all(pair.chunk_id == chunks[0].id for pair in pairs)
    assert all(pair.book_id == book.book_id for pair in pairs)
    assert all(pair.chapter == chunks[0].chapter for pair in pairs)
    assert all(pair.model == "fake" for pair in pairs)
    assert len({pair.id for pair in pairs}) == 2
    assert all(pair.quotes for pair in pairs)
    haystack = " ".join(chunks[0].text.split())
    assert all(quote in haystack for pair in pairs for quote in pair.quotes)


def test_generate_pairs_prompt_requests_the_exact_count(
    corpus_factory: CorpusFactory,
) -> None:
    """The prompt asks for exactly N questions per call (REQ-SQ-003)."""
    book, chunks = corpus_factory()
    prompts: list[str] = []
    llm = FakeLLM(response=lambda user: prompts.append(user) or "[]")

    generate_pairs(llm, book, chunks[:1], questions_per_chunk=3)

    assert "exactly 3 question" in prompts[0]
    assert "<chunk>" in prompts[0]


def test_generate_pairs_tolerates_llm_errors_and_bad_json(
    corpus_factory: CorpusFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One failing or unparseable chunk does not abort generation (REQ-SQ-004)."""

    def response(user: str) -> str:
        if "Chapter One" in user:
            raise RuntimeError("teacher down")
        return "garbage"

    book, chunks = corpus_factory()
    with caplog.at_level(logging.WARNING):
        pairs = generate_pairs(
            FakeLLM(response=response), book, chunks, questions_per_chunk=1
        )

    assert pairs == []
    assert len(caplog.records) >= 2
