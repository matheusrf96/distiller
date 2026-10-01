"""Unit tests for contextual enrichment (contextual retrieval)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import pytest

from distiller.enrichment.contextual import ContextualEnricher
from distiller.models import Block, BookDocument, Chapter, Chunk

if TYPE_CHECKING:
    from pathlib import Path


class StubLLM:
    """Records prompts and answers with a canned or callable response."""

    def __init__(self, response: str | Any = "situating context") -> None:
        self._response = response
        self.prompts: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        """Stable identifier for the stub."""
        return "stub"

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.prompts.append((system, user))
        if callable(self._response):
            return str(self._response(user))
        return self._response


def build_book() -> BookDocument:
    """Two-chapter book with enough text to excerpt."""
    return BookDocument(
        book_id="contextual",
        title="The Lantern Keeper",
        source_path="lantern.epub",
        source_format="epub",
        chapters=[
            Chapter(
                title="Chapter One",
                blocks=[
                    Block(type="heading", text="Chapter One", level=1),
                    Block(
                        type="paragraph",
                        text="The keeper counted the steps every morning. " * 20,
                    ),
                ],
            ),
            Chapter(
                title="Chapter Two",
                blocks=[
                    Block(type="heading", text="Chapter Two", level=1),
                    Block(
                        type="paragraph",
                        text="The storm cracked the lantern in 1887. " * 20,
                    ),
                ],
            ),
        ],
    )


def build_chunks(book: BookDocument) -> list[Chunk]:
    """One chunk per chapter, with distinct content."""
    chunks = []
    for ordinal, chapter in enumerate(book.chapters):
        text = " ".join(block.text for block in chapter.blocks)
        chunks.append(
            Chunk(
                id=Chunk.make_id(book.book_id, ordinal, text),
                book_id=book.book_id,
                ordinal=ordinal,
                text=text,
                chapter=chapter.title,
                heading_path=[chapter.title],
                char_count=len(text),
            )
        )
    return chunks


def test_index_text_appends_context_only_when_present() -> None:
    """index_text equals text without a context and prefixes it with one."""
    chunk = Chunk(
        id="b:0000:x",
        book_id="b",
        ordinal=0,
        text="The lantern cracked.",
        chapter="C1",
        heading_path=["C1"],
    )

    assert chunk.index_text == "The lantern cracked."
    contextual = chunk.model_copy(update={"context": "A storm scene."})
    assert contextual.index_text == "A storm scene.\n\nThe lantern cracked."
    assert contextual.text == "The lantern cracked."  # original untouched


def test_enricher_generates_and_persists_contexts(tmp_path: Path) -> None:
    """Every chunk gets a context and the cache file records them (REQ-CR-001/002)."""
    book = build_book()
    chunks = build_chunks(book)
    llm = StubLLM(response="A keeper scene in the tower.")
    cache = tmp_path / "enrichment.jsonl"

    enriched = ContextualEnricher(llm, book, cache_path=cache).enrich(chunks)

    assert all(chunk.context == "A keeper scene in the tower." for chunk in enriched)
    assert len(llm.prompts) == len(chunks)
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    assert {row["chunk_id"] for row in rows} == {chunk.id for chunk in chunks}


def test_enricher_tolerates_an_unwritable_cache(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A cache write failure is logged and the run still returns contexts."""
    book = build_book()
    chunks = build_chunks(book)
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not a directory", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        enriched = ContextualEnricher(
            StubLLM(response="A storm scene."),
            book,
            cache_path=blocker / "enrichment.jsonl",
        ).enrich(chunks)

    assert all(chunk.context == "A storm scene." for chunk in enriched)
    assert "Could not write enrichment cache" in caplog.text


def test_enricher_reuses_cache_without_calling_the_llm(tmp_path: Path) -> None:
    """A second run over the same chunks performs no LLM calls (REQ-CR-003)."""
    book = build_book()
    chunks = build_chunks(book)
    cache = tmp_path / "enrichment.jsonl"

    ContextualEnricher(StubLLM(), book, cache_path=cache).enrich(chunks)
    second_llm = StubLLM()
    enriched = ContextualEnricher(second_llm, book, cache_path=cache).enrich(chunks)

    assert second_llm.prompts == []
    assert all(chunk.context for chunk in enriched)


def test_enricher_tolerates_llm_failures(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One failing chunk does not abort the run (REQ-CR-004)."""
    book = build_book()
    chunks = build_chunks(book)
    failing_id = chunks[0].id

    def response(user: str) -> str:
        if failing_id in user or chunks[0].text[:40] in user:
            raise RuntimeError("backend down")
        return "A storm scene."

    llm = StubLLM(response=response)
    with caplog.at_level(logging.WARNING):
        enriched = ContextualEnricher(
            llm, book, cache_path=tmp_path / "c.jsonl"
        ).enrich(chunks)

    assert enriched[0].context is None
    assert enriched[1].context == "A storm scene."
    assert any("failed" in record.message.lower() for record in caplog.records)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  too   much\nwhitespace ", "too much whitespace"),
        ("ok", None),  # below the minimum length
        ("I couldn't find that in The Book.", None),  # refusal-like
        ("   ", None),  # empty
    ],
)
def test_enricher_normalizes_and_discards(
    raw: str, expected: str | None, tmp_path: Path
) -> None:
    """Whitespace is collapsed; unusable completions are discarded (REQ-CR-009)."""
    book = build_book()
    chunk = build_chunks(book)[:1]
    llm = StubLLM(response=raw)

    enriched = ContextualEnricher(llm, book, cache_path=tmp_path / "c.jsonl").enrich(
        chunk
    )

    assert enriched[0].context == expected


def test_enricher_caps_overlong_contexts(tmp_path: Path) -> None:
    """Contexts are truncated to the configured cap (REQ-CR-009/CON-CR-005)."""
    book = build_book()
    chunk = build_chunks(book)[:1]
    llm = StubLLM(response="word " * 400)

    enriched = ContextualEnricher(
        llm, book, cache_path=tmp_path / "c.jsonl", max_context_chars=50
    ).enrich(chunk)

    assert enriched[0].context is not None
    assert len(enriched[0].context) <= 50


def test_prompt_carries_book_chapter_and_section(tmp_path: Path) -> None:
    """The prompt includes the locator fields (REQ-CR-010)."""
    book = build_book()
    chunk = build_chunks(book)[:1]
    llm = StubLLM()

    ContextualEnricher(llm, book, cache_path=tmp_path / "c.jsonl").enrich(chunk)

    _, user = llm.prompts[0]
    assert "The Lantern Keeper" in user
    assert "Chapter One" in user
    assert chunk[0].text[:60] in user


def test_enricher_survives_a_corrupt_cache(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt cache file is ignored with a warning and regenerated."""
    cache = tmp_path / "enrichment.jsonl"
    cache.write_text("{not json\n", encoding="utf-8")
    book = build_book()
    chunks = build_chunks(book)
    llm = StubLLM(response="A keeper scene in the tower.")

    with caplog.at_level(logging.WARNING):
        enriched = ContextualEnricher(llm, book, cache_path=cache).enrich(chunks)

    assert all(chunk.context for chunk in enriched)
    assert len(llm.prompts) == len(chunks)
    assert any("cache" in record.message.lower() for record in caplog.records)


def test_enricher_without_cache_path_still_works() -> None:
    """Cache is optional; enrichment works in-memory."""
    book = build_book()
    chunks = build_chunks(book)
    llm = StubLLM(response="A keeper scene in the tower.")

    enriched = ContextualEnricher(llm, book).enrich(chunks)

    assert all(chunk.context for chunk in enriched)
