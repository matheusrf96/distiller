"""Unit tests for cached node summarization and the fake ``<source>`` branch."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from distiller.config import ThematicSettings
from distiller.llm.fake import FakeLLM
from distiller.thematic import (
    Summarizer,
    build_tree,
    load_summary_cache,
    summary_cache_key,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from distiller.models import BookDocument, Chunk

    CorpusFactory = Callable[..., tuple[BookDocument, list[Chunk]]]


def test_corrupt_summary_cache_is_ignored_with_a_warning(
    corpus_factory: CorpusFactory,
    tmp_path: Path,
    caplog,
) -> None:
    """A truncated summaries.jsonl is ignored and every summary regenerates (AC4)."""
    book, chunks = corpus_factory()
    cache = tmp_path / "summaries.jsonl"
    cache.write_text(
        '{"key": "abc", "node_id": "book:chapter:1", "summary": "trunc',
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="distiller.thematic.summarizer"):
        run = build_tree(
            book,
            chunks,
            FakeLLM(),
            ThematicSettings(window_size=2),
            cache_path=cache,
        )

    assert "Ignoring unreadable summary cache" in caplog.text
    assert run.manifest.generated_summaries == run.manifest.node_count
    assert run.manifest.reused_summaries == 0
    assert all(node.summary for node in run.tree.nodes)


def test_fake_llm_summarizes_source_blocks_deterministically() -> None:
    """The fake answers a ``<source>`` prompt with a stable grounded summary (AC12)."""
    llm = FakeLLM()
    user = (
        "Summarize the text:\n"
        "<source>\nThe lantern cracked during the storm.\n</source>\n"
    )

    first = llm.complete(system="system", user=user)
    second = llm.complete(system="system", user=user)
    other = llm.complete(
        system="system",
        user="Summarize:\n<source>\nThe keeper painted the tower white.\n</source>\n",
    )

    assert first == second
    assert "lantern cracked" in first
    assert "<source>" not in first
    assert other != first
    assert "tower white" in other


def test_summary_cache_key_embeds_model_and_source() -> None:
    """The cache key changes with the model, the node and the source text (AC3)."""
    base = summary_cache_key(
        "book:chapter:1",
        "chapter text",
        "fake",
        max_source_chars=6000,
        max_summary_chars=1200,
    )

    assert base == summary_cache_key(
        "book:chapter:1",
        "chapter text",
        "fake",
        max_source_chars=6000,
        max_summary_chars=1200,
    )
    assert base != summary_cache_key(
        "book:chapter:1",
        "chapter text",
        "other-model",
        max_source_chars=6000,
        max_summary_chars=1200,
    )
    assert base != summary_cache_key(
        "book:chapter:2",
        "chapter text",
        "fake",
        max_source_chars=6000,
        max_summary_chars=1200,
    )
    assert base != summary_cache_key(
        "book:chapter:1",
        "edited chapter text",
        "fake",
        max_source_chars=6000,
        max_summary_chars=1200,
    )


def test_summarizer_rejects_blank_sources_and_unusable_completions(
    tmp_path: Path,
) -> None:
    """Blank sources and unusable completions fail softly, with warnings."""
    cache = tmp_path / "summaries.jsonl"
    blank = Summarizer(FakeLLM(), cache_path=cache)

    assert (
        blank.summarize("n", book_title="B", unit_title="U", source_text="   ") is None
    )

    too_short = Summarizer(FakeLLM(response="short"), cache_path=cache)
    assert (
        too_short.summarize(
            "n", book_title="B", unit_title="U", source_text="A real source."
        )
        is None
    )

    refusal = Summarizer(
        FakeLLM(response="I'm sorry, I cannot help with that."), cache_path=cache
    )
    assert (
        refusal.summarize(
            "n", book_title="B", unit_title="U", source_text="A real source."
        )
        is None
    )


def test_summary_cache_ignores_rows_without_a_summary(tmp_path: Path) -> None:
    """Cache rows missing a key or summary are skipped."""
    cache = tmp_path / "summaries.jsonl"
    cache.write_text(
        '{"key": "abc"}\n{"key": "def", "summary": "ok"}\n', encoding="utf-8"
    )

    assert load_summary_cache(cache) == {"def": "ok"}


def test_summarizer_tolerates_an_unwritable_cache(tmp_path: Path) -> None:
    """A cache write failure is logged and the summary is still returned."""
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not a directory", encoding="utf-8")
    summarizer = Summarizer(FakeLLM(), cache_path=blocker / "summaries.jsonl")

    summary = summarizer.summarize(
        "n", book_title="B", unit_title="U", source_text="A real source text."
    )

    assert summary is not None
