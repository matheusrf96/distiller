"""Unit tests for synthesis orchestration (sampling, manifest, reuse)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from distiller.config import SynthesisSettings
from distiller.llm.fake import FakeLLM
from distiller.synthesis.dataset import load_cached_pairs, sample_chunks, synthesize

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest

    from distiller.models import BookDocument, Chunk

    CorpusFactory = Callable[..., tuple[BookDocument, list[Chunk]]]


def many_chapters() -> list[tuple[str, str]]:
    """Six short chapters for sampling tests."""
    return [
        (
            f"Chapter {index}",
            f"Passage number {index} about the sea and the wind. " * 10,
        )
        for index in range(1, 7)
    ]


def test_sample_chunks_is_deterministic_and_in_reading_order(
    corpus_factory: CorpusFactory,
) -> None:
    """Sampling is seeded and returned in reading order (REQ-SQ-011)."""
    _, chunks = corpus_factory(many_chapters())

    first = sample_chunks(chunks, max_chunks=4, seed=11)
    second = sample_chunks(chunks, max_chunks=4, seed=11)

    assert len(first) == 4
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert [chunk.ordinal for chunk in first] == sorted(
        chunk.ordinal for chunk in first
    )


def test_sample_chunks_returns_everything_when_cap_exceeds_corpus(
    corpus_factory: CorpusFactory,
) -> None:
    """A cap above the corpus size keeps all chunks."""
    _, chunks = corpus_factory()

    assert len(sample_chunks(chunks, max_chunks=99, seed=1)) == len(chunks)


def test_synthesize_generates_filters_and_formats(
    corpus_factory: CorpusFactory,
) -> None:
    """The full offline pipeline produces consistent counts (REQ-SQ-001/009/013)."""
    book, chunks = corpus_factory()
    settings = SynthesisSettings(
        max_chunks=2,
        questions_per_chunk=2,
        distractors=1,
        negative_ratio=0.0,
        seed=5,
    )

    run = synthesize(book, chunks, FakeLLM(), settings)

    assert run.manifest.book_id == book.book_id
    assert run.manifest.source == "generated"
    assert run.manifest.model == "fake"
    assert run.manifest.chunk_count == len(chunks)
    assert run.manifest.sampled_chunks == 2
    assert run.manifest.generated_pairs == 4
    assert run.manifest.kept_pairs == len(run.outcome.kept) == 4
    assert run.manifest.rejected == {}
    assert run.manifest.example_count == len(run.examples) == 4
    assert run.manifest.answerable_examples == 4
    assert run.manifest.unanswerable_examples == 0


def test_synthesize_reuses_cached_pairs_without_calling_the_llm(
    corpus_factory: CorpusFactory,
) -> None:
    """Cached pairs skip generation entirely (REQ-SQ-010)."""
    book, chunks = corpus_factory()
    settings = SynthesisSettings(
        max_chunks=1,
        questions_per_chunk=1,
        distractors=1,
        negative_ratio=0.0,
        seed=5,
    )
    first = synthesize(book, chunks, FakeLLM(), settings)

    class ExplodingLLM:
        @property
        def name(self) -> str:
            return "exploding"

        def complete(self, **kwargs: Any) -> str:
            raise AssertionError("the LLM must not be called when pairs are cached")

    second = synthesize(
        book, chunks, ExplodingLLM(), settings, cached_pairs=first.pairs
    )

    assert second.manifest.source == "cache"
    assert second.manifest.model == first.manifest.model  # provenance from the cache
    assert second.manifest.generated_pairs == len(first.pairs)
    assert [example.id for example in second.examples] == [
        example.id for example in first.examples
    ]


def test_manifest_records_index_identity_and_config(
    corpus_factory: CorpusFactory,
) -> None:
    """The manifest carries provenance for later comparison (REQ-SQ-009)."""
    book, chunks = corpus_factory()
    settings = SynthesisSettings(max_chunks=1, questions_per_chunk=1, seed=9)

    run = synthesize(
        book,
        chunks,
        FakeLLM(),
        settings,
        index_identity={"embedder": "hash:64", "contextual": True},
    )

    assert run.manifest.index == {"embedder": "hash:64", "contextual": True}
    assert run.manifest.config["questions_per_chunk"] == 1
    assert run.manifest.config["seed"] == 9


def test_load_cached_pairs_ignores_corrupt_caches(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt QA cache is ignored with a warning and regenerates."""
    cache = tmp_path / "qa.jsonl"
    cache.write_text("{not json\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert load_cached_pairs(cache) is None

    assert "Ignoring unreadable QA cache" in caplog.text
