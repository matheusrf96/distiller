"""Unit tests for the thematic summary tree (shape, cache, failure, manifest)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from distiller.config import ThematicSettings
from distiller.exceptions import ThematicError
from distiller.llm.fake import FakeLLM
from distiller.thematic import (
    SummaryTree,
    build_tree,
    chapter_node_id,
    load_manifest,
    load_tree,
    root_node_id,
    window_node_id,
)
from distiller.utils import stable_hash_hex, write_json

if TYPE_CHECKING:
    from collections.abc import Callable

    from distiller.models import BookDocument, Chunk

    CorpusFactory = Callable[..., tuple[BookDocument, list[Chunk]]]


def source_summary(user: str) -> str:
    """Deterministic, source-dependent fake summary for cache tests."""
    source = user.split("<source>\n", 1)[1].split("\n</source>", 1)[0]
    return f"Summary of: {' '.join(source.split())}"


def test_build_tree_summarizes_chapters_windows_and_root(
    corpus_factory: CorpusFactory,
) -> None:
    """Three chapters at window_size=2 yield three chapters, two windows, root (AC1)."""
    book, chunks = corpus_factory()

    run = build_tree(book, chunks, FakeLLM(), ThematicSettings(window_size=2))
    tree = run.tree

    assert tree.book_id == book.book_id
    assert tree.root_id == root_node_id(book.book_id)
    chapters = [node for node in tree.nodes if node.level == 1]
    windows = [node for node in tree.nodes if node.level == 2]
    assert [node.id for node in chapters] == [
        chapter_node_id(book.book_id, index) for index in range(1, 4)
    ]
    assert [node.id for node in windows] == [
        window_node_id(book.book_id, 1, 2),
        window_node_id(book.book_id, 3, 3),
    ]
    assert [node.title for node in windows] == ["Chapters 1-2", "Chapters 3-3"]
    assert all(node.summary for node in tree.nodes)
    assert tree.root.level == 3
    assert tree.root.children == [node.id for node in windows]
    assert run.manifest.chapter_count == 3
    assert run.manifest.window_count == 2
    assert run.manifest.node_count == len(tree.nodes) == 6


def test_build_tree_skips_windows_for_short_books(
    corpus_factory: CorpusFactory,
) -> None:
    """A book with no more chapters than window_size has no level-2 nodes (AC1)."""
    book, chunks = corpus_factory()

    run = build_tree(book, chunks, FakeLLM(), ThematicSettings(window_size=4))

    assert not [node for node in run.tree.nodes if node.level == 2]
    assert run.manifest.window_count == 0
    assert run.manifest.node_count == 4
    assert run.tree.root.children == [
        node.id for node in run.tree.nodes if node.level == 1
    ]
    assert all(node.summary for node in run.tree.nodes)


def test_node_ids_are_deterministic_and_cover_chunks(
    corpus_factory: CorpusFactory,
    tmp_path: Path,
) -> None:
    """Ids survive rebuilds; levels, children and chunk coverage are exact (AC2)."""
    book, chunks = corpus_factory()
    settings = ThematicSettings(window_size=2)
    cache = tmp_path / "summaries.jsonl"

    first = build_tree(book, chunks, FakeLLM(), settings, cache_path=cache)
    second = build_tree(book, chunks, FakeLLM(), settings, cache_path=cache)

    assert [node.id for node in first.tree.nodes] == [
        node.id for node in second.tree.nodes
    ]
    assert [node.summary for node in first.tree.nodes] == [
        node.summary for node in second.tree.nodes
    ]

    chapters = [node for node in first.tree.nodes if node.level == 1]
    for index, node in enumerate(chapters, start=1):
        assert node.id == chapter_node_id(book.book_id, index)
        assert node.chunk_ids == [
            chunk.id for chunk in chunks if chunk.chapter == node.title
        ]

    windows = [node for node in first.tree.nodes if node.level == 2]
    assert windows[0].children == [chapters[0].id, chapters[1].id]
    assert windows[1].children == [chapters[2].id]
    assert first.tree.root.chunk_ids == [chunk.id for chunk in chunks]


def test_rebuild_reuses_cached_summaries_without_calling_the_llm(
    corpus_factory: CorpusFactory,
    tmp_path: Path,
) -> None:
    """A second build with an LLM that would raise makes zero calls (AC3)."""
    book, chunks = corpus_factory()
    settings = ThematicSettings(window_size=2)
    cache = tmp_path / "summaries.jsonl"
    build_tree(book, chunks, FakeLLM(), settings, cache_path=cache)

    def explode(user: str) -> str:
        pytest.fail(f"unexpected LLM call: {user[:60]}")

    run = build_tree(
        book, chunks, FakeLLM(response=explode), settings, cache_path=cache
    )

    assert run.manifest.generated_summaries == 0
    assert run.manifest.reused_summaries == run.manifest.node_count
    assert run.manifest.failed_node_ids == []
    assert all(node.summary for node in run.tree.nodes)


def test_regenerate_forces_new_llm_calls(
    corpus_factory: CorpusFactory,
    tmp_path: Path,
) -> None:
    """--regenerate calls the LLM for every node and reuses nothing (AC3)."""
    book, chunks = corpus_factory()
    settings = ThematicSettings(window_size=2)
    cache = tmp_path / "summaries.jsonl"
    calls: list[str] = []

    def respond(user: str) -> str:
        calls.append(user)
        return "Regenerated summary text."

    first = build_tree(
        book, chunks, FakeLLM(response=respond), settings, cache_path=cache
    )
    assert first.manifest.generated_summaries == first.manifest.node_count
    assert len(calls) == first.manifest.node_count

    second = build_tree(
        book,
        chunks,
        FakeLLM(response=respond),
        settings,
        cache_path=cache,
        regenerate=True,
    )

    assert second.manifest.generated_summaries == second.manifest.node_count
    assert second.manifest.reused_summaries == 0
    assert len(calls) == 2 * second.manifest.node_count


def test_failed_node_is_skipped_and_retried_on_rebuild(
    corpus_factory: CorpusFactory,
    tmp_path: Path,
    caplog,
) -> None:
    """One failing chapter is skipped with a warning and retried later (AC5)."""
    book, chunks = corpus_factory()
    settings = ThematicSettings(window_size=2)
    cache = tmp_path / "summaries.jsonl"

    def flaky(user: str) -> str:
        if "great storm" in user:
            raise RuntimeError("flaky endpoint")
        return source_summary(user)

    with caplog.at_level(logging.WARNING, logger="distiller.thematic.summarizer"):
        first = build_tree(
            book, chunks, FakeLLM(response=flaky), settings, cache_path=cache
        )

    failed_id = chapter_node_id(book.book_id, 2)
    assert first.manifest.failed_node_ids == [failed_id]
    assert first.manifest.generated_summaries == 5
    chapter_two = next(node for node in first.tree.nodes if node.id == failed_id)
    assert chapter_two.summary is None
    assert "flaky endpoint" in caplog.text

    second = build_tree(
        book,
        chunks,
        FakeLLM(response=source_summary),
        settings,
        cache_path=cache,
    )

    assert second.manifest.failed_node_ids == []
    assert second.manifest.generated_summaries == 3
    assert second.manifest.reused_summaries == 3
    assert all(node.summary for node in second.tree.nodes)


def test_manifest_records_provenance_and_counts(
    corpus_factory: CorpusFactory,
    tmp_path: Path,
) -> None:
    """The manifest carries model, config, hashes, counts and failed ids (AC13)."""
    book, chunks = corpus_factory()
    settings = ThematicSettings(window_size=2, map_top_k=3)

    run = build_tree(
        book,
        chunks,
        FakeLLM(),
        settings,
        cache_path=tmp_path / "summaries.jsonl",
    )
    manifest = run.manifest

    assert manifest.book_id == book.book_id
    assert manifest.model == "fake"
    assert manifest.config["window_size"] == 2
    assert manifest.config["map_top_k"] == 3
    assert manifest.source_hash == stable_hash_hex(
        "\n".join(chunk.id for chunk in chunks)
    )
    assert manifest.tree_hash
    assert manifest.chapter_count == 3
    assert manifest.window_count == 2
    assert manifest.node_count == 6
    assert manifest.generated_summaries == 6
    assert manifest.reused_summaries == 0
    assert manifest.failed_node_ids == []


def test_tree_and_manifest_roundtrip(
    corpus_factory: CorpusFactory,
    tmp_path: Path,
) -> None:
    """Written tree.json and manifest.json load back as validated models."""
    book, chunks = corpus_factory()
    run = build_tree(book, chunks, FakeLLM(), ThematicSettings())
    tree_path = tmp_path / "tree.json"
    manifest_path = tmp_path / "manifest.json"
    write_json(tree_path, run.tree.model_dump())
    write_json(manifest_path, run.manifest.model_dump())

    assert load_tree(tree_path) == run.tree
    assert load_manifest(manifest_path) == run.manifest

    with pytest.raises(ThematicError, match="distiller tree build"):
        load_tree(tmp_path / "missing-tree.json")
    manifest_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ThematicError, match="distiller tree build"):
        load_manifest(manifest_path)


def test_tree_without_root_node_is_rejected(corpus_factory: CorpusFactory) -> None:
    """A tree whose root_id is absent fails validation instead of raising later."""
    book, chunks = corpus_factory()
    run = build_tree(book, chunks, FakeLLM(), ThematicSettings())

    with pytest.raises(ValueError, match="root_id"):
        SummaryTree(
            book_id=book.book_id,
            root_id="missing",
            nodes=run.tree.nodes,
        )


def test_thematic_questions_doc_documents_the_workflow() -> None:
    """The thematic doc names the commands and the golden-item shape (AC14)."""
    root = Path(__file__).resolve().parents[2]
    doc = (root / "docs" / "thematic-questions.md").read_text(encoding="utf-8")
    index = (root / "docs" / "README.md").read_text(encoding="utf-8")

    assert "distiller tree build" in doc
    assert "ask --global" in doc
    assert "eval --global" in doc
    assert "expected_answer_contains" in doc
    assert "thematic-questions.md" in index
