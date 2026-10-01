"""Unit tests for the per-book artifact layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

from distiller.paths import BookPaths

if TYPE_CHECKING:
    from pathlib import Path


def test_book_paths_cover_the_documented_artifact_layout(tmp_path: Path) -> None:
    """Every artifact accessor points inside the book directory."""
    paths = BookPaths.for_book(tmp_path, "the-book")
    root = tmp_path / "the-book"

    assert paths.golden_yaml == root / "golden.yaml"
    assert paths.thematic_dir == root / "thematic"
    assert paths.thematic_tree_json == root / "thematic" / "tree.json"
    assert paths.thematic_manifest_json == root / "thematic" / "manifest.json"
    assert paths.thematic_summaries_jsonl == root / "thematic" / "summaries.jsonl"
    assert paths.gguf_dir == root / "training" / "gguf"
    assert paths.gguf_file == root / "training" / "gguf" / "model.gguf"
