"""Thematic summary tree: models and the hierarchical build.

Level 1 nodes summarize chapters, level 2 nodes summarize windows of
``window_size`` consecutive chapter summaries and the level 3 root summarizes
its children (window summaries when level 2 exists, chapter summaries
otherwise). Chapter text is the only raw input; higher levels only ever see
already-distilled summaries, so the build costs ``#chapters + #windows + 1``
LLM calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import Field, model_validator

from ..exceptions import ThematicError
from ..models import DomainModel
from ..utils import read_json, stable_hash_hex
from .summarizer import Summarizer

if TYPE_CHECKING:
    from pathlib import Path

    from ..config import ThematicSettings
    from ..llm.base import LLMClient
    from ..models import BookDocument, Chapter, Chunk

logger = logging.getLogger(__name__)

__all__ = [
    "SummaryNode",
    "SummaryTree",
    "TreeManifest",
    "TreeRun",
    "build_tree",
    "chapter_node_id",
    "load_manifest",
    "load_tree",
    "root_node_id",
    "source_hash",
    "tree_hash",
    "tree_identity",
    "window_node_id",
]


class SummaryNode(DomainModel):
    """One node of the thematic summary tree.

    Attributes:
        id: Deterministic node id (chapter, window or root).
        level: 1 for chapters, 2 for windows, 3 for the root.
        title: Human-readable title (chapter title, ``Chapters a-b``, book title).
        summary: Generated summary; None when the node failed.
        children: Child node ids (empty for chapter nodes).
        chunk_ids: Chunk ids the node covers.
    """

    id: str
    level: int
    title: str
    summary: str | None = None
    children: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class SummaryTree(DomainModel):
    """A book's hierarchical summary tree.

    Attributes:
        book_id: Book the tree was built from.
        root_id: Id of the whole-book node.
        nodes: All nodes in reading order (chapters, windows, root).
    """

    book_id: str
    root_id: str
    nodes: list[SummaryNode] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_root(self) -> SummaryTree:
        if not any(node.id == self.root_id for node in self.nodes):
            raise ValueError(f"root_id {self.root_id!r} is not among the tree nodes")
        return self

    @property
    def root(self) -> SummaryNode:
        """The whole-book node."""
        for node in self.nodes:
            if node.id == self.root_id:
                return node
        raise ValueError(f"Summary tree for '{self.book_id}' has no root node")


class TreeManifest(DomainModel):
    """Provenance and counts for one tree build.

    Attributes:
        book_id: Book the tree was built from.
        model: LLM that produced the summaries.
        config: Thematic settings used for the build (window size, budgets).
        source_hash: Hash of the ordered chunk ids the tree was built from.
        tree_hash: Hash of the ordered node ids and summaries.
        chapter_count: Level-1 nodes.
        window_count: Level-2 nodes.
        node_count: All nodes.
        generated_summaries: Summaries produced in this run.
        reused_summaries: Summaries reused from the cache.
        failed_node_ids: Nodes whose summary failed and will be retried.
    """

    book_id: str
    model: str
    config: dict[str, Any] = Field(default_factory=dict)
    source_hash: str
    tree_hash: str
    chapter_count: int = 0
    window_count: int = 0
    node_count: int = 0
    generated_summaries: int = 0
    reused_summaries: int = 0
    failed_node_ids: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TreeRun:
    """Everything one tree build produced (runtime container).

    Attributes:
        tree: The built summary tree.
        manifest: Provenance and counts for the build.
    """

    tree: SummaryTree
    manifest: TreeManifest


def chapter_node_id(book_id: str, index: int) -> str:
    """Deterministic id of the chapter node at 1-based ``index``."""
    return f"{book_id}:chapter:{index}"


def window_node_id(book_id: str, start: int, end: int) -> str:
    """Deterministic id of the window node covering chapters ``start``-``end``."""
    return f"{book_id}:window:{start}-{end}"


def root_node_id(book_id: str) -> str:
    """Deterministic id of the whole-book node."""
    return f"{book_id}:root"


def source_hash(chunks: list[Chunk]) -> str:
    """Hash the ordered chunk ids the tree was built from.

    Chunk ids embed content hashes, so edits and re-chunking invalidate the
    tree's source identity.
    """
    return stable_hash_hex("\n".join(chunk.id for chunk in chunks))


def tree_hash(tree: SummaryTree) -> str:
    """Hash the ordered node ids and summaries of a tree."""
    payload = "\n".join(f"{node.id}\x1f{node.summary or ''}" for node in tree.nodes)
    return stable_hash_hex(payload)


def tree_identity(manifest: TreeManifest) -> dict[str, Any]:
    """Build the JSON-ready tree identity recorded in eval reports."""
    return {
        "hash": manifest.tree_hash,
        "model": manifest.model,
        "chapter_count": manifest.chapter_count,
        "window_count": manifest.window_count,
        "node_count": manifest.node_count,
    }


def build_tree(
    book: BookDocument,
    chunks: list[Chunk],
    llm: LLMClient,
    settings: ThematicSettings,
    *,
    cache_path: Path | None = None,
    regenerate: bool = False,
) -> TreeRun:
    """Summarize chapters, windows and the whole book into a tree.

    Args:
        book: Parsed book (chapter text is the only raw input).
        chunks: Indexed chunks of the book.
        llm: Chat client used for summarization.
        settings: Thematic settings (window size and character budgets).
        cache_path: Optional ``summaries.jsonl`` cache (read and appended).
        regenerate: Ignore cached summaries and call the LLM again.

    Returns:
        The tree plus its manifest. Failed nodes keep ``summary=None`` and are
        recorded in the manifest for a later retry.

    Raises:
        ThematicError: When the book has no chunks or chapters, or when every
            chapter summary fails (nothing is written in that case).
    """
    if not chunks:
        raise ThematicError(
            f"Book '{book.book_id}' has no chunks. "
            f"Run `distiller index {book.book_id}` first."
        )
    if not book.chapters:
        raise ThematicError(
            f"Book '{book.book_id}' has no chapters; nothing to summarize."
        )

    summarizer = Summarizer(
        llm,
        cache_path=cache_path,
        max_source_chars=settings.max_source_chars,
        max_summary_chars=settings.max_summary_chars,
    )
    chunk_ids_by_chapter = _chunk_ids_by_chapter(chunks)

    chapter_nodes, failed = _build_chapter_nodes(
        book, chunk_ids_by_chapter, summarizer, regenerate=regenerate
    )
    if all(node.summary is None for node in chapter_nodes):
        raise ThematicError(
            f"Summarizing every chapter of '{book.book_id}' failed "
            f"with model '{llm.name}'."
        )

    window_nodes, window_failed = _build_window_nodes(
        book, chapter_nodes, settings.window_size, summarizer, regenerate=regenerate
    )
    failed.extend(window_failed)

    children = window_nodes or chapter_nodes
    root, root_failed = _build_root_node(
        book, children, chunks, summarizer, regenerate=regenerate
    )
    failed.extend(root_failed)

    nodes = [*chapter_nodes, *window_nodes, root]
    tree = SummaryTree(book_id=book.book_id, root_id=root.id, nodes=nodes)
    manifest = TreeManifest(
        book_id=book.book_id,
        model=llm.name,
        config=settings.model_dump(),
        source_hash=source_hash(chunks),
        tree_hash=tree_hash(tree),
        chapter_count=len(chapter_nodes),
        window_count=len(window_nodes),
        node_count=len(nodes),
        generated_summaries=summarizer.generated,
        reused_summaries=summarizer.reused,
        failed_node_ids=failed,
    )
    return TreeRun(tree=tree, manifest=manifest)


def load_tree(path: Path) -> SummaryTree:
    """Load ``thematic/tree.json``, or raise a friendly error.

    Args:
        path: Tree file written by ``distiller tree build``.

    Returns:
        The validated summary tree.

    Raises:
        ThematicError: When the file is missing or corrupt; the message names
            the ``distiller tree build`` command.
    """
    if not path.exists():
        raise ThematicError(
            f"No summary tree at {path}. Run `distiller tree build` first."
        )
    try:
        return SummaryTree.model_validate(read_json(path))
    except (OSError, ValueError) as exc:
        raise ThematicError(
            f"Summary tree {path} is unreadable or corrupt: {exc}. "
            f"Re-run `distiller tree build`."
        ) from exc


def load_manifest(path: Path) -> TreeManifest:
    """Load ``thematic/manifest.json``, or raise a friendly error.

    Args:
        path: Manifest file written by ``distiller tree build``.

    Returns:
        The validated tree manifest.

    Raises:
        ThematicError: When the file is missing or corrupt; the message names
            the ``distiller tree build`` command.
    """
    if not path.exists():
        raise ThematicError(
            f"No tree manifest at {path}. Run `distiller tree build` first."
        )
    try:
        return TreeManifest.model_validate(read_json(path))
    except (OSError, ValueError) as exc:
        raise ThematicError(
            f"Tree manifest {path} is unreadable or corrupt: {exc}. "
            f"Re-run `distiller tree build`."
        ) from exc


def _build_chapter_nodes(
    book: BookDocument,
    chunk_ids_by_chapter: dict[str, list[str]],
    summarizer: Summarizer,
    *,
    regenerate: bool,
) -> tuple[list[SummaryNode], list[str]]:
    """Summarize every chapter (level 1) and collect failed node ids."""
    nodes: list[SummaryNode] = []
    failed: list[str] = []
    for index, chapter in enumerate(book.chapters, start=1):
        node_id = chapter_node_id(book.book_id, index)
        summary = summarizer.summarize(
            node_id,
            book_title=book.title,
            unit_title=chapter.title,
            source_text=_chapter_text(chapter),
            regenerate=regenerate,
        )
        if summary is None:
            failed.append(node_id)
        nodes.append(
            SummaryNode(
                id=node_id,
                level=1,
                title=chapter.title,
                summary=summary,
                chunk_ids=chunk_ids_by_chapter.get(chapter.title, []),
            )
        )
    return nodes, failed


def _build_window_nodes(
    book: BookDocument,
    chapter_nodes: list[SummaryNode],
    window_size: int,
    summarizer: Summarizer,
    *,
    regenerate: bool,
) -> tuple[list[SummaryNode], list[str]]:
    """Summarize windows of consecutive chapters (level 2).

    Windows exist only when the book has more chapters than ``window_size``; a
    trailing window may be shorter. Failed children are excluded from the
    window's input.
    """
    nodes: list[SummaryNode] = []
    failed: list[str] = []
    if len(book.chapters) <= window_size:
        return nodes, failed

    for start in range(0, len(chapter_nodes), window_size):
        group = chapter_nodes[start : start + window_size]
        start_index = start + 1
        end_index = start + len(group)
        node_id = window_node_id(book.book_id, start_index, end_index)
        title = f"Chapters {start_index}-{end_index}"
        summary = summarizer.summarize(
            node_id,
            book_title=book.title,
            unit_title=title,
            source_text=_summaries_text(group),
            regenerate=regenerate,
        )
        if summary is None:
            failed.append(node_id)
        nodes.append(
            SummaryNode(
                id=node_id,
                level=2,
                title=title,
                summary=summary,
                children=[node.id for node in group],
                chunk_ids=[chunk_id for node in group for chunk_id in node.chunk_ids],
            )
        )
    return nodes, failed


def _build_root_node(
    book: BookDocument,
    children: list[SummaryNode],
    chunks: list[Chunk],
    summarizer: Summarizer,
    *,
    regenerate: bool,
) -> tuple[SummaryNode, list[str]]:
    """Summarize the whole book (level 3); failed children are excluded."""
    node_id = root_node_id(book.book_id)
    summary = summarizer.summarize(
        node_id,
        book_title=book.title,
        unit_title=book.title,
        source_text=_summaries_text(children),
        regenerate=regenerate,
    )
    failed = [node_id] if summary is None else []
    root = SummaryNode(
        id=node_id,
        level=3,
        title=book.title,
        summary=summary,
        children=[node.id for node in children],
        chunk_ids=[chunk.id for chunk in chunks],
    )
    return root, failed


def _chunk_ids_by_chapter(chunks: list[Chunk]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.chapter, []).append(chunk.id)
    return grouped


def _chapter_text(chapter: Chapter) -> str:
    return "\n\n".join(block.text for block in chapter.blocks)


def _summaries_text(nodes: list[SummaryNode]) -> str:
    return "\n\n".join(node.summary for node in nodes if node.summary)
