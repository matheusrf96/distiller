"""Shared pytest fixtures.

Everything here generates books locally: no network, no binary fixtures, no
model downloads. The offline pipeline uses the hashing embedder and the fake LLM.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path

import pytest

from distiller.config import Settings, load_settings
from distiller.models import Block, BookDocument, Chapter, Chunk, refusal_text
from distiller.synthesis import RaftContext, RaftExample
from distiller.thematic import (
    SummaryNode,
    SummaryTree,
    chapter_node_id,
    root_node_id,
    window_node_id,
)
from distiller.utils import wrap_text

FIXTURE_CHAPTERS: list[tuple[str, str]] = [
    (
        "Chapter One",
        "The lighthouse keeper counted three hundred steps "
        "to the top of the tower every morning.",
    ),
    (
        "Chapter Two",
        "During the great storm of 1887 the lantern cracked, "
        "and the keeper rowed to the village for a new one.",
    ),
    (
        "Chapter Three",
        "The keeper returned in the spring and painted the tower white, "
        "a colour the fishermen trusted.",
    ),
]

EpubFactory = Callable[..., Path]
PdfFactory = Callable[..., Path]
CorpusFactory = Callable[..., "tuple[BookDocument, list[Chunk]]"]
RaftFactory = Callable[..., RaftExample]
AdapterFactory = Callable[..., Path]
GgufFactory = Callable[..., Path]
TreeFactory = Callable[..., SummaryTree]

# GGUF metadata value types used by the synthetic files (spec v2/v3).
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9


@pytest.fixture()
def offline_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the pipeline at a temp artifacts dir and offline backends.

    Returns:
        The temporary artifacts directory.
    """
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("DISTILLER_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("DISTILLER_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("DISTILLER_LLM__MODEL", "fake")
    return artifacts


@pytest.fixture()
def settings(offline_env: Path) -> Settings:
    """Settings wired to the offline environment."""
    return load_settings()


@pytest.fixture()
def epub_factory() -> EpubFactory:
    """Factory fixture building small EPUB files with EbookLib."""

    def _make(path: Path, chapters: list[tuple[str, str]] | None = None) -> Path:
        from ebooklib import epub

        book = epub.EpubBook()
        book.set_identifier("fixture-lantern-keeper")
        book.set_title("The Lantern Keeper")
        book.set_language("en")
        book.add_author("Ada Fixture")

        items = []
        for index, (title, text) in enumerate(chapters or FIXTURE_CHAPTERS, start=1):
            item = epub.EpubHtml(
                title=title, file_name=f"chap_{index}.xhtml", lang="en"
            )
            item.content = f"<h1>{title}</h1><h2>Section {index}</h2><p>{text}</p>"
            book.add_item(item)
            items.append(item)

        book.toc = tuple(
            epub.Link(item.file_name, item.title, f"chap-{index}")
            for index, item in enumerate(items, start=1)
        )
        book.spine = ["nav", *items]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        epub.write_epub(str(path), book)
        return path

    return _make


@pytest.fixture()
def pdf_factory() -> PdfFactory:
    """Factory fixture building small text-based PDF files with PyMuPDF."""

    def _make(path: Path, chapters: list[tuple[str, str]] | None = None) -> Path:
        import pymupdf

        document = pymupdf.open()
        for title, text in chapters or FIXTURE_CHAPTERS:
            page = document.new_page()
            page.insert_text((72, 90), title, fontsize=20)
            y_position = 130
            for line in wrap_text(text, 78):
                page.insert_text((72, y_position), line, fontsize=11)
                y_position += 18
        document.set_metadata({"title": "The Lantern Keeper", "author": "Ada Fixture"})
        document.save(str(path))
        document.close()
        return path

    return _make


@pytest.fixture()
def sample_epub(epub_factory: EpubFactory, tmp_path: Path) -> Path:
    """A three-chapter EPUB fixture."""
    return epub_factory(tmp_path / "lantern.epub")


@pytest.fixture()
def sample_pdf(pdf_factory: PdfFactory, tmp_path: Path) -> Path:
    """A three-page PDF fixture."""
    return pdf_factory(tmp_path / "lantern.pdf")


@pytest.fixture()
def corpus_factory() -> CorpusFactory:
    """Factory building an in-memory book plus one chunk per chapter.

    Used by synthesis tests, which need chunks but no files or embedder.
    """

    def _make(
        chapters: list[tuple[str, str]] | None = None,
    ) -> tuple[BookDocument, list[Chunk]]:
        book = BookDocument(
            book_id="synthesis-book",
            title="The Lantern Keeper",
            source_path="lantern.epub",
            source_format="epub",
            chapters=[
                Chapter(
                    title=title,
                    blocks=[
                        Block(type="heading", text=title, level=1),
                        Block(type="paragraph", text=text),
                    ],
                )
                for title, text in (chapters or FIXTURE_CHAPTERS)
            ],
        )
        chunks: list[Chunk] = []
        for ordinal, chapter in enumerate(book.chapters):
            chunk_text = "\n\n".join(block.text for block in chapter.blocks)
            chunks.append(
                Chunk(
                    id=Chunk.make_id(book.book_id, ordinal, chunk_text),
                    book_id=book.book_id,
                    ordinal=ordinal,
                    text=chunk_text,
                    chapter=chapter.title,
                    heading_path=[chapter.title],
                    char_count=len(chunk_text),
                )
            )
        return book, chunks

    return _make


@pytest.fixture()
def tree_factory() -> TreeFactory:
    """Factory building an in-memory thematic summary tree.

    ``failed_chapters`` (1-based indices) get ``summary=None``; windows and the
    root summarize only the children that succeeded, mirroring ``build_tree``.
    """

    def _make(
        book: BookDocument,
        chunks: list[Chunk],
        *,
        window_size: int = 4,
        failed_chapters: tuple[int, ...] = (),
    ) -> SummaryTree:
        chunk_ids_by_chapter: dict[str, list[str]] = {}
        for chunk in chunks:
            chunk_ids_by_chapter.setdefault(chunk.chapter, []).append(chunk.id)

        chapter_nodes = [
            SummaryNode(
                id=chapter_node_id(book.book_id, index),
                level=1,
                title=chapter.title,
                summary=(
                    None if index in failed_chapters else f"Summary of {chapter.title}."
                ),
                chunk_ids=chunk_ids_by_chapter.get(chapter.title, []),
            )
            for index, chapter in enumerate(book.chapters, start=1)
        ]

        window_nodes: list[SummaryNode] = []
        if len(book.chapters) > window_size:
            for start in range(0, len(chapter_nodes), window_size):
                group = chapter_nodes[start : start + window_size]
                window_nodes.append(
                    SummaryNode(
                        id=window_node_id(book.book_id, start + 1, start + len(group)),
                        level=2,
                        title=f"Chapters {start + 1}-{start + len(group)}",
                        summary=_join_summaries(group),
                        children=[node.id for node in group],
                        chunk_ids=[
                            chunk_id for node in group for chunk_id in node.chunk_ids
                        ],
                    )
                )

        children = window_nodes or chapter_nodes
        root = SummaryNode(
            id=root_node_id(book.book_id),
            level=3,
            title=book.title,
            summary=_join_summaries(children),
            children=[node.id for node in children],
            chunk_ids=[chunk.id for chunk in chunks],
        )
        return SummaryTree(
            book_id=book.book_id,
            root_id=root.id,
            nodes=[*chapter_nodes, *window_nodes, root],
        )

    return _make


def _join_summaries(nodes: list[SummaryNode]) -> str | None:
    """Join available child summaries, or None when every child failed."""
    summaries = [node.summary for node in nodes if node.summary]
    return " ".join(summaries) if summaries else None


@pytest.fixture()
def raft_factory() -> RaftFactory:
    """Factory building RAFT examples against a book's chunks.

    The golden context (when answerable) comes first, then the distractors, so
    the target's ``[1]`` citation always points at the golden chunk.
    """

    def _make(
        chunks: list[Chunk],
        *,
        question: str = "What happened during the storm?",
        answerable: bool = True,
        golden_index: int = 0,
        distractor_count: int = 1,
        answer: str | None = None,
        suffix: str = "0",
    ) -> RaftExample:
        golden = chunks[golden_index]
        distractors = [chunk for chunk in chunks if chunk.id != golden.id][
            :distractor_count
        ]
        contexts = [
            RaftContext(chunk_id=chunk.id, text=chunk.text) for chunk in distractors
        ]
        if answerable:
            contexts.insert(
                0,
                RaftContext(chunk_id=golden.id, text=golden.text, is_golden=True),
            )
        if answer is None:
            answer = (
                'The relevant passage says: "quote" [1]\n\nanswer text'
                if answerable
                else refusal_text("The Lantern Keeper")
            )
        return RaftExample(
            id=f"example-{suffix}",
            book_id=golden.book_id,
            question=question,
            contexts=contexts,
            answer=answer,
            answerable=answerable,
        )

    return _make


@pytest.fixture()
def adapter_factory() -> AdapterFactory:
    """Factory building a valid T4 adapter directory for register tests."""

    def _make(
        path: Path,
        *,
        book_id: str = "the-lantern-keeper",
        base_model: str = "Qwen/Qwen3-4B",
        dataset_hash: str = "dataset-hash",
        config_hash: str = "config-hash",
        **overrides: object,
    ) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text(
            json.dumps({"r": 16, "lora_alpha": 32}), encoding="utf-8"
        )
        (path / "adapter_model.safetensors").write_bytes(b"fake-adapter-weights")
        report: dict[str, object] = {
            "book_id": book_id,
            "base_model": base_model,
            "dataset_hash": dataset_hash,
            "config_hash": config_hash,
            "created_at": "2026-10-01T00:00:00+00:00",
            "train_count": 4,
            "validation_count": 1,
            "epochs": 3,
            "learning_rate": 0.0002,
            "seed": 13,
            "loss_history": [1.2, 0.8, 0.5],
            "hardware": "Tesla T4",
        }
        report.update(overrides)
        (path / "run.json").write_text(json.dumps(report), encoding="utf-8")
        return path

    return _make


def _gguf_string(value: str) -> bytes:
    """Encode a GGUF string: u64 length prefix plus UTF-8 bytes."""
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_value(value: object) -> bytes:
    """Encode one synthetic metadata value (u32, string or string array)."""
    if isinstance(value, str):
        return struct.pack("<I", _GGUF_TYPE_STRING) + _gguf_string(value)
    if isinstance(value, list):
        return (
            struct.pack("<I", _GGUF_TYPE_ARRAY)
            + struct.pack("<I", _GGUF_TYPE_STRING)
            + struct.pack("<Q", len(value))
            + b"".join(_gguf_string(item) for item in value)
        )
    return struct.pack("<I", _GGUF_TYPE_UINT32) + struct.pack("<I", int(value))


@pytest.fixture()
def gguf_factory() -> GgufFactory:
    """Factory building small synthetic GGUF files (valid and broken).

    The files carry a real header, metadata KV table and tensor-info table, so
    the dependency-free reader can be exercised without binaries or downloads.
    """

    def _make(
        path: Path,
        *,
        version: int = 3,
        architecture: str = "qwen3",
        model_name: str = "distiller-lantern-q4_k_m",
        file_type: int | None = 15,
        tensors: list[tuple[str, tuple[int, ...]]] | None = None,
        extra_metadata: dict[str, object] | None = None,
        magic: bytes = b"GGUF",
        truncate: int | None = None,
    ) -> Path:
        tensor_shapes = tensors or [
            ("token_embd.weight", (128, 64)),
            ("output.weight", (32,)),
        ]
        metadata: dict[str, object] = {
            "general.architecture": architecture,
            "general.name": model_name,
        }
        if file_type is not None:
            metadata["general.file_type"] = file_type
        metadata.update(extra_metadata or {})

        payload = magic + struct.pack("<I", version)
        payload += struct.pack("<QQ", len(tensor_shapes), len(metadata))
        for key, value in metadata.items():
            payload += _gguf_string(key) + _gguf_value(value)
        offset = 0
        for name, dimensions in tensor_shapes:
            payload += _gguf_string(name)
            payload += struct.pack("<I", len(dimensions))
            payload += struct.pack(f"<{len(dimensions)}Q", *dimensions)
            payload += struct.pack("<I", 2)  # ggml tensor type (unused here)
            payload += struct.pack("<Q", offset)
            offset += 4096

        if truncate is not None:
            payload = payload[:truncate]
        path.write_bytes(payload)
        return path

    return _make
