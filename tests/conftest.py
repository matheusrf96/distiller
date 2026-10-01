"""Shared pytest fixtures.

Everything here generates books locally: no network, no binary fixtures, no
model downloads. The offline pipeline uses the hashing embedder and the fake LLM.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from distiller.config import Settings, load_settings
from distiller.models import Block, BookDocument, Chapter, Chunk, refusal_text
from distiller.synthesis import RaftContext, RaftExample
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
