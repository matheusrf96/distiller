"""Shared CLI helpers: validated settings overrides and pipeline construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import typer
from pydantic import BaseModel, ValidationError

from ..evaluation import load_golden
from ..exceptions import DistillerError
from ..indexing import IndexBundle, build_index, load_index
from ..ingest import ingest_book
from ..llm import get_llm
from ..rag import Generator, QAPipeline, Retriever, get_reranker

if TYPE_CHECKING:
    from pathlib import Path

    from ..config import Settings
    from ..evaluation import GoldenItem
    from ..models import BookDocument


def apply_overrides[ModelT: BaseModel](model: ModelT, **changes: Any) -> ModelT:
    """Return a validated copy of a settings model with CLI overrides applied.

    Args:
        model: Settings model to copy.
        **changes: Field overrides, validated against the model.

    Returns:
        Validated copy with the overrides applied.

    Raises:
        typer.BadParameter: When an overridden value fails validation.
    """
    payload = {**model.model_dump(), **changes}
    try:
        return type(model).model_validate(payload)
    except ValidationError as exc:
        raise typer.BadParameter(f"Invalid setting: {_first_error(exc)}") from exc


def ingest_or_fail(
    source: Path,
    *,
    settings: Settings,
    title: str | None = None,
) -> BookDocument:
    """Parse a book, turning domain errors into friendly CLI errors.

    Args:
        source: Path to the book file (validated to exist by Typer).
        settings: Pipeline settings.
        title: Optional title override.

    Returns:
        Parsed book document.

    Raises:
        typer.BadParameter: When the format is unsupported or parsing fails.
    """
    try:
        return ingest_book(source, settings=settings, title=title)
    except DistillerError as exc:
        raise typer.BadParameter(str(exc)) from exc


def build_index_or_fail(book_id: str, settings: Settings) -> dict[str, Any]:
    """Build a book index, turning domain errors into friendly CLI errors.

    Args:
        book_id: Slug of the ingested book.
        settings: Pipeline settings.

    Returns:
        Index metadata written to ``index/metadata.json``.

    Raises:
        typer.BadParameter: When the book was never ingested or chunking fails.
    """
    try:
        return build_index(book_id, settings)
    except DistillerError as exc:
        raise typer.BadParameter(str(exc)) from exc


def load_golden_set(path: Path) -> list[GoldenItem]:
    """Load a golden question set, turning format errors into friendly CLI errors.

    Args:
        path: Golden set file (YAML, JSON or JSONL).

    Returns:
        Validated golden items.

    Raises:
        typer.BadParameter: When the file is malformed or empty.
    """
    try:
        return load_golden(path)
    except ValidationError as exc:
        raise typer.BadParameter(
            f"Invalid golden set {path}: {_first_error(exc)}"
        ) from exc
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid golden set {path}: {exc}") from exc


def load_book_index(settings: Settings, book_id: str) -> IndexBundle:
    """Load a book index, turning domain errors into friendly CLI errors.

    Args:
        settings: Pipeline settings.
        book_id: Slug of the book to load.

    Returns:
        Loaded index bundle.

    Raises:
        typer.BadParameter: When artifacts are missing or inconsistent.
    """
    try:
        return load_index(book_id, settings)
    except DistillerError as exc:
        raise typer.BadParameter(str(exc)) from exc


def build_pipeline(
    settings: Settings, bundle: IndexBundle, *, rerank: bool = False
) -> QAPipeline:
    """Assemble the retrieval -> (rerank) -> generation pipeline for one book.

    Args:
        settings: Pipeline settings.
        bundle: Loaded index bundle for the book.
        rerank: Force-enable cross-encoder reranking for this run.

    Returns:
        Pipeline that answers questions with citations.
    """
    retriever = Retriever(bundle, settings.retrieval)
    generator = Generator(
        get_llm(settings),
        bundle.book.title,
        max_tokens=settings.llm.max_tokens,
    )
    reranker = None
    if rerank or settings.retrieval.rerank:
        reranker = get_reranker(apply_overrides(settings.retrieval, rerank=True))
    return QAPipeline(
        retriever, generator, reranker=reranker, settings=settings.retrieval
    )


def _first_error(exc: ValidationError) -> str:
    """Format the first validation error as ``location: message``."""
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}" if location else str(error["msg"])
