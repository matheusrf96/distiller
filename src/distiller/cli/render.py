"""Rich rendering for CLI output. Commands stay thin; presentation lives here."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from pathlib import Path

    from ..models import Answer, BookDocument

console = Console()


def render_ingest(book: BookDocument, artifacts_root: Path) -> None:
    """Render the summary table shown after ``distiller ingest``."""
    summary_table(
        f"Ingested: {book.title}",
        [
            ("book id", book.book_id),
            ("format", book.source_format),
            ("language", book.language or "-"),
            ("authors", ", ".join(book.authors) or "-"),
            ("chapters", str(len(book.chapters))),
            ("blocks", str(book.block_count)),
            ("characters", f"{book.char_count:,}"),
            ("backend", str(book.metadata.get("backend", "-"))),
            ("artifacts", str(artifacts_root)),
        ],
    )


def render_index(book_id: str, metadata: dict[str, Any]) -> None:
    """Render the summary table shown after ``distiller index``."""
    keys = ("chunk_count", "average_chunk_chars", "embedder", "dim", "store")
    summary_table(
        f"Indexed: {book_id}", [(key, str(metadata.get(key))) for key in keys]
    )


def render_answer(answer: Answer, book_title: str) -> None:
    """Render the answer panel and its sources table for ``distiller ask``."""
    border = "yellow" if answer.refused else "green"
    console.print(
        Panel(answer.text, title=f"[bold]{book_title}[/bold]", border_style=border)
    )

    if answer.citations:
        table = Table(title="Sources")
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Location")
        table.add_column("chunk", style="dim")
        for citation in answer.citations:
            location = citation.label
            table.add_row(
                f"[{citation.index}]", location, citation.chunk_id.split(":")[-1]
            )
        console.print(table)
    elif not answer.refused:
        console.print(
            "[yellow]No citations extracted — "
            "inspect the answer for grounding.[/yellow]"
        )


def render_eval(book_id: str, summary: dict[str, Any], item_count: int) -> None:
    """Render the metrics table shown after ``distiller eval``."""
    rows = [
        (key, "—" if value is None else str(value)) for key, value in summary.items()
    ]
    summary_table(f"Eval: {book_id} ({item_count} questions)", rows)


def render_ragas(payload: dict[str, Any]) -> None:
    """Render the RAGAS metric payload."""
    console.print(
        Panel(json.dumps(payload, indent=2), title="RAGAS", border_style="magenta")
    )


def render_books(rows: list[dict[str, Any]]) -> None:
    """Render the book list shown by ``distiller books``."""
    if not rows:
        console.print("No books ingested yet. Run: distiller ingest <file>")
        return

    table = Table(title="Books")
    for column in ("book_id", "title", "format", "chapters", "indexed"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            str(row["book_id"]),
            str(row["title"]),
            str(row["format"]),
            str(row["chapters"]),
            "yes" if row["indexed"] else "no",
        )
    console.print(table)


def render_info(
    book_id: str, book: dict[str, Any], index_metadata: dict[str, Any] | None
) -> None:
    """Render the book + index details shown by ``distiller info``."""
    authors = ", ".join(book.get("authors") or [])
    console.print(Panel(f"[bold]{book.get('title')}[/bold]\n{authors}", title=book_id))
    if index_metadata:
        summary_table(
            "Index", [(str(key), str(value)) for key, value in index_metadata.items()]
        )
    else:
        console.print(
            f"[yellow]Not indexed yet.[/yellow] Run: distiller index {book_id}"
        )


def summary_table(title: str, rows: list[tuple[str, str]]) -> None:
    """Print a two-column, headerless summary table."""
    table = Table(title=title, show_header=False)
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)
