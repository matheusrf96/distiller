"""Rich rendering for CLI output. Commands stay thin; presentation lives here."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..evaluation import metric_deltas

if TYPE_CHECKING:
    from pathlib import Path

    from ..evaluation import AblationReport
    from ..models import Answer, BookDocument

console = Console()

_ABLATION_COLUMNS = (
    ("retrieval_hit_rate", "hit rate"),
    ("contains_rate", "contains"),
    ("citation_coverage", "citations"),
    ("refusal_accuracy", "refusal"),
)
_ABLATION_DELTA_COLUMNS = (
    ("retrieval_hit_rate", "Δ hit"),
    ("contains_rate", "Δ contains"),
)


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


def render_ablation(report: AblationReport) -> None:
    """Render the ablation comparison table with deltas versus the baseline."""
    deltas = metric_deltas(
        report, tuple(metric for metric, _ in _ABLATION_DELTA_COLUMNS)
    )

    table = Table(title=f"Ablation: {report.book_id} ({report.item_count} questions)")
    table.add_column("variant")
    table.add_column("rerank", justify="center")
    table.add_column("top-k", justify="right")
    for _, label in _ABLATION_COLUMNS:
        table.add_column(label, justify="right")
    for _, label in _ABLATION_DELTA_COLUMNS:
        table.add_column(label, justify="right")

    for result in report.variants:
        variant = result.variant
        row = [
            variant.name,
            "yes" if variant.rerank else "no",
            str(variant.top_k_final),
        ]
        if result.skipped_reason is None:
            row.extend(
                _format_metric(result.metrics.get(metric))
                for metric, _ in _ABLATION_COLUMNS
            )
            row.extend(
                _format_delta(deltas.get(variant.name, {}).get(metric))
                for metric, _ in _ABLATION_DELTA_COLUMNS
            )
        else:
            row.extend("—" for _ in _ABLATION_COLUMNS)
            row.extend("—" for _ in _ABLATION_DELTA_COLUMNS)
        table.add_row(*row)
    console.print(table)

    for result in report.variants:
        if result.skipped_reason:
            console.print(
                f"[yellow]{result.variant.name}: skipped — "
                f"{result.skipped_reason}[/yellow]"
            )


def _format_metric(value: Any) -> str:
    return "—" if value is None else str(value)


def _format_delta(value: float | None) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "0"
    return f"{value:+.4f}"


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
