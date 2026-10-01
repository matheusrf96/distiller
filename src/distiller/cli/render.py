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
    from ..paths import BookPaths
    from ..synthesis import DatasetManifest
    from ..training import TrainingBundle, TrainingComparison, TrainingReport

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
_TRAINING_COLUMNS = (
    ("contains_rate", "contains"),
    ("refusal_accuracy", "refusal"),
    ("citation_coverage", "citations"),
)
_TRAINING_DELTA_COLUMNS = (
    ("contains_rate", "Δ contains"),
    ("refusal_accuracy", "Δ refusal"),
    ("citation_coverage", "Δ citations"),
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


def render_synthesis(manifest: DatasetManifest, dataset_dir: Path) -> None:
    """Render the synth summary table shown after `distiller synth`."""
    rejected_total = sum(manifest.rejected.values())
    summary_table(
        f"Synthesized: {manifest.book_id}",
        [
            ("model", manifest.model),
            ("source", manifest.source),
            ("chunks sampled", f"{manifest.sampled_chunks} of {manifest.chunk_count}"),
            ("pairs generated", str(manifest.generated_pairs)),
            ("pairs kept", str(manifest.kept_pairs)),
            ("pairs rejected", str(rejected_total)),
            ("raft examples", str(manifest.example_count)),
            (
                "answerable / negative",
                f"{manifest.answerable_examples} / {manifest.unanswerable_examples}",
            ),
            ("dataset", str(dataset_dir)),
        ],
    )


def render_training(bundle: TrainingBundle, paths: BookPaths) -> None:
    """Render the training-preparation summary table shown after `train`."""
    manifest = bundle.manifest
    summary_table(
        f"Training data: {manifest.book_id}",
        [
            ("train examples", str(manifest.train_count)),
            ("validation examples", str(manifest.validation_count)),
            (
                "answerable / negative",
                f"{manifest.answerable_count} / {manifest.unanswerable_count}",
            ),
            ("seed", str(manifest.seed)),
            ("validation ratio", str(manifest.val_ratio)),
            ("source hash", manifest.source_hash),
            ("dataset hash", manifest.dataset_hash),
            ("prompt hash", manifest.prompt_hash),
            ("train split", str(paths.training_train_jsonl)),
            ("validation split", str(paths.training_validation_jsonl)),
            ("manifest", str(paths.training_manifest)),
            ("qlora config", str(paths.training_qlora_json)),
            ("notebook", str(paths.training_notebook)),
        ],
    )


def render_comparison(report: TrainingComparison) -> None:
    """Render the base-vs-adapter table with deltas versus the base variant.

    Metrics and deltas are printed as two compact tables so every column stays
    readable on an 80-column terminal.
    """
    metrics_table = Table(
        title=f"Training comparison: {report.book_id} ({report.item_count} questions)"
    )
    metrics_table.add_column("variant")
    metrics_table.add_column("generator")
    for _, label in _TRAINING_COLUMNS:
        metrics_table.add_column(label, justify="right")

    deltas_table = Table(title=f"Deltas vs {report.baseline or 'base'}")
    deltas_table.add_column("variant")
    for _, label in _TRAINING_DELTA_COLUMNS:
        deltas_table.add_column(label, justify="right")

    for result in report.variants:
        variant = result.variant
        generator = f"{variant.kind} ({variant.model})"
        if result.skipped_reason is None:
            metrics_table.add_row(
                variant.name,
                generator,
                *(
                    _format_metric(result.metrics.get(metric))
                    for metric, _ in _TRAINING_COLUMNS
                ),
            )
            deltas_table.add_row(
                variant.name,
                *(
                    _format_delta(report.deltas.get(variant.name, {}).get(metric))
                    for metric, _ in _TRAINING_DELTA_COLUMNS
                ),
            )
        else:
            metrics_table.add_row(
                variant.name, generator, *("—" for _ in _TRAINING_COLUMNS)
            )
            deltas_table.add_row(variant.name, *("—" for _ in _TRAINING_DELTA_COLUMNS))
    console.print(metrics_table)
    console.print(deltas_table)

    for result in report.variants:
        if result.skipped_reason:
            console.print(
                f"[yellow]{result.variant.name}: skipped — "
                f"{result.skipped_reason}[/yellow]"
            )


def render_registration(report: TrainingReport, adapter_dir: Path) -> None:
    """Render the adapter registration summary shown after `train --register`."""
    summary_table(
        f"Registered adapter: {report.book_id}",
        [
            ("base model", report.base_model),
            ("dataset hash", report.dataset_hash),
            ("config hash", report.config_hash),
            ("dataset match", "yes" if report.dataset_hash_matches else "no"),
            (
                "examples",
                f"{report.train_count} train / {report.validation_count} validation",
            ),
            ("loss points", str(len(report.loss_history))),
            ("hardware", report.hardware or "-"),
            ("adapter", str(adapter_dir)),
        ],
    )
    if not report.dataset_hash_matches:
        console.print(
            "[yellow]Adapter was trained on a different dataset; comparison "
            "results may not reflect the current split.[/yellow]"
        )


def render_runtime(versions: dict[str, str]) -> None:
    """Render the `train --check-runtime` version table."""
    summary_table(
        "Training runtime", [(name, version) for name, version in versions.items()]
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
