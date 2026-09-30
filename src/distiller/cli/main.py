"""`distiller` command line interface."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003 - typer resolves command annotations at runtime
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from rich.logging import RichHandler

from ..config import Settings, load_settings
from ..evaluation import ItemResult, evaluate_item, load_golden, run_ragas, summarize
from ..indexing import build_index, list_books
from ..ingest import ingest_book
from ..ingest.common import render_book_markdown
from ..paths import BookPaths
from ..utils import read_json, write_json
from .context import apply_overrides, build_pipeline, load_book_index
from .render import (
    console,
    render_answer,
    render_books,
    render_eval,
    render_index,
    render_info,
    render_ingest,
    render_ragas,
)

if TYPE_CHECKING:
    from ..evaluation import GoldenItem
    from ..rag import QAPipeline

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Distill PDF/ePub books into a searchable index and a specialized tiny model.",
)


@app.callback()
def main(
    ctx: typer.Context,
    artifacts_dir: Annotated[
        Path | None,
        typer.Option(
            "--artifacts-dir",
            help="Where book artifacts are stored (default: ./artifacts).",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Show debug logs (parser fallbacks, model loading)."
        ),
    ] = False,
) -> None:
    """Configure logging and shared settings for every command."""
    configure_logging(verbose)
    settings = load_settings()
    if artifacts_dir is not None:
        settings.artifacts_dir = artifacts_dir
    ctx.obj = settings


@app.command()
def ingest(
    ctx: typer.Context,
    source: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Book file (.pdf, .epub, .md, .txt).",
        ),
    ],
    title: Annotated[
        str | None,
        typer.Option(
            "--title", "-t", help="Override the book title (also sets the book id)."
        ),
    ] = None,
    pdf_backend: Annotated[
        str | None,
        typer.Option("--pdf-backend", help="auto | docling | pymupdf4llm | pymupdf"),
    ] = None,
) -> None:
    """Parse a book into `artifacts/<book-id>/book.json` + `parsed.md`."""
    settings = settings_from_context(ctx)
    if pdf_backend is not None:
        settings.ingest = apply_overrides(settings.ingest, pdf_backend=pdf_backend)

    with console.status(f"Parsing {source.name}..."):
        book = ingest_book(source, settings=settings, title=title)

    paths = BookPaths.for_book(settings.artifacts_dir, book.book_id).ensure()
    write_json(paths.book_json, book.model_dump())
    paths.parsed_md.write_text(render_book_markdown(book), encoding="utf-8")
    render_ingest(book, paths.root)


@app.command()
def index(
    ctx: typer.Context,
    book_id: Annotated[
        str, typer.Argument(help="Book id printed by `distiller ingest`.")
    ],
    embedding_backend: Annotated[
        str | None,
        typer.Option(
            "--embedding-backend", help="sentence-transformers | hash (offline tests)"
        ),
    ] = None,
) -> None:
    """Chunk + embed + index a book (writes chunks.jsonl and the vector store)."""
    settings = settings_from_context(ctx)
    if embedding_backend is not None:
        settings.embedding = apply_overrides(
            settings.embedding, backend=embedding_backend
        )

    with console.status(
        f"Building index for '{book_id}' "
        f"with {settings.embedding.backend} embeddings..."
    ):
        metadata = build_index(book_id, settings)
    render_index(book_id, metadata)


@app.command()
def ask(
    ctx: typer.Context,
    book_id: Annotated[str, typer.Argument(help="Book id.")],
    question: Annotated[
        list[str],
        typer.Argument(help="The question (quote it or just type the words)."),
    ],
    top_k: Annotated[
        int | None,
        typer.Option("--top-k", "-k", help="How many chunks to feed the model."),
    ] = None,
    chapter: Annotated[
        str | None,
        typer.Option(
            "--chapter", help="Restrict retrieval to chapters containing this text."
        ),
    ] = None,
    rerank: Annotated[
        bool,
        typer.Option("--rerank/--no-rerank", help="Enable cross-encoder reranking."),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the full Answer object as JSON."),
    ] = False,
) -> None:
    """Ask a question about an indexed book; answers carry citations."""
    settings = settings_from_context(ctx)
    bundle = load_book_index(settings, book_id)
    pipeline = build_pipeline(settings, bundle, rerank=rerank)

    with console.status("Thinking..."):
        answer = pipeline.ask(" ".join(question).strip(), chapter=chapter, top_k=top_k)

    if as_json:
        typer.echo(answer.model_dump_json(indent=2))
        return
    render_answer(answer, bundle.book.title)


@app.command("eval")
def evaluate(
    ctx: typer.Context,
    book_id: Annotated[str, typer.Argument(help="Book id.")],
    golden: Annotated[
        Path | None,
        typer.Option(
            "--golden", help="Golden set file (default: artifacts/<book>/golden.yaml)."
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Evaluate only the first N questions."),
    ] = None,
    with_ragas: Annotated[
        bool,
        typer.Option(
            "--ragas", help="Also run RAGAS LLM-judged metrics (needs the eval extra)."
        ),
    ] = False,
) -> None:
    """Run the golden question set and write eval/report.json."""
    settings = settings_from_context(ctx)
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    golden_path = golden or paths.golden_yaml
    if not golden_path.exists():
        raise typer.BadParameter(
            f"Golden set not found: {golden_path} (create one or pass --golden)"
        )

    golden_items = load_golden(golden_path)
    if limit:
        golden_items = golden_items[:limit]

    bundle = load_book_index(settings, book_id)
    pipeline = build_pipeline(settings, bundle)
    results, samples = run_golden_set(pipeline, golden_items)

    report: dict[str, Any] = {
        "book_id": book_id,
        "model": pipeline_model_name(pipeline),
        "metrics": summarize(results),
        "items": [result.model_dump() for result in results],
    }
    if with_ragas:
        report["ragas"] = run_ragas(
            samples,
            model=settings.evaluation.judge_model or settings.llm.model,
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            max_samples=settings.evaluation.max_samples,
        )

    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = paths.eval_dir / "report.json"
    write_json(report_path, report)

    render_eval(book_id, report["metrics"], len(results))
    if "ragas" in report:
        render_ragas(report["ragas"])
    console.print(f"Report written to [cyan]{report_path}[/cyan]")


@app.command("books")
def books(ctx: typer.Context) -> None:
    """List ingested books."""
    render_books(list_books(settings_from_context(ctx).artifacts_dir))


@app.command("info")
def info(
    ctx: typer.Context,
    book_id: Annotated[str, typer.Argument(help="Book id.")],
) -> None:
    """Show details for one book and its index."""
    settings = settings_from_context(ctx)
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    if not paths.book_json.exists():
        raise typer.BadParameter(
            f"Unknown book '{book_id}' in {settings.artifacts_dir}"
        )

    index_metadata = (
        read_json(paths.index_metadata) if paths.index_metadata.exists() else None
    )
    render_info(book_id, read_json(paths.book_json), index_metadata)


def settings_from_context(ctx: typer.Context) -> Settings:
    """Return the settings attached to the Typer context by the callback."""
    return cast("Settings", ctx.obj)


def run_golden_set(
    pipeline: QAPipeline,
    golden_items: list[GoldenItem],
) -> tuple[list[ItemResult], list[dict[str, Any]]]:
    """Ask every golden question and collect results plus RAGAS samples.

    Args:
        pipeline: Assembled QA pipeline for the book.
        golden_items: Validated golden questions.

    Returns:
        Per-question results and the sample rows needed for RAGAS.
    """
    results: list[ItemResult] = []
    samples: list[dict[str, Any]] = []
    for golden_item in golden_items:
        answer = pipeline.ask(golden_item.question)
        results.append(evaluate_item(golden_item, answer))
        samples.append(
            {
                "question": golden_item.question,
                "answer": answer.text,
                "contexts": [context.chunk.text for context in answer.contexts],
            }
        )
    return results, samples


def pipeline_model_name(pipeline: QAPipeline) -> str:
    """Return the generator's model identifier for the report."""
    return getattr(pipeline.generator.llm, "name", "unknown")


def configure_logging(verbose: bool) -> None:
    """Route Python logging through rich, at DEBUG when ``--verbose`` is set."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=verbose)],
        force=True,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
