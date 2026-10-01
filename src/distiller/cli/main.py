"""`distiller` command line interface."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003 - typer resolves command annotations at runtime
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from rich.logging import RichHandler

from ..config import Settings, load_settings
from ..evaluation import (
    ItemResult,
    build_variants,
    evaluate_item,
    run_ablation,
    run_ragas,
    summarize,
)
from ..indexing import list_books
from ..ingest.common import render_book_markdown
from ..llm import get_llm
from ..paths import BookPaths
from ..synthesis import load_cached_pairs, synthesize
from ..training import run_comparison
from ..utils import read_json, write_json, write_jsonl
from .context import (
    adapter_identity,
    adapter_llm_or_fail,
    apply_overrides,
    build_ablation_runs,
    build_index_or_fail,
    build_pipeline,
    build_training_runs,
    check_training_runtime_or_fail,
    ingest_or_fail,
    load_book_and_chunks,
    load_book_index,
    load_golden_set,
    load_registered_adapter_or_fail,
    load_training_examples,
    parse_top_k_values,
    pipeline_model_name,
    prepare_training_or_fail,
    register_adapter_or_fail,
)
from .render import (
    console,
    render_ablation,
    render_answer,
    render_books,
    render_comparison,
    render_eval,
    render_index,
    render_info,
    render_ingest,
    render_ragas,
    render_registration,
    render_runtime,
    render_synthesis,
    render_training,
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
        book = ingest_or_fail(source, settings=settings, title=title)

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
    contextual: Annotated[
        bool | None,
        typer.Option(
            "--contextual/--no-contextual",
            help="Generate per-chunk context before indexing (contextual retrieval).",
        ),
    ] = None,
) -> None:
    """Chunk + embed + index a book (writes chunks.jsonl and the vector store)."""
    settings = settings_from_context(ctx)
    if embedding_backend is not None:
        settings.embedding = apply_overrides(
            settings.embedding, backend=embedding_backend
        )
    if contextual is not None:
        settings.enrichment = apply_overrides(settings.enrichment, enabled=contextual)

    with console.status(
        f"Building index for '{book_id}' "
        f"with {settings.embedding.backend} embeddings..."
    ):
        metadata = build_index_or_fail(book_id, settings)
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
    adapter: Annotated[
        bool,
        typer.Option("--adapter", help="Answer with the registered adapter endpoint."),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the full Answer object as JSON."),
    ] = False,
) -> None:
    """Ask a question about an indexed book; answers carry citations."""
    settings = settings_from_context(ctx)
    bundle = load_book_index(settings, book_id)

    llm = None
    if adapter:
        load_registered_adapter_or_fail(settings, book_id)
        llm = adapter_llm_or_fail(settings)
    pipeline = build_pipeline(settings, bundle, rerank=rerank, llm=llm)

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
    adapter: Annotated[
        bool,
        typer.Option(
            "--adapter", help="Evaluate with the registered adapter endpoint."
        ),
    ] = False,
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

    golden_items = load_golden_set(golden_path)
    if limit:
        golden_items = golden_items[:limit]

    bundle = load_book_index(settings, book_id)

    adapter_provenance: dict[str, Any] | None = None
    if adapter:
        adapter_report, adapter_dir = load_registered_adapter_or_fail(settings, book_id)
        adapter_provenance = adapter_identity(adapter_report, adapter_dir)
        pipeline = build_pipeline(settings, bundle, llm=adapter_llm_or_fail(settings))
    else:
        pipeline = build_pipeline(settings, bundle)

    results, samples = run_golden_set(pipeline, golden_items)

    index_metadata = (
        read_json(paths.index_metadata) if paths.index_metadata.exists() else {}
    )

    generator: dict[str, Any] = {
        "kind": "adapter" if adapter else "base",
        "model": pipeline_model_name(pipeline),
        "adapter": adapter_provenance,
    }
    report: dict[str, Any] = {
        "book_id": book_id,
        # Legacy top-level key, kept for backward compatibility; the
        # authoritative generator identity lives in the `generator` block.
        "model": generator["model"],
        "generator": generator,
        "index": {
            "embedder": index_metadata.get("embedder"),
            "contextual": index_metadata.get("contextual"),
            "enriched_chunks": index_metadata.get("enriched_chunks"),
        },
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


@app.command("ablation")
def ablation(
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
    top_k: Annotated[
        str | None,
        typer.Option("--top-k", help="Comma-separated top-k sweep, e.g. 4,8,12."),
    ] = None,
    no_rerank: Annotated[
        bool,
        typer.Option("--no-rerank", help="Run only non-reranked variants."),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full report as JSON.")
    ] = False,
) -> None:
    """Compare retrieval configurations (rerank, top-k) on one golden set."""
    settings = settings_from_context(ctx)
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    golden_path = golden or paths.golden_yaml
    if not golden_path.exists():
        raise typer.BadParameter(
            f"Golden set not found: {golden_path} (create one or pass --golden)"
        )

    golden_items = load_golden_set(golden_path)
    if limit:
        golden_items = golden_items[:limit]

    bundle = load_book_index(settings, book_id)
    variants = build_variants(
        top_k_final=settings.retrieval.top_k_final,
        rerank_pool=settings.retrieval.rerank_pool,
        top_k_values=parse_top_k_values(top_k),
        include_rerank=not no_rerank,
    )
    runs = build_ablation_runs(settings, bundle, variants)

    index_metadata = (
        read_json(paths.index_metadata) if paths.index_metadata.exists() else {}
    )
    with console.status("Running ablation..."):
        report = run_ablation(
            book_id,
            golden_items,
            runs,
            index_identity={
                "embedder": index_metadata.get("embedder"),
                "contextual": index_metadata.get("contextual"),
                "enriched_chunks": index_metadata.get("enriched_chunks"),
            },
        )

    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = paths.eval_dir / "ablation.json"
    write_json(report_path, report.model_dump())

    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    render_ablation(report)
    console.print(f"Report written to [cyan]{report_path}[/cyan]")


@app.command("synth")
def synth(
    ctx: typer.Context,
    book_id: Annotated[str, typer.Argument(help="Book id.")],
    max_chunks: Annotated[
        int | None,
        typer.Option("--max-chunks", help="Maximum chunks sampled for generation."),
    ] = None,
    questions_per_chunk: Annotated[
        int | None,
        typer.Option("--questions-per-chunk", help="Questions requested per chunk."),
    ] = None,
    distractors: Annotated[
        int | None,
        typer.Option("--distractors", help="Distractor chunks per RAFT example."),
    ] = None,
    negative_ratio: Annotated[
        float | None,
        typer.Option(
            "--negative-ratio",
            help="Fraction of examples that are unanswerable negatives.",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for sampling and shuffling."),
    ] = None,
    regenerate: Annotated[
        bool,
        typer.Option(
            "--regenerate", help="Ignore cached qa.jsonl and call the LLM again."
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the manifest as JSON.")
    ] = False,
) -> None:
    """Generate grounded QA pairs and RAFT training examples for a book."""
    settings = settings_from_context(ctx)
    overrides: dict[str, Any] = {
        key: value
        for key, value in {
            "max_chunks": max_chunks,
            "questions_per_chunk": questions_per_chunk,
            "distractors": distractors,
            "negative_ratio": negative_ratio,
            "seed": seed,
        }.items()
        if value is not None
    }
    if overrides:
        settings.synthesis = apply_overrides(settings.synthesis, **overrides)

    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    book, chunks = load_book_and_chunks(settings, book_id)
    index_metadata = (
        read_json(paths.index_metadata) if paths.index_metadata.exists() else {}
    )

    cached_pairs = None if regenerate else load_cached_pairs(paths.dataset_qa_jsonl)
    with console.status("Synthesizing dataset..."):
        run = synthesize(
            book,
            chunks,
            get_llm(settings),
            settings.synthesis,
            index_identity={
                "embedder": index_metadata.get("embedder"),
                "contextual": index_metadata.get("contextual"),
                "enriched_chunks": index_metadata.get("enriched_chunks"),
            },
            cached_pairs=cached_pairs,
        )

    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.dataset_qa_jsonl, run.pairs)
    write_jsonl(paths.dataset_rejected_jsonl, run.outcome.rejected)
    write_jsonl(paths.dataset_raft_jsonl, run.examples)
    write_json(paths.dataset_manifest, run.manifest.model_dump())

    if as_json:
        typer.echo(run.manifest.model_dump_json(indent=2))
        return
    render_synthesis(run.manifest, paths.dataset_dir)
    console.print(f"Dataset written to [cyan]{paths.dataset_dir}[/cyan]")


@app.command("train")
def train(
    ctx: typer.Context,
    book_id: Annotated[str, typer.Argument(help="Book id.")],
    seed: Annotated[
        int | None, typer.Option("--seed", help="Seed for the train/validation split.")
    ] = None,
    val_ratio: Annotated[
        float | None,
        typer.Option("--val-ratio", help="Fraction held out for validation."),
    ] = None,
    register: Annotated[
        Path | None,
        typer.Option(
            "--register", help="Register an adapter directory downloaded from the T4."
        ),
    ] = None,
    check_runtime: Annotated[
        bool,
        typer.Option(
            "--check-runtime",
            help="Report whether the heavy training stack is installed.",
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the report as JSON.")
    ] = False,
) -> None:
    """Prepare RAFT data as chat examples, or register a T4 adapter."""
    settings = settings_from_context(ctx)
    if check_runtime:
        render_runtime(check_training_runtime_or_fail())
        return

    if seed is not None or val_ratio is not None:
        overrides: dict[str, Any] = {}
        if seed is not None:
            overrides["seed"] = seed
        if val_ratio is not None:
            overrides["val_ratio"] = val_ratio
        settings.training = apply_overrides(settings.training, **overrides)

    paths = BookPaths.for_book(settings.artifacts_dir, book_id)

    if register is not None:
        report, adapter_dir = register_adapter_or_fail(settings, book_id, register)
        if as_json:
            typer.echo(report.model_dump_json(indent=2))
            return
        render_registration(report, adapter_dir)
        console.print(f"Adapter registered at [cyan]{adapter_dir}[/cyan]")
        return

    book, chunks = load_book_and_chunks(settings, book_id)
    examples = load_training_examples(settings, book_id)
    with console.status("Preparing training data..."):
        bundle = prepare_training_or_fail(book, examples, chunks, settings, paths)

    if as_json:
        typer.echo(bundle.manifest.model_dump_json(indent=2))
        return
    render_training(bundle, paths)
    console.print(f"Training data written to [cyan]{paths.training_dir}[/cyan]")


@app.command("train-eval")
def train_eval(
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
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full report as JSON.")
    ] = False,
) -> None:
    """Compare the base model and the registered adapter on one golden set."""
    settings = settings_from_context(ctx)
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    golden_path = golden or paths.golden_yaml
    if not golden_path.exists():
        raise typer.BadParameter(
            f"Golden set not found: {golden_path} (create one or pass --golden)"
        )

    golden_items = load_golden_set(golden_path)
    if limit:
        golden_items = golden_items[:limit]

    adapter_report, adapter_dir = load_registered_adapter_or_fail(settings, book_id)
    adapter_llm = adapter_llm_or_fail(settings)
    bundle = load_book_index(settings, book_id)
    runs = build_training_runs(
        settings,
        bundle,
        adapter_llm,
        adapter_identity=adapter_identity(adapter_report, adapter_dir),
    )

    index_metadata = (
        read_json(paths.index_metadata) if paths.index_metadata.exists() else {}
    )
    with console.status("Comparing base and adapter..."):
        comparison = run_comparison(
            book_id,
            golden_items,
            runs,
            index_identity={
                "embedder": index_metadata.get("embedder"),
                "contextual": index_metadata.get("contextual"),
                "enriched_chunks": index_metadata.get("enriched_chunks"),
            },
            retrieval_identity=settings.retrieval.model_dump(),
        )

    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    write_json(paths.eval_training_json, comparison.model_dump())

    if as_json:
        typer.echo(comparison.model_dump_json(indent=2))
        return
    render_comparison(comparison)
    console.print(f"Report written to [cyan]{paths.eval_training_json}[/cyan]")


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
