"""`distiller` command line interface."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003 - typer resolves command annotations at runtime
from typing import TYPE_CHECKING, Annotated, Any, Protocol, cast

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
from ..exceptions import DistillerError
from ..indexing import list_books
from ..ingest.common import render_book_markdown
from ..llm import get_llm
from ..paths import BookPaths
from ..synthesis import load_cached_pairs, synthesize
from ..thematic import tree_identity
from ..training import run_comparison
from ..utils import read_json, write_json, write_jsonl
from .context import (
    adapter_identity,
    adapter_llm_or_fail,
    apply_overrides,
    build_ablation_runs,
    build_global_pipeline,
    build_index_or_fail,
    build_pipeline,
    build_training_runs,
    build_tree_or_fail,
    check_training_runtime_or_fail,
    gguf_identity,
    gguf_llm_or_fail,
    ingest_or_fail,
    load_book_and_chunks,
    load_book_index,
    load_golden_set,
    load_registered_adapter_or_fail,
    load_registered_gguf_or_fail,
    load_thematic_manifest_or_fail,
    load_thematic_tree_or_fail,
    load_training_examples,
    parse_top_k_values,
    prepare_training_or_fail,
    register_adapter_or_fail,
    register_gguf_or_fail,
)
from .render import (
    console,
    render_ablation,
    render_answer,
    render_books,
    render_comparison,
    render_eval,
    render_gguf_registration,
    render_gguf_serving,
    render_index,
    render_info,
    render_ingest,
    render_ragas,
    render_registration,
    render_runtime,
    render_synthesis,
    render_training,
    render_tree_build,
)

if TYPE_CHECKING:
    from ..evaluation import GoldenItem
    from ..llm import LLMClient
    from ..models import Answer
    from ..rag import QAPipeline


class AnsweringPipeline(Protocol):
    """Minimal interface shared by the local and global pipelines."""

    def ask(self, question: str) -> Answer:
        """Answer one question about the book."""
        ...  # pragma: no cover - protocol stub


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
    gguf: Annotated[
        bool,
        typer.Option("--gguf", help="Answer with the registered GGUF endpoint."),
    ] = False,
    global_mode: Annotated[
        bool,
        typer.Option(
            "--global",
            help="Answer from the thematic summary tree (whole-book questions).",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the full Answer object as JSON."),
    ] = False,
) -> None:
    """Ask a question about an indexed book; answers carry citations."""
    settings = settings_from_context(ctx)
    if adapter and gguf:
        raise typer.BadParameter("Choose one generator: --adapter or --gguf, not both.")
    if global_mode:
        reject_local_only_flags(top_k=top_k, chapter=chapter, rerank=rerank)

    question_text = " ".join(question).strip()
    if global_mode:
        llm = generator_llm_or_fail(settings, book_id, adapter=adapter, gguf=gguf)
        tree = load_thematic_tree_or_fail(settings, book_id)
        pipeline = build_global_pipeline(settings, tree, llm=llm)
        book_title = pipeline.book_title
        with console.status("Thinking..."):
            try:
                answer = pipeline.ask(question_text)
            except DistillerError as exc:
                raise typer.BadParameter(str(exc)) from exc
    else:
        local_pipeline, book_title = build_local_pipeline(
            settings, book_id, rerank=rerank, adapter=adapter, gguf=gguf
        )
        with console.status("Thinking..."):
            try:
                answer = local_pipeline.ask(question_text, chapter=chapter, top_k=top_k)
            except DistillerError as exc:
                raise typer.BadParameter(str(exc)) from exc

    if as_json:
        typer.echo(answer.model_dump_json(indent=2))
        return
    render_answer(answer, book_title)


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
    gguf: Annotated[
        bool,
        typer.Option("--gguf", help="Evaluate with the registered GGUF endpoint."),
    ] = False,
    global_mode: Annotated[
        bool,
        typer.Option(
            "--global",
            help="Evaluate the golden items through the thematic summary tree.",
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
    if adapter and gguf:
        raise typer.BadParameter("Choose one generator: --adapter or --gguf, not both.")
    if global_mode and with_ragas:
        raise typer.BadParameter(
            "--ragas is local-only and cannot be combined with --global."
        )
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    golden_path = golden or paths.golden_yaml
    if not golden_path.exists():
        raise typer.BadParameter(
            f"Golden set not found: {golden_path} (create one or pass --golden)"
        )

    golden_items = load_golden_set(golden_path)
    if limit:
        golden_items = golden_items[:limit]

    adapter_provenance: dict[str, Any] | None = None
    gguf_provenance: dict[str, Any] | None = None
    pipeline: AnsweringPipeline
    if global_mode:
        llm, adapter_provenance, gguf_provenance = generator_override_or_fail(
            settings, book_id, adapter=adapter, gguf=gguf
        )
        tree = load_thematic_tree_or_fail(settings, book_id)
        manifest = load_thematic_manifest_or_fail(settings, book_id)
        pipeline = build_global_pipeline(settings, tree, llm=llm)
        retrieval: dict[str, Any] = {
            "mode": "global",
            "tree": tree_identity(manifest),
        }
    else:
        bundle = load_book_index(settings, book_id)
        llm, adapter_provenance, gguf_provenance = generator_override_or_fail(
            settings, book_id, adapter=adapter, gguf=gguf
        )
        pipeline = build_pipeline(settings, bundle, llm=llm)
        retrieval = {"mode": "local", "tree": None}

    try:
        results, samples = run_golden_set(pipeline, golden_items)
    except DistillerError as exc:
        raise typer.BadParameter(str(exc)) from exc

    index_metadata = (
        read_json(paths.index_metadata) if paths.index_metadata.exists() else {}
    )

    kind = "adapter" if adapter else "gguf" if gguf else "base"
    generator: dict[str, Any] = {
        "kind": kind,
        "model": answering_model_name(pipeline),
        "adapter": adapter_provenance,
        "gguf": gguf_provenance,
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
        "retrieval": retrieval,
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
    gguf: Annotated[
        bool,
        typer.Option("--gguf", help="Also compare the registered GGUF export."),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full report as JSON.")
    ] = False,
) -> None:
    """Compare the base model, the registered adapter and the GGUF export."""
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
    gguf_llm: LLMClient | None = None
    gguf_provenance: dict[str, Any] | None = None
    if gguf:
        gguf_report, gguf_dir = load_registered_gguf_or_fail(settings, book_id)
        gguf_llm = gguf_llm_or_fail(settings, gguf_report)
        gguf_provenance = gguf_identity(gguf_report, gguf_dir)
    runs = build_training_runs(
        settings,
        bundle,
        adapter_llm,
        adapter_identity=adapter_identity(adapter_report, adapter_dir),
        gguf_llm=gguf_llm,
        gguf_identity=gguf_provenance,
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


tree_app = typer.Typer(
    no_args_is_help=True,
    help="Build the thematic summary tree for whole-book questions (Phase 5).",
)
app.add_typer(tree_app, name="tree")


@tree_app.command("build")
def tree_build(
    ctx: typer.Context,
    book_id: Annotated[str, typer.Argument(help="Book id.")],
    window_size: Annotated[
        int | None,
        typer.Option("--window-size", help="Consecutive chapters per level-2 window."),
    ] = None,
    regenerate: Annotated[
        bool,
        typer.Option(
            "--regenerate", help="Ignore cached summaries and call the LLM again."
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the manifest as JSON.")
    ] = False,
) -> None:
    """Summarize chapters, windows and the whole book into thematic/tree.json."""
    settings = settings_from_context(ctx)
    if window_size is not None:
        settings.thematic = apply_overrides(settings.thematic, window_size=window_size)

    book, chunks = load_book_and_chunks(settings, book_id)
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)

    with console.status("Building the summary tree..."):
        run = build_tree_or_fail(
            book,
            chunks,
            get_llm(settings),
            settings.thematic,
            cache_path=paths.thematic_summaries_jsonl,
            regenerate=regenerate,
        )

    paths.thematic_dir.mkdir(parents=True, exist_ok=True)
    write_json(paths.thematic_tree_json, run.tree.model_dump())
    write_json(paths.thematic_manifest_json, run.manifest.model_dump())

    if as_json:
        typer.echo(run.manifest.model_dump_json(indent=2))
        return
    render_tree_build(run.manifest, paths.thematic_dir)
    console.print(f"Thematic tree written to [cyan]{paths.thematic_dir}[/cyan]")


gguf_app = typer.Typer(
    no_args_is_help=True,
    help="Validate, register and serve a GGUF export (Phase 4).",
)
app.add_typer(gguf_app, name="gguf")


@gguf_app.command("register")
def gguf_register(
    ctx: typer.Context,
    book_id: Annotated[str, typer.Argument(help="Book id.")],
    file: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Downloaded .gguf file."
        ),
    ],
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the registration report as JSON.")
    ] = False,
) -> None:
    """Validate a GGUF export, copy it into training/gguf/ and emit serving files."""
    settings = settings_from_context(ctx)
    report, gguf_dir = register_gguf_or_fail(settings, book_id, file)

    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    render_gguf_registration(report, gguf_dir)
    console.print(f"GGUF registered at [cyan]{gguf_dir}[/cyan]")


@gguf_app.command("serve")
def gguf_serve(
    ctx: typer.Context,
    book_id: Annotated[str, typer.Argument(help="Book id.")],
) -> None:
    """Print the serving commands for a registered GGUF (launches nothing)."""
    settings = settings_from_context(ctx)
    report, gguf_dir = load_registered_gguf_or_fail(settings, book_id)
    render_gguf_serving(report, gguf_dir)


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


def reject_local_only_flags(
    *,
    top_k: int | None,
    chapter: str | None,
    rerank: bool,
) -> None:
    """Reject local retrieval flags when ``--global`` selects the thematic mode.

    Args:
        top_k: ``--top-k`` value, when given.
        chapter: ``--chapter`` value, when given.
        rerank: Whether ``--rerank`` was passed.

    Raises:
        typer.BadParameter: For any flag that only applies to local retrieval.
    """
    if top_k is not None:
        raise typer.BadParameter(
            "--top-k is local-only and cannot be combined with --global."
        )
    if chapter is not None:
        raise typer.BadParameter(
            "--chapter is local-only and cannot be combined with --global."
        )
    if rerank:
        raise typer.BadParameter(
            "--rerank is local-only and cannot be combined with --global."
        )


def build_local_pipeline(
    settings: Settings,
    book_id: str,
    *,
    rerank: bool,
    adapter: bool,
    gguf: bool,
) -> tuple[QAPipeline, str]:
    """Assemble the local retrieval pipeline and its book title.

    Args:
        settings: Pipeline settings.
        book_id: Book to load the index for.
        rerank: Force-enable cross-encoder reranking for this run.
        adapter: Answer with the registered adapter endpoint.
        gguf: Answer with the registered GGUF endpoint.

    Returns:
        The local pipeline and the book title for rendering.
    """
    bundle = load_book_index(settings, book_id)
    llm = generator_llm_or_fail(settings, book_id, adapter=adapter, gguf=gguf)
    pipeline = build_pipeline(settings, bundle, rerank=rerank, llm=llm)
    return pipeline, bundle.book.title


def generator_override_or_fail(
    settings: Settings,
    book_id: str,
    *,
    adapter: bool,
    gguf: bool,
) -> tuple[LLMClient | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Resolve the optional adapter/GGUF generator and its provenance.

    Args:
        settings: Pipeline settings.
        book_id: Book whose registration must exist for ``--adapter``/``--gguf``.
        adapter: Answer with the registered adapter endpoint.
        gguf: Answer with the registered GGUF endpoint.

    Returns:
        ``(llm, adapter_provenance, gguf_provenance)``; the client is None when
        no override was requested.
    """
    if adapter:
        adapter_report, adapter_dir = load_registered_adapter_or_fail(settings, book_id)
        return (
            adapter_llm_or_fail(settings),
            adapter_identity(adapter_report, adapter_dir),
            None,
        )
    if gguf:
        gguf_report, gguf_dir = load_registered_gguf_or_fail(settings, book_id)
        return (
            gguf_llm_or_fail(settings, gguf_report),
            None,
            gguf_identity(gguf_report, gguf_dir),
        )
    return None, None, None


def generator_llm_or_fail(
    settings: Settings,
    book_id: str,
    *,
    adapter: bool,
    gguf: bool,
) -> LLMClient | None:
    """Resolve the optional adapter/GGUF generator override, or a friendly error.

    Args:
        settings: Pipeline settings.
        book_id: Book whose registration must exist for ``--adapter``/``--gguf``.
        adapter: Answer with the registered adapter endpoint.
        gguf: Answer with the registered GGUF endpoint.

    Returns:
        The override client, or None to use the configured base LLM.
    """
    llm, _, _ = generator_override_or_fail(
        settings, book_id, adapter=adapter, gguf=gguf
    )
    return llm


def answering_model_name(pipeline: AnsweringPipeline) -> str:
    """Return the answering model identifier for the eval report.

    Works for both the local pipeline (``generator.llm``) and the global
    pipeline (``llm``).
    """
    llm = getattr(pipeline, "llm", None)
    if llm is None:
        llm = getattr(getattr(pipeline, "generator", None), "llm", None)
    return str(getattr(llm, "name", "unknown"))


def run_golden_set(
    pipeline: AnsweringPipeline,
    golden_items: list[GoldenItem],
) -> tuple[list[ItemResult], list[dict[str, Any]]]:
    """Ask every golden question and collect results plus RAGAS samples.

    Args:
        pipeline: Assembled local or global pipeline for the book.
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
