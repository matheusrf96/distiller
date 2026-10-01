"""Shared CLI helpers: validated settings overrides and pipeline construction."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any

import typer
from pydantic import BaseModel, ValidationError

from ..evaluation import (
    AblationRun,
    AblationVariant,
    load_golden,
)
from ..exceptions import DistillerError, ThematicError
from ..gguf import load_gguf_report, register_gguf
from ..indexing import IndexBundle, build_index, load_index
from ..ingest import ingest_book
from ..llm import get_adapter_llm, get_gguf_llm, get_llm
from ..models import BookDocument, Chunk
from ..optional_deps import is_available
from ..paths import BookPaths
from ..rag import Generator, QAPipeline, Retriever, get_reranker
from ..thematic import (
    GlobalPipeline,
    SummaryTree,
    TreeManifest,
    TreeRun,
    build_tree,
    load_manifest,
    load_tree,
)
from ..training import (
    GeneratorVariant,
    QLoRAConfig,
    TrainingBundle,
    TrainingReport,
    TrainingRun,
    load_adapter_report,
    load_raft_examples,
    prepare_dataset,
    register_adapter,
    training_stack_versions,
)
from ..utils import read_json, read_jsonl

if TYPE_CHECKING:
    from pathlib import Path

    from ..config import Settings, ThematicSettings
    from ..evaluation import GoldenItem
    from ..gguf import GgufReport
    from ..llm import LLMClient
    from ..synthesis import RaftExample

logger = logging.getLogger(__name__)


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
    settings: Settings,
    bundle: IndexBundle,
    *,
    rerank: bool = False,
    llm: LLMClient | None = None,
) -> QAPipeline:
    """Assemble the retrieval -> (rerank) -> generation pipeline for one book.

    Args:
        settings: Pipeline settings.
        bundle: Loaded index bundle for the book.
        rerank: Force-enable cross-encoder reranking for this run.
        llm: Generator client override (the served adapter); defaults to the
            configured base LLM.

    Returns:
        Pipeline that answers questions with citations.
    """
    retriever = Retriever(bundle, settings.retrieval)
    generator = Generator(
        llm or get_llm(settings),
        bundle.book.title,
        max_tokens=settings.llm.max_tokens,
    )
    reranker = None
    if rerank or settings.retrieval.rerank:
        reranker = get_reranker(apply_overrides(settings.retrieval, rerank=True))
    return QAPipeline(
        retriever, generator, reranker=reranker, settings=settings.retrieval
    )


def pipeline_model_name(pipeline: QAPipeline) -> str:
    """Return the generator's model identifier for the reports."""
    return getattr(pipeline.generator.llm, "name", "unknown")


def adapter_llm_or_fail(settings: Settings) -> LLMClient:
    """Return the configured adapter client, or a friendly CLI error.

    Args:
        settings: Pipeline settings (adapter endpoint).

    Returns:
        Client pointed at the served LoRA adapter.

    Raises:
        typer.BadParameter: When no adapter endpoint is configured.
    """
    try:
        return get_adapter_llm(settings)
    except DistillerError as exc:
        raise typer.BadParameter(str(exc)) from exc


def load_registered_adapter_or_fail(
    settings: Settings, book_id: str
) -> tuple[TrainingReport, Path]:
    """Load the registered adapter report, or a friendly CLI error.

    Args:
        settings: Pipeline settings.
        book_id: Book whose adapter registry should be read.

    Returns:
        The validated training report and the adapter directory.

    Raises:
        typer.BadParameter: When no adapter is registered or its run.json is
            invalid.
    """
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    if not paths.adapter_run_json.exists():
        raise typer.BadParameter(
            f"No registered adapter for '{book_id}'. "
            f"Run `distiller train {book_id} --register <dir>` first."
        )
    try:
        return load_adapter_report(paths.adapter_dir), paths.adapter_dir
    except DistillerError as exc:
        raise typer.BadParameter(str(exc)) from exc


def adapter_identity(report: TrainingReport, adapter_dir: Path) -> dict[str, Any]:
    """Build the adapter provenance block recorded in eval reports.

    Args:
        report: Registered adapter training report.
        adapter_dir: Registry directory of the adapter.

    Returns:
        JSON-ready provenance: path, base model and dataset/config hashes.
    """
    return {
        "path": str(adapter_dir),
        "base_model": report.base_model,
        "dataset_hash": report.dataset_hash,
        "config_hash": report.config_hash,
        "trained_at": report.created_at.isoformat(),
    }


def load_registered_gguf_or_fail(
    settings: Settings, book_id: str
) -> tuple[GgufReport, Path]:
    """Load the registered GGUF report, or a friendly CLI error.

    Args:
        settings: Pipeline settings.
        book_id: Book whose GGUF registry should be read.

    Returns:
        The validated GGUF report and the registry directory.

    Raises:
        typer.BadParameter: When no GGUF is registered, its report is invalid
            or the emitted Modelfile/serve.sh are missing.
    """
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    if not paths.gguf_report_json.exists():
        raise typer.BadParameter(
            f"No registered GGUF for '{book_id}'. "
            f"Run `distiller gguf register {book_id} <file.gguf>` first."
        )
    if not paths.gguf_modelfile.exists() or not paths.gguf_serve_script.exists():
        raise typer.BadParameter(
            f"Registered GGUF for '{book_id}' is incomplete "
            f"(Modelfile/serve.sh missing). "
            f"Re-run `distiller gguf register {book_id} <file.gguf>`."
        )
    try:
        return load_gguf_report(paths.gguf_report_json), paths.gguf_dir
    except DistillerError as exc:
        raise typer.BadParameter(str(exc)) from exc


def gguf_llm_or_fail(settings: Settings, report: GgufReport) -> LLMClient:
    """Return the configured GGUF client, or a friendly CLI error.

    Args:
        settings: Pipeline settings (GGUF endpoint).
        report: Registered GGUF report providing the default model name.

    Returns:
        Client pointed at the served GGUF.

    Raises:
        typer.BadParameter: When no endpoint is configured.
    """
    model_name = settings.gguf.model or report.model_name
    try:
        return get_gguf_llm(settings.gguf, model_name)
    except DistillerError as exc:
        raise typer.BadParameter(str(exc)) from exc


def gguf_identity(report: GgufReport, gguf_dir: Path) -> dict[str, Any]:
    """Build the GGUF provenance block recorded in eval reports.

    Args:
        report: Registered GGUF report.
        gguf_dir: Registry directory of the GGUF.

    Returns:
        JSON-ready provenance: path, file hash, quantization, architecture and
        served model name.
    """
    return {
        "path": str(gguf_dir),
        "model_name": report.model_name,
        "sha256": report.sha256,
        "quantization": report.metadata.quantization,
        "architecture": report.metadata.architecture,
    }


def register_gguf_or_fail(
    settings: Settings, book_id: str, source: Path
) -> tuple[GgufReport, Path]:
    """Validate and register a downloaded GGUF, or a friendly CLI error.

    Args:
        settings: Pipeline settings.
        book_id: Book the GGUF was exported for.
        source: GGUF file downloaded from the training machine.

    Returns:
        The registration report and the registry directory.

    Raises:
        typer.BadParameter: When the book is not ingested or the file is
            invalid (nothing is copied in that case).
    """
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    if not paths.book_json.exists():
        raise typer.BadParameter(
            f"No ingested book for id '{book_id}'. Run `distiller ingest` first."
        )
    try:
        book = BookDocument.model_validate(read_json(paths.book_json))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(
            f"Artifacts for '{book_id}' are incomplete or corrupt: {exc}"
        ) from exc
    try:
        report = register_gguf(
            source,
            paths.gguf_dir,
            book_id=book_id,
            book_title=book.title,
            model_name=settings.gguf.model,
            adapter=_registered_adapter_identity(paths),
        )
    except DistillerError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return report, paths.gguf_dir


def build_training_runs(
    settings: Settings,
    bundle: IndexBundle,
    adapter_llm: LLMClient,
    *,
    adapter_identity: dict[str, Any] | None = None,
    gguf_llm: LLMClient | None = None,
    gguf_identity: dict[str, Any] | None = None,
) -> list[TrainingRun]:
    """Pair the base, adapter and (optionally) GGUF generators with pipelines.

    Args:
        settings: Pipeline settings (retrieval is shared by all variants).
        bundle: Loaded index, reused by all variants.
        adapter_llm: Client for the served LoRA adapter.
        adapter_identity: Provenance recorded on the adapter variant.
        gguf_llm: Optional client for the served GGUF export.
        gguf_identity: Provenance recorded on the GGUF variant.

    Returns:
        The base run first, then adapter, then GGUF when configured.
    """
    base_pipeline = build_pipeline(settings, bundle)
    adapter_pipeline = build_pipeline(settings, bundle, llm=adapter_llm)
    runs = [
        TrainingRun(
            variant=GeneratorVariant(
                name="base",
                kind="base",
                model=pipeline_model_name(base_pipeline),
            ),
            answer=partial(base_pipeline.ask),
        ),
        TrainingRun(
            variant=GeneratorVariant(
                name="adapter",
                kind="adapter",
                model=adapter_llm.name,
                adapter=dict(adapter_identity or {}),
            ),
            answer=partial(adapter_pipeline.ask),
        ),
    ]
    if gguf_llm is not None:
        gguf_pipeline = build_pipeline(settings, bundle, llm=gguf_llm)
        runs.append(
            TrainingRun(
                variant=GeneratorVariant(
                    name="gguf",
                    kind="gguf",
                    model=gguf_llm.name,
                    gguf=dict(gguf_identity or {}),
                ),
                answer=partial(gguf_pipeline.ask),
            )
        )
    return runs


def _registered_adapter_identity(paths: BookPaths) -> dict[str, Any]:
    """Adapter provenance for the GGUF report, empty when unavailable."""
    if not paths.adapter_run_json.exists():
        return {}
    try:
        report = load_adapter_report(paths.adapter_dir)
    except DistillerError as exc:
        logger.warning(
            "Ignoring unreadable adapter registry %s: %s", paths.adapter_dir, exc
        )
        return {}
    return adapter_identity(report, paths.adapter_dir)


def load_training_examples(settings: Settings, book_id: str) -> list[RaftExample]:
    """Load ``dataset/raft.jsonl``, or a friendly CLI error.

    Args:
        settings: Pipeline settings.
        book_id: Book whose RAFT dataset should be loaded.

    Returns:
        Validated RAFT examples.

    Raises:
        typer.BadParameter: When the dataset is missing or corrupt.
    """
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    if not paths.dataset_raft_jsonl.exists():
        raise typer.BadParameter(
            f"No RAFT dataset for '{book_id}'. Run `distiller synth {book_id}` first."
        )
    try:
        return load_raft_examples(paths.dataset_raft_jsonl)
    except DistillerError as exc:
        raise typer.BadParameter(str(exc)) from exc


def prepare_training_or_fail(
    book: BookDocument,
    examples: list[RaftExample],
    chunks: list[Chunk],
    settings: Settings,
    paths: BookPaths,
) -> TrainingBundle:
    """Prepare and write the training dataset, or a friendly CLI error.

    Args:
        book: Parsed book.
        examples: RAFT examples from ``dataset/raft.jsonl``.
        chunks: Indexed chunks.
        settings: Pipeline settings (training seed and ratio).
        paths: Book artifact layout to write into.

    Returns:
        The prepared training bundle.

    Raises:
        typer.BadParameter: On formatting or validation failures.
    """
    try:
        return prepare_dataset(book, examples, chunks, settings.training, paths=paths)
    except DistillerError as exc:
        raise typer.BadParameter(str(exc)) from exc


def register_adapter_or_fail(
    settings: Settings, book_id: str, source_dir: Path
) -> tuple[TrainingReport, Path]:
    """Validate and register a T4 adapter directory, or a friendly CLI error.

    Args:
        settings: Pipeline settings.
        book_id: Book the adapter must belong to.
        source_dir: Adapter directory downloaded from the T4 run.

    Returns:
        The registered report and the registry directory.

    Raises:
        typer.BadParameter: When training artifacts are missing or the adapter
            fails validation.
    """
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    if not paths.training_qlora_json.exists():
        raise typer.BadParameter(
            f"No QLoRA configuration for '{book_id}'. "
            f"Run `distiller train {book_id}` first."
        )
    try:
        config = QLoRAConfig.model_validate(read_json(paths.training_qlora_json))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(
            f"Training configuration {paths.training_qlora_json} is unreadable: "
            f"{exc}. Re-run `distiller train {book_id}`."
        ) from exc
    try:
        report = register_adapter(
            source_dir,
            paths.adapter_dir,
            book_id=book_id,
            expected_base_model=config.base_model,
            expected_dataset_hash=_expected_dataset_hash(paths),
        )
    except DistillerError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return report, paths.adapter_dir


def check_training_runtime_or_fail() -> dict[str, str]:
    """Report the installed training-stack versions, or the missing-extra error.

    Returns:
        Mapping module name -> version.

    Raises:
        typer.BadParameter: When the ``training`` extra is not installed.
    """
    try:
        return training_stack_versions()
    except DistillerError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _expected_dataset_hash(paths: BookPaths) -> str | None:
    if not paths.training_manifest.exists():
        return None
    try:
        value = read_json(paths.training_manifest).get("dataset_hash")
    except (OSError, ValueError) as exc:
        logger.warning(
            "Ignoring unreadable training manifest %s: %s",
            paths.training_manifest,
            exc,
        )
        return None
    return str(value) if value else None


def load_book_and_chunks(
    settings: Settings, book_id: str
) -> tuple[BookDocument, list[Chunk]]:
    """Load the parsed book and its indexed chunks (no embedder needed).

    Synthesis works from `chunks.jsonl`, so it runs on a plain install and is
    unaffected by embedder choices.

    Args:
        settings: Pipeline settings.
        book_id: Slug of the book.

    Returns:
        The parsed book and all of its chunks.

    Raises:
        typer.BadParameter: When the book was never ingested or indexed, or the
            artifacts are unreadable.
    """
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    if not paths.book_json.exists():
        raise typer.BadParameter(
            f"No ingested book for id '{book_id}'. Run `distiller ingest` first."
        )
    if not paths.chunks_jsonl.exists():
        raise typer.BadParameter(
            f"No chunks for book '{book_id}'. Run `distiller index {book_id}` first."
        )
    try:
        book = BookDocument.model_validate(read_json(paths.book_json))
        chunks = [Chunk.model_validate(row) for row in read_jsonl(paths.chunks_jsonl)]
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(
            f"Artifacts for '{book_id}' are incomplete or corrupt: {exc}"
        ) from exc
    return book, chunks


def build_tree_or_fail(
    book: BookDocument,
    chunks: list[Chunk],
    llm: LLMClient,
    settings: ThematicSettings,
    *,
    cache_path: Path | None,
    regenerate: bool,
) -> TreeRun:
    """Build the thematic summary tree, turning domain errors into CLI errors.

    Args:
        book: Parsed book.
        chunks: Indexed chunks of the book.
        llm: Chat client used for summarization.
        settings: Thematic settings (window size and character budgets).
        cache_path: Optional ``summaries.jsonl`` cache (read and appended).
        regenerate: Ignore cached summaries and call the LLM again.

    Returns:
        The built tree plus its manifest.

    Raises:
        typer.BadParameter: When the book cannot be summarized (no chapters,
            no chunks, or every chapter summary failed).
    """
    try:
        return build_tree(
            book,
            chunks,
            llm,
            settings,
            cache_path=cache_path,
            regenerate=regenerate,
        )
    except ThematicError as exc:
        raise typer.BadParameter(str(exc)) from exc


def load_thematic_tree_or_fail(settings: Settings, book_id: str) -> SummaryTree:
    """Load ``thematic/tree.json``, or a friendly CLI error.

    Args:
        settings: Pipeline settings.
        book_id: Book whose tree should be loaded.

    Returns:
        The validated summary tree.

    Raises:
        typer.BadParameter: When the tree is missing or corrupt; the message
            names ``distiller tree build``.
    """
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    try:
        return load_tree(paths.thematic_tree_json)
    except ThematicError as exc:
        raise typer.BadParameter(str(exc)) from exc


def load_thematic_manifest_or_fail(settings: Settings, book_id: str) -> TreeManifest:
    """Load ``thematic/manifest.json``, or a friendly CLI error.

    Args:
        settings: Pipeline settings.
        book_id: Book whose manifest should be loaded.

    Returns:
        The validated tree manifest.

    Raises:
        typer.BadParameter: When the manifest is missing or corrupt; the
            message names ``distiller tree build``.
    """
    paths = BookPaths.for_book(settings.artifacts_dir, book_id)
    try:
        return load_manifest(paths.thematic_manifest_json)
    except ThematicError as exc:
        raise typer.BadParameter(str(exc)) from exc


def build_global_pipeline(
    settings: Settings,
    tree: SummaryTree,
    *,
    llm: LLMClient | None = None,
) -> GlobalPipeline:
    """Assemble the global (map-reduce) pipeline for one book.

    Selection uses the index embedder when the index is loadable; a book
    without one still answers globally by mapping every chapter summary in
    reading order.

    Args:
        settings: Pipeline settings.
        tree: Summary tree built by ``distiller tree build``.
        llm: Generator client override (the served adapter/GGUF); defaults to
            the configured base LLM.

    Returns:
        Pipeline that answers whole-book questions from tree summaries.
    """
    embedder = None
    try:
        embedder = load_index(tree.book_id, settings).embedder
    except DistillerError as exc:
        logger.debug("Global selection without an index embedder: %s", exc)
    return GlobalPipeline(
        tree,
        llm or get_llm(settings),
        settings.thematic,
        embedder=embedder,
        max_tokens=settings.llm.max_tokens,
    )


def build_ablation_runs(
    settings: Settings,
    bundle: IndexBundle,
    variants: list[AblationVariant],
) -> list[AblationRun]:
    """Pair each variant with an answering pipeline, skipping unavailable rerankers.

    Args:
        settings: Pipeline settings.
        bundle: Loaded index, reused by every variant.
        variants: Variants to run.

    Returns:
        Runs in input order; rerank variants are skipped (with an actionable
        reason) when the ``embed`` extra is not installed.
    """
    runs: list[AblationRun] = []
    for variant in variants:
        if variant.rerank and not is_available("sentence_transformers"):
            runs.append(
                AblationRun(
                    variant=variant,
                    skipped_reason=(
                        "reranking needs the 'embed' extra (uv sync --extra embed)"
                    ),
                )
            )
            continue

        retrieval = apply_overrides(
            settings.retrieval,
            rerank=variant.rerank,
            top_k_final=variant.top_k_final,
            rerank_pool=variant.rerank_pool,
        )
        retriever = Retriever(bundle, retrieval)
        generator = Generator(
            get_llm(settings),
            bundle.book.title,
            max_tokens=settings.llm.max_tokens,
        )
        reranker = get_reranker(retrieval) if variant.rerank else None
        pipeline = QAPipeline(
            retriever, generator, reranker=reranker, settings=retrieval
        )
        runs.append(AblationRun(variant=variant, answer=partial(pipeline.ask)))
    return runs


def parse_top_k_values(raw: str | None) -> list[int] | None:
    """Parse a comma-separated top-k sweep such as ``4,8,12``.

    Args:
        raw: Raw CLI value; None means "no sweep".

    Returns:
        Positive integers in input order, or None when no sweep was requested.

    Raises:
        typer.BadParameter: When the value is empty, malformed or non-positive.
    """
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise typer.BadParameter(
            f"Invalid --top-k value: {raw!r} (expected e.g. 4,8,12)"
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise typer.BadParameter(
            f"Invalid --top-k value: {raw!r} (values must be positive)"
        )
    return values


def _first_error(exc: ValidationError) -> str:
    """Format the first validation error as ``location: message``."""
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}" if location else str(error["msg"])
