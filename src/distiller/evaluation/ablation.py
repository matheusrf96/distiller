"""Retrieval ablation harness: compare configurations on one golden set."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..models import DomainModel
from .metrics import (
    ItemResult,
    evaluate_item,
    metric_deltas_against,
    summarize,
)

if TYPE_CHECKING:
    from ..models import Answer
    from .golden import GoldenItem

AnswerFn = Callable[[str], "Answer"]


class AblationVariant(DomainModel):
    """One retrieval configuration under test.

    Attributes:
        name: Short variant name used in reports and tables.
        rerank: Whether cross-encoder reranking is enabled.
        top_k_final: Number of chunks handed to the generator.
        rerank_pool: Candidate pool fed to the reranker when reranking.
    """

    name: str
    rerank: bool = False
    top_k_final: int = 8
    rerank_pool: int = 40


@dataclass(frozen=True, slots=True)
class AblationRun:
    """A variant paired with the callable that answers for it (runtime only).

    Attributes:
        variant: Configuration under test.
        answer: Callable answering one question, or None when the variant is skipped.
        skipped_reason: Why the variant will not run, when applicable.
    """

    variant: AblationVariant
    answer: AnswerFn | None = None
    skipped_reason: str | None = None


class VariantResult(DomainModel):
    """Metrics (or a skip reason) for one variant.

    Attributes:
        variant: The configuration that was run.
        skipped_reason: Why the variant did not run, when applicable.
        metrics: Aggregated metrics from ``summarize``.
        items: Per-question results.
    """

    variant: AblationVariant
    skipped_reason: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    items: list[ItemResult] = Field(default_factory=list)


class AblationReport(DomainModel):
    """Full ablation outcome for one book and golden set.

    Attributes:
        book_id: Book the ablation ran against.
        index: Index identity (embedder, contextual flag, ...).
        item_count: Number of golden items evaluated per variant.
        baseline: Name of the first variant that produced metrics.
        variants: Per-variant results, in run order.
    """

    book_id: str
    index: dict[str, Any] = Field(default_factory=dict)
    item_count: int = 0
    baseline: str | None = None
    variants: list[VariantResult] = Field(default_factory=list)


def build_variants(
    *,
    top_k_final: int,
    rerank_pool: int,
    top_k_values: list[int] | None = None,
    include_rerank: bool = True,
) -> list[AblationVariant]:
    """Build the variant matrix: reranking off/on crossed with top-k values.

    Args:
        top_k_final: Default number of chunks handed to the generator.
        rerank_pool: Candidate pool size for reranking variants.
        top_k_values: Optional sweep of top-k values; defaults to the base value.
        include_rerank: Include reranked variants (False for ``--no-rerank``).

    Returns:
        Variants with the non-reranked baseline first.
    """
    values = top_k_values or [top_k_final]
    labelled = len(values) > 1 or values[0] != top_k_final
    variants: list[AblationVariant] = []
    for value in values:
        suffix = f"-k{value}" if labelled else ""
        variants.append(
            AblationVariant(
                name=f"hybrid{suffix}",
                rerank=False,
                top_k_final=value,
                rerank_pool=rerank_pool,
            )
        )
        if include_rerank:
            variants.append(
                AblationVariant(
                    name=f"hybrid+rerank{suffix}",
                    rerank=True,
                    top_k_final=value,
                    rerank_pool=rerank_pool,
                )
            )
    return variants


def run_ablation(
    book_id: str,
    golden_items: list[GoldenItem],
    runs: list[AblationRun],
    *,
    index_identity: dict[str, Any] | None = None,
) -> AblationReport:
    """Evaluate every run against the golden set and assemble the report.

    Args:
        book_id: Book the ablation runs against.
        golden_items: Golden questions (the same subset for every variant).
        runs: Variants paired with their answer callables (or skip reasons).
        index_identity: Index identity recorded in the report.

    Returns:
        Report with per-variant metrics, per-item results and the baseline name.
    """
    results: list[VariantResult] = []
    for run in runs:
        if run.answer is None:
            results.append(
                VariantResult(
                    variant=run.variant,
                    skipped_reason=run.skipped_reason or "not run",
                )
            )
            continue
        try:
            items = [
                evaluate_item(item, run.answer(item.question)) for item in golden_items
            ]
        except Exception as exc:
            results.append(
                VariantResult(
                    variant=run.variant,
                    skipped_reason=f"failed: {type(exc).__name__}: {exc}",
                )
            )
            continue
        results.append(
            VariantResult(variant=run.variant, metrics=summarize(items), items=items)
        )

    baseline = next(
        (result.variant.name for result in results if result.skipped_reason is None),
        None,
    )
    return AblationReport(
        book_id=book_id,
        index=dict(index_identity or {}),
        item_count=len(golden_items),
        baseline=baseline,
        variants=results,
    )


def metric_deltas(
    report: AblationReport, metrics: tuple[str, ...]
) -> dict[str, dict[str, float | None]]:
    """Per-variant deltas of selected metrics versus the baseline variant.

    Args:
        report: Ablation report to summarise.
        metrics: Metric names to diff (keys of ``summarize`` output).

    Returns:
        Mapping variant name -> metric -> delta versus the baseline. Skipped
        variants are excluded; a delta is None when either side is missing.
    """
    baseline = next(
        (
            result
            for result in report.variants
            if result.variant.name == report.baseline and result.skipped_reason is None
        ),
        None,
    )
    if baseline is None:
        return {}
    return metric_deltas_against(
        baseline.metrics,
        (
            (result.variant.name, result.metrics)
            for result in report.variants
            if result.skipped_reason is None
        ),
        metrics,
    )
