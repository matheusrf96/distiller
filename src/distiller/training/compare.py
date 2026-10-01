"""Base-vs-adapter comparison over one index and golden set (Phase 3, step 4).

Only the generator changes between variants: the report records one shared
``index`` + ``retrieval`` identity and one ``generator`` identity per variant,
so a reader can verify that fine-tuned-student + RAG and base + RAG differ in
exactly one variable. Deltas use the same helper as the retrieval ablation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from ..evaluation import (
    ItemResult,
    evaluate_item,
    metric_deltas_against,
    summarize,
)
from ..models import DomainModel

if TYPE_CHECKING:
    from ..evaluation import GoldenItem
    from ..models import Answer

AnswerFn = Callable[[str], "Answer"]

COMPARISON_METRICS = (
    "retrieval_hit_rate",
    "contains_rate",
    "refusal_accuracy",
    "citation_coverage",
)

__all__ = [
    "COMPARISON_METRICS",
    "GeneratorResult",
    "GeneratorVariant",
    "TrainingComparison",
    "TrainingRun",
    "run_comparison",
]


class GeneratorVariant(DomainModel):
    """One generator under test (the only variable in the comparison).

    Attributes:
        name: Short variant name used in reports and tables.
        kind: ``base`` for the unmodified model, ``adapter`` for the fine-tune.
        model: Model identifier reported by the client.
        adapter: Adapter provenance (run.json identity), empty for the base.
    """

    name: str
    kind: Literal["base", "adapter"]
    model: str
    adapter: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """A generator paired with the callable that answers for it (runtime only).

    Attributes:
        variant: Generator under test.
        answer: Callable answering one question, or None when the variant is skipped.
        skipped_reason: Why the variant will not run, when applicable.
    """

    variant: GeneratorVariant
    answer: AnswerFn | None = None
    skipped_reason: str | None = None


class GeneratorResult(DomainModel):
    """Metrics (or a skip reason) for one generator variant.

    Attributes:
        variant: The generator that was run.
        skipped_reason: Why the variant did not run, when applicable.
        metrics: Aggregated metrics from ``summarize``.
        items: Per-question results.
    """

    variant: GeneratorVariant
    skipped_reason: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    items: list[ItemResult] = Field(default_factory=list)


class TrainingComparison(DomainModel):
    """Full base-vs-adapter outcome for one book and golden set.

    Attributes:
        book_id: Book the comparison ran against.
        index: Shared index identity (embedder, contextual flag, ...).
        retrieval: Shared retrieval settings used by both variants.
        item_count: Number of golden items evaluated per variant.
        baseline: Name of the base variant when it produced metrics.
        variants: Per-variant results, in run order.
        deltas: Per-variant metric deltas versus the base variant.
    """

    book_id: str
    index: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    item_count: int = 0
    baseline: str | None = None
    variants: list[GeneratorResult] = Field(default_factory=list)
    deltas: dict[str, dict[str, float | None]] = Field(default_factory=dict)


def run_comparison(
    book_id: str,
    golden_items: list[GoldenItem],
    runs: list[TrainingRun],
    *,
    index_identity: dict[str, Any] | None = None,
    retrieval_identity: dict[str, Any] | None = None,
    metrics: tuple[str, ...] = COMPARISON_METRICS,
) -> TrainingComparison:
    """Evaluate every generator variant and assemble the comparison report.

    Mirrors the ablation harness: a variant that raises mid-run is recorded as
    skipped and the other variants still report.

    Args:
        book_id: Book the comparison runs against.
        golden_items: Golden questions (the same subset for every variant).
        runs: Generators paired with their answer callables (or skip reasons).
        index_identity: Shared index identity recorded in the report.
        retrieval_identity: Shared retrieval settings recorded in the report.
        metrics: Metric names diffed against the base variant.

    Returns:
        Report with per-variant metrics, per-item results and deltas.
    """
    results: list[GeneratorResult] = []
    for run in runs:
        if run.answer is None:
            results.append(
                GeneratorResult(
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
                GeneratorResult(
                    variant=run.variant,
                    skipped_reason=f"failed: {type(exc).__name__}: {exc}",
                )
            )
            continue
        results.append(
            GeneratorResult(variant=run.variant, metrics=summarize(items), items=items)
        )

    base = next(
        (
            result
            for result in results
            if result.variant.kind == "base" and result.skipped_reason is None
        ),
        None,
    )
    deltas = (
        {}
        if base is None
        else metric_deltas_against(
            base.metrics,
            (
                (result.variant.name, result.metrics)
                for result in results
                if result.skipped_reason is None
            ),
            metrics,
        )
    )
    return TrainingComparison(
        book_id=book_id,
        index=dict(index_identity or {}),
        retrieval=dict(retrieval_identity or {}),
        item_count=len(golden_items),
        baseline=base.variant.name if base else None,
        variants=results,
        deltas=deltas,
    )
