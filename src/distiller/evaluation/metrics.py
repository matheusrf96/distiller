"""Deterministic, LLM-free evaluation metrics (run in CI on every change).

These are the cheap, stable signals; RAGAS (optional) adds LLM-judged metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, computed_field

from ..models import DomainModel

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..models import Answer
    from .golden import GoldenItem


class ItemResult(DomainModel):
    """One evaluated question.

    Pydantic (not a dataclass) because this object is serialized into
    ``eval/report.json`` and consumed by other tools.

    Attributes:
        question: The question that was asked.
        answerable: Whether the golden set expects an answer.
        refused: Whether the model refused.
        answer_text: The generated answer text.
        retrieved_chapters: Chapters present in the retrieved contexts.
        expected_chapters: Chapters the golden set expects in retrieval.
        expected_contains: Snippets the answer should contain.
        contains_hits: Number of expected snippets found in the answer.
        contains_total: Number of expected snippets checked.
        citation_count: Number of citations in the answer.
    """

    question: str
    answerable: bool
    refused: bool
    answer_text: str
    retrieved_chapters: list[str] = Field(default_factory=list)
    expected_chapters: list[str] = Field(default_factory=list)
    expected_contains: list[str] = Field(default_factory=list)
    contains_hits: int = 0
    contains_total: int = 0
    citation_count: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def contains_rate(self) -> float | None:
        """Fraction of expected snippets found in the answer (None when unchecked)."""
        if self.contains_total == 0:
            return None
        return round(self.contains_hits / self.contains_total, 4)


def evaluate_item(golden_item: GoldenItem, answer: Answer) -> ItemResult:
    """Compare one golden question against one generated answer.

    Args:
        golden_item: Expected behavior for the question.
        answer: Generated answer with contexts and citations.

    Returns:
        Per-question result record for the report.
    """
    answer_normalized = _normalize(answer.text)
    hits = sum(
        1
        for snippet in golden_item.expected_answer_contains
        if _normalize(snippet) in answer_normalized
    )

    return ItemResult(
        question=golden_item.question,
        answerable=golden_item.answerable,
        refused=answer.refused,
        answer_text=answer.text,
        retrieved_chapters=_unique(
            [context.chunk.chapter for context in answer.contexts]
        ),
        expected_chapters=golden_item.expected_chapters,
        expected_contains=golden_item.expected_answer_contains,
        contains_hits=hits,
        contains_total=len(golden_item.expected_answer_contains),
        citation_count=len(answer.citations),
    )


def retrieval_hit(result: ItemResult) -> bool | None:
    """Report whether retrieval surfaced any expected chapter.

    Args:
        result: Per-question result.

    Returns:
        True/False for answerable questions with expectations, else None.
    """
    if not result.answerable or not result.expected_chapters:
        return None
    retrieved = [_normalize(chapter) for chapter in result.retrieved_chapters]
    return any(
        any(_normalize(expected) in chapter for chapter in retrieved)
        for expected in result.expected_chapters
    )


def summarize(results: list[ItemResult]) -> dict[str, Any]:
    """Aggregate per-question results into report metrics.

    Args:
        results: Per-question results from one evaluation run.

    Returns:
        Metric dictionary written to ``eval/report.json``.
    """
    if not results:
        return {"item_count": 0}

    answerable = [result for result in results if result.answerable]
    unanswerable = [result for result in results if not result.answerable]
    with_contains = [result for result in results if result.contains_rate is not None]
    answered = [result for result in results if not result.refused]
    hits = [
        hit
        for hit in (retrieval_hit(result) for result in answerable)
        if hit is not None
    ]

    return {
        "item_count": len(results),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "retrieval_hit_rate": _mean([1.0 if hit else 0.0 for hit in hits]),
        "answer_rate": _mean([0.0 if result.refused else 1.0 for result in answerable]),
        "refusal_accuracy": _mean(
            [
                1.0 if (result.answerable != result.refused) else 0.0
                for result in results
            ]
        ),
        "unanswerable_refusal_rate": _mean(
            [1.0 if result.refused else 0.0 for result in unanswerable]
        ),
        "contains_rate": _mean(
            [
                result.contains_rate
                for result in with_contains
                if result.contains_rate is not None
            ]
        ),
        "citation_coverage": _mean(
            [1.0 if result.citation_count > 0 else 0.0 for result in answered]
        ),
        "average_citations": _mean(
            [float(result.citation_count) for result in answered]
        ),
    }


def metric_deltas_against(
    baseline: dict[str, Any],
    rows: Iterable[tuple[str, dict[str, Any]]],
    metrics: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    """Per-row deltas of selected metrics versus a baseline metric dict.

    Shared by the retrieval ablation and the base-vs-adapter comparison so both
    reports use identical delta semantics.

    Args:
        baseline: Metric dictionary of the reference variant.
        rows: ``(name, metrics)`` pairs to diff, in report order.
        metrics: Metric names to diff (keys of :func:`summarize` output).

    Returns:
        Mapping name -> metric -> delta versus the baseline. A delta is None
        when either side is missing.
    """
    deltas: dict[str, dict[str, float | None]] = {}
    for name, values in rows:
        row: dict[str, float | None] = {}
        for metric in metrics:
            base_value = baseline.get(metric)
            value = values.get(metric)
            row[metric] = (
                None
                if base_value is None or value is None
                else round(float(value) - float(base_value), 4)
            )
        deltas[name] = row
    return deltas


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
