"""Unit tests for the retrieval ablation harness."""

from __future__ import annotations

import pytest
import typer

from distiller.evaluation.ablation import (
    AblationReport,
    AblationRun,
    AblationVariant,
    VariantResult,
    build_variants,
    metric_deltas,
    run_ablation,
)
from distiller.evaluation.golden import GoldenItem
from distiller.models import Answer, Chunk, Citation, RetrievedChunk


def build_answer(text: str, chapter: str, *, refused: bool = False) -> Answer:
    """Answer quoting one context from the given chapter."""
    context = RetrievedChunk(
        chunk=Chunk(
            id="b:0000:x",
            book_id="b",
            ordinal=0,
            text=f"text of {chapter}",
            chapter=chapter,
            heading_path=[chapter],
        )
    )
    return Answer(
        question="q",
        text=text,
        contexts=[context],
        refused=refused,
        citations=[Citation(index=1, chunk_id=context.chunk.id, chapter=chapter)],
    )


def build_items() -> list[GoldenItem]:
    """Two golden questions: one answerable, one not."""
    return [
        GoldenItem(question="q1", expected_chapters=["Chapter One"]),
        GoldenItem(question="q2", answerable=False),
    ]


def test_build_variants_defaults() -> None:
    """The default matrix is hybrid then hybrid+rerank (REQ-RA-002)."""
    variants = build_variants(top_k_final=8, rerank_pool=40)

    assert [variant.name for variant in variants] == ["hybrid", "hybrid+rerank"]
    assert [variant.rerank for variant in variants] == [False, True]
    assert all(variant.top_k_final == 8 for variant in variants)
    assert all(variant.rerank_pool == 40 for variant in variants)


def test_build_variants_sweep_and_no_rerank() -> None:
    """A sweep crosses top-k values with both rerank states (REQ-RA-003/REQ-RA-009)."""
    swept = build_variants(top_k_final=8, rerank_pool=40, top_k_values=[4, 12])

    assert [variant.name for variant in swept] == [
        "hybrid-k4",
        "hybrid+rerank-k4",
        "hybrid-k12",
        "hybrid+rerank-k12",
    ]
    assert [variant.top_k_final for variant in swept] == [4, 4, 12, 12]

    baseline_only = build_variants(top_k_final=8, rerank_pool=40, include_rerank=False)
    assert [variant.name for variant in baseline_only] == ["hybrid"]


def test_build_variants_keeps_plain_names_for_the_default_value() -> None:
    """A sweep that equals the configured value keeps the unlabelled names."""
    variants = build_variants(top_k_final=8, rerank_pool=40, top_k_values=[8])

    assert [variant.name for variant in variants] == ["hybrid", "hybrid+rerank"]


def test_run_ablation_records_metrics_skips_and_failures() -> None:
    """Metrics, skip reasons and failures are recorded per variant.

    Covers REQ-RA-004, REQ-RA-011 and REQ-RA-012.
    """

    def good(question: str) -> Answer:
        return build_answer("three hundred steps [1]", "Chapter One")

    def broken(question: str) -> Answer:
        raise RuntimeError("llm down")

    runs = [
        AblationRun(variant=AblationVariant(name="hybrid"), answer=good),
        AblationRun(
            variant=AblationVariant(name="hybrid+rerank", rerank=True),
            skipped_reason="no extra",
        ),
        AblationRun(
            variant=AblationVariant(name="hybrid-k12", top_k_final=12),
            answer=broken,
        ),
    ]
    report = run_ablation(
        "book", build_items(), runs, index_identity={"embedder": "hash:64"}
    )

    assert report.book_id == "book"
    assert report.index == {"embedder": "hash:64"}
    assert report.item_count == 2
    assert report.baseline == "hybrid"

    baseline, skipped, failed = report.variants
    assert baseline.metrics["item_count"] == 2
    assert baseline.metrics["retrieval_hit_rate"] == 1.0
    assert len(baseline.items) == 2
    assert skipped.skipped_reason == "no extra"
    assert skipped.metrics == {}
    assert failed.skipped_reason is not None
    assert "llm down" in failed.skipped_reason


def test_metric_deltas_against_baseline() -> None:
    """Deltas are computed versus the first non-skipped variant (REQ-RA-007)."""
    report = AblationReport(
        book_id="book",
        baseline="hybrid",
        variants=[
            VariantResult(
                variant=AblationVariant(name="hybrid"),
                metrics={"retrieval_hit_rate": 0.5, "contains_rate": 0.25},
            ),
            VariantResult(
                variant=AblationVariant(name="hybrid+rerank", rerank=True),
                metrics={"retrieval_hit_rate": 0.75, "contains_rate": 0.25},
            ),
            VariantResult(
                variant=AblationVariant(name="hybrid-k12"),
                skipped_reason="no extra",
            ),
        ],
    )

    deltas = metric_deltas(report, ("retrieval_hit_rate", "contains_rate"))

    assert deltas["hybrid"]["retrieval_hit_rate"] == 0.0
    assert deltas["hybrid+rerank"]["retrieval_hit_rate"] == 0.25
    assert deltas["hybrid+rerank"]["contains_rate"] == 0.0
    assert "hybrid-k12" not in deltas  # skipped variants are excluded


def test_metric_deltas_without_a_baseline_is_empty() -> None:
    """A report with no runnable variant yields no deltas."""
    report = AblationReport(
        book_id="book",
        variants=[
            VariantResult(variant=AblationVariant(name="hybrid"), skipped_reason="x")
        ],
    )

    assert metric_deltas(report, ("retrieval_hit_rate",)) == {}


def test_report_round_trips_through_json() -> None:
    """The report serializes to JSON and validates back (it is written to disk)."""
    report = AblationReport(
        book_id="book",
        index={"embedder": "hash:64", "contextual": True},
        item_count=1,
        baseline="hybrid",
        variants=[
            VariantResult(
                variant=AblationVariant(name="hybrid"),
                metrics={"item_count": 1},
            )
        ],
    )

    payload = report.model_dump_json()
    assert AblationReport.model_validate_json(payload) == report


def test_parse_top_k_values() -> None:
    """The sweep parser accepts lists, spaces, and rejects garbage (REQ-RA-003)."""
    from distiller.cli.context import parse_top_k_values

    assert parse_top_k_values(None) is None
    assert parse_top_k_values("4, 8,12") == [4, 8, 12]

    with pytest.raises(typer.BadParameter):
        parse_top_k_values("4,x")
    with pytest.raises(typer.BadParameter):
        parse_top_k_values("0")
    with pytest.raises(typer.BadParameter):
        parse_top_k_values(",")


def test_build_ablation_runs_does_not_mutate_settings(monkeypatch) -> None:
    """Variant overrides are copies; global settings stay untouched (REQ-RA-005)."""
    from types import SimpleNamespace

    from distiller.cli.context import build_ablation_runs
    from distiller.config import Settings

    settings = Settings()
    settings.retrieval = settings.retrieval.model_copy(update={"top_k_final": 7})
    before = settings.retrieval.model_dump()

    bundle = SimpleNamespace(book=SimpleNamespace(title="Book"))
    monkeypatch.setattr("distiller.cli.context.is_available", lambda module: False)
    variants = [
        AblationVariant(name="hybrid", rerank=False, top_k_final=7),
        AblationVariant(
            name="hybrid+rerank", rerank=True, top_k_final=3, rerank_pool=9
        ),
    ]

    runs = build_ablation_runs(settings, bundle, variants)  # type: ignore[arg-type]

    assert settings.retrieval.model_dump() == before
    assert runs[0].answer is not None
    assert runs[1].skipped_reason is not None
