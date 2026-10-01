"""Unit tests for the base-vs-adapter comparison."""

from __future__ import annotations

from distiller.evaluation.golden import GoldenItem
from distiller.models import Answer, Chunk, Citation, RetrievedChunk
from distiller.training.compare import (
    GeneratorResult,
    GeneratorVariant,
    TrainingComparison,
    TrainingRun,
    run_comparison,
)


def build_answer(text: str, chapter: str, *, citations: bool = True) -> Answer:
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
        citations=(
            [Citation(index=1, chunk_id=context.chunk.id, chapter=chapter)]
            if citations
            else []
        ),
    )


def build_items() -> list[GoldenItem]:
    """Two golden questions: one answerable, one not."""
    return [
        GoldenItem(
            question="q1",
            expected_chapters=["Chapter One"],
            expected_answer_contains=["three hundred"],
        ),
        GoldenItem(question="q2", answerable=False),
    ]


def test_run_comparison_computes_deltas_against_base() -> None:
    """The adapter's metrics are diffed against the base variant (REQ-TR-011)."""

    def base_answer(question: str) -> Answer:
        return build_answer("I cannot answer that.", "Chapter One", citations=False)

    def adapter_answer(question: str) -> Answer:
        return build_answer("three hundred steps [1]", "Chapter One")

    runs = [
        TrainingRun(
            variant=GeneratorVariant(name="base", kind="base", model="fake"),
            answer=base_answer,
        ),
        TrainingRun(
            variant=GeneratorVariant(
                name="adapter",
                kind="adapter",
                model="fake",
                adapter={"dataset_hash": "abc"},
            ),
            answer=adapter_answer,
        ),
    ]
    report = run_comparison(
        "book",
        build_items(),
        runs,
        index_identity={"embedder": "hash:64"},
        retrieval_identity={"top_k_final": 8},
    )

    assert report.book_id == "book"
    assert report.index == {"embedder": "hash:64"}
    assert report.retrieval == {"top_k_final": 8}
    assert report.item_count == 2
    assert report.baseline == "base"
    assert report.variants[0].metrics["item_count"] == 2
    assert report.variants[1].variant.adapter == {"dataset_hash": "abc"}
    assert report.variants[0].metrics["contains_rate"] == 0.0
    assert report.variants[1].metrics["contains_rate"] == 1.0
    assert report.deltas["adapter"]["contains_rate"] == 1.0
    assert report.deltas["adapter"]["citation_coverage"] == 1.0
    assert report.deltas["base"]["contains_rate"] == 0.0


def test_run_comparison_records_a_failing_variant_as_skipped() -> None:
    """One broken variant is skipped; the other still reports (REQ-TR-011)."""

    def broken(question: str) -> Answer:
        raise RuntimeError("endpoint down")

    def good(question: str) -> Answer:
        return build_answer("three hundred steps [1]", "Chapter One")

    runs = [
        TrainingRun(
            variant=GeneratorVariant(name="base", kind="base", model="fake"),
            answer=good,
        ),
        TrainingRun(
            variant=GeneratorVariant(name="adapter", kind="adapter", model="fake"),
            answer=broken,
        ),
        TrainingRun(
            variant=GeneratorVariant(name="adapter-2", kind="adapter", model="fake"),
            skipped_reason="not configured",
        ),
    ]
    report = run_comparison("book", build_items(), runs)

    assert report.variants[0].metrics["item_count"] == 2
    assert report.variants[1].skipped_reason is not None
    assert "endpoint down" in report.variants[1].skipped_reason
    assert report.variants[2].skipped_reason == "not configured"
    assert report.deltas["base"]["contains_rate"] == 0.0
    assert "adapter" not in report.deltas


def test_run_comparison_without_a_base_yields_no_deltas() -> None:
    """Without a base variant there is nothing to diff against (REQ-TR-011)."""

    def good(question: str) -> Answer:
        return build_answer("three hundred steps [1]", "Chapter One")

    runs = [
        TrainingRun(
            variant=GeneratorVariant(name="adapter", kind="adapter", model="fake"),
            answer=good,
        )
    ]
    report = run_comparison("book", build_items(), runs)

    assert report.baseline is None
    assert report.deltas == {}


def test_comparison_round_trips_through_json() -> None:
    """The report serializes to JSON and validates back (REQ-TR-012)."""
    report = TrainingComparison(
        book_id="book",
        index={"embedder": "hash:64"},
        retrieval={"top_k_final": 8},
        item_count=1,
        baseline="base",
        variants=[
            GeneratorResult(
                variant=GeneratorVariant(name="base", kind="base", model="fake"),
                metrics={"item_count": 1},
            )
        ],
        deltas={"base": {"contains_rate": 0.0}},
    )

    payload = report.model_dump_json()
    assert TrainingComparison.model_validate_json(payload) == report
