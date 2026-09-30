"""Unit tests for the deterministic evaluation metrics."""

from __future__ import annotations

from distiller.evaluation import evaluate_item, summarize
from distiller.evaluation.golden import GoldenItem
from distiller.models import Answer, Chunk, Citation, RetrievedChunk


def build_answer(
    text: str,
    chapters: list[str],
    *,
    refused: bool = False,
    citation_count: int = 1,
) -> Answer:
    """Build an answer with contexts and citations for metric tests."""
    contexts = [
        RetrievedChunk(
            chunk=Chunk(
                id=f"b:{index:04d}:xyz",
                book_id="b",
                ordinal=index,
                text=f"text of {chapter}",
                chapter=chapter,
                heading_path=[chapter],
            )
        )
        for index, chapter in enumerate(chapters)
    ]
    return Answer(
        question="q",
        text=text,
        contexts=contexts,
        refused=refused,
        citations=[
            Citation(
                index=index + 1,
                chunk_id=context.chunk.id,
                chapter=context.chunk.chapter,
            )
            for index, context in enumerate(contexts[:citation_count])
        ],
    )


def test_summarize_computes_expected_metrics() -> None:
    """Aggregated metrics react correctly to hits, misses and refusals."""
    golden_items = [
        GoldenItem(
            question="q1",
            expected_chapters=["Chapter One"],
            expected_answer_contains=["three hundred"],
        ),
        GoldenItem(
            question="q2",
            expected_chapters=["Chapter Two"],
            expected_answer_contains=["cracked"],
        ),
        GoldenItem(question="q3", answerable=False),
    ]
    answers = [
        build_answer('It counted "three hundred" steps [1].', ["Chapter One"]),
        # retrieval missed the target chapter and the snippet is absent
        build_answer("The tower was painted white [1].", ["Chapter Three"]),
        build_answer(
            "I couldn't find that in Book.",
            ["Chapter Two"],
            refused=True,
            citation_count=0,
        ),
    ]

    results = [
        evaluate_item(golden_item, answer)
        for golden_item, answer in zip(golden_items, answers, strict=True)
    ]
    summary = summarize(results)

    assert summary["item_count"] == 3
    assert summary["retrieval_hit_rate"] == 0.5  # q1 hit, q2 missed
    assert summary["answer_rate"] == 1.0  # both answerable items were answered
    assert summary["refusal_accuracy"] == 1.0  # q1/q2 answered, q3 refused
    assert summary["unanswerable_refusal_rate"] == 1.0
    assert summary["contains_rate"] == 0.5  # q1 hit, q2 missed
    assert summary["citation_coverage"] == 1.0
    assert summary["average_citations"] == 1.0


def test_summarize_handles_empty_results() -> None:
    """An empty run reports zero items instead of dividing by zero."""
    assert summarize([]) == {"item_count": 0}


def test_golden_item_roundtrip(tmp_path) -> None:
    """Golden sets save and load as YAML without loss."""
    from distiller.evaluation import load_golden, save_golden

    path = tmp_path / "golden.yaml"
    items = [
        GoldenItem(question="q1", expected_chapters=["One"]),
        GoldenItem(question="q2", answerable=False),
    ]
    save_golden(path, items)

    assert load_golden(path) == items
