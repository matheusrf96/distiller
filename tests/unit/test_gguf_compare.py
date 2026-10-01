"""Unit tests for the GGUF generator variant in the training comparison."""

from __future__ import annotations

from distiller.evaluation.golden import GoldenItem
from distiller.models import Answer, Chunk, Citation, RetrievedChunk
from distiller.training.compare import GeneratorVariant, TrainingRun, run_comparison


def _answer(text: str, *, citations: bool = True) -> Answer:
    """Answer quoting one context from Chapter One."""
    context = RetrievedChunk(
        chunk=Chunk(
            id="b:0000:x",
            book_id="b",
            ordinal=0,
            text="text of Chapter One",
            chapter="Chapter One",
            heading_path=["Chapter One"],
        )
    )
    return Answer(
        question="q",
        text=text,
        contexts=[context],
        citations=(
            [Citation(index=1, chunk_id=context.chunk.id, chapter="Chapter One")]
            if citations
            else []
        ),
    )


def test_comparison_records_a_gguf_variant_with_deltas() -> None:
    """The gguf variant shares the golden set and gets deltas vs base (AC10)."""
    items = [
        GoldenItem(
            question="q1",
            expected_chapters=["Chapter One"],
            expected_answer_contains=["three hundred"],
        ),
        GoldenItem(question="q2", answerable=False),
    ]

    def base_answer(question: str) -> Answer:
        return _answer("I cannot answer that.", citations=False)

    def gguf_answer(question: str) -> Answer:
        return _answer("three hundred steps [1]")

    runs = [
        TrainingRun(
            variant=GeneratorVariant(name="base", kind="base", model="fake"),
            answer=base_answer,
        ),
        TrainingRun(
            variant=GeneratorVariant(
                name="gguf",
                kind="gguf",
                model="openai-compat:distiller-book",
                gguf={"sha256": "abc", "quantization": "Q4_K_M"},
            ),
            answer=gguf_answer,
        ),
    ]
    report = run_comparison(
        "book",
        items,
        runs,
        index_identity={"embedder": "hash:64"},
        retrieval_identity={"top_k_final": 8},
    )

    assert report.baseline == "base"
    assert report.item_count == 2
    assert [result.variant.name for result in report.variants] == ["base", "gguf"]
    assert report.variants[1].variant.kind == "gguf"
    assert report.variants[1].variant.gguf["quantization"] == "Q4_K_M"
    assert report.variants[1].metrics["contains_rate"] == 1.0
    assert report.deltas["gguf"]["contains_rate"] == 1.0
    assert report.deltas["gguf"]["citation_coverage"] == 1.0
