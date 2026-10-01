"""Unit tests for CLI rendering branches (captured console output)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from distiller.cli.render import (
    render_ablation,
    render_answer,
    render_books,
    render_comparison,
    render_info,
    render_ragas,
    render_registration,
    render_runtime,
    render_tree_build,
)
from distiller.evaluation.ablation import (
    AblationReport,
    AblationVariant,
    VariantResult,
)
from distiller.models import Answer
from distiller.thematic import TreeManifest
from distiller.training.compare import (
    GeneratorResult,
    GeneratorVariant,
    TrainingComparison,
)
from distiller.training.registry import TrainingReport


def test_render_answer_without_citations_prints_a_hint(capsys) -> None:
    """An unrefused answer without citations warns about grounding."""
    render_answer(Answer(question="q", text="plain answer"), "The Book")

    assert "No citations extracted" in capsys.readouterr().out


def test_render_ragas_prints_the_payload(capsys) -> None:
    """The RAGAS panel shows the raw payload."""
    render_ragas({"model": "judge", "sample_count": 2})

    assert "RAGAS" in capsys.readouterr().out


def test_render_ablation_without_a_baseline_uses_dashes(capsys) -> None:
    """Deltas are dashes when no variant produced a baseline."""
    report = AblationReport(
        book_id="the-book",
        item_count=1,
        baseline=None,
        variants=[
            VariantResult(
                variant=AblationVariant(name="hybrid", top_k_final=4),
                metrics={
                    "retrieval_hit_rate": 1.0,
                    "contains_rate": 1.0,
                    "citation_coverage": 1.0,
                    "refusal_accuracy": 1.0,
                },
            )
        ],
    )

    render_ablation(report)

    output = capsys.readouterr().out
    assert "hybrid" in output
    assert "—" in output


def test_render_comparison_marks_skipped_variants(capsys) -> None:
    """Skipped variants render dashes and a warning instead of metrics."""
    report = TrainingComparison(
        book_id="the-book",
        item_count=1,
        baseline="base",
        variants=[
            GeneratorResult(
                variant=GeneratorVariant(name="base", kind="base", model="fake"),
                metrics={
                    "contains_rate": 1.0,
                    "refusal_accuracy": 1.0,
                    "citation_coverage": 1.0,
                },
            ),
            GeneratorResult(
                variant=GeneratorVariant(
                    name="adapter", kind="adapter", model="adapter-model"
                ),
                skipped_reason="failed: RuntimeError: endpoint down",
            ),
        ],
    )

    render_comparison(report)

    output = capsys.readouterr().out
    assert "endpoint down" in output
    assert "skipped" in output


def test_render_registration_warns_on_a_dataset_mismatch(capsys) -> None:
    """A dataset mismatch is surfaced after the registration table."""
    report = TrainingReport(
        book_id="the-book",
        base_model="Qwen/Qwen3-4B",
        dataset_hash="abc",
        config_hash="def",
        created_at=datetime.now(UTC),
        dataset_hash_matches=False,
    )

    render_registration(report, Path("adapter"))

    assert "different dataset" in capsys.readouterr().out


def test_render_tree_build_warns_about_failed_nodes(capsys) -> None:
    """Failed nodes are reported with the retry hint."""
    manifest = TreeManifest(
        book_id="the-book",
        model="fake",
        source_hash="src",
        tree_hash="tree",
        failed_node_ids=["the-book:chapter:2"],
    )

    render_tree_build(manifest, Path("thematic"))

    assert "re-run" in capsys.readouterr().out


def test_render_books_and_info_cover_empty_states(capsys) -> None:
    """An empty library and an unindexed book both print guidance."""
    render_books([])
    render_info("the-book", {"title": "The Book", "authors": []}, None)

    output = capsys.readouterr().out
    assert "No books ingested yet" in output
    assert "Not indexed yet" in output


def test_render_runtime_prints_versions(capsys) -> None:
    """The runtime table lists every module version."""
    render_runtime({"torch": "2.2.0"})

    assert "torch" in capsys.readouterr().out


def test_render_answer_refusal_prints_no_sources_hint(capsys) -> None:
    """A refusal prints the panel only, with no sources hint."""
    render_answer(
        Answer(question="q", text="I couldn't find that.", refused=True), "The Book"
    )

    assert "No citations extracted" not in capsys.readouterr().out
