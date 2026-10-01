"""Integration test: the full offline CLI pipeline, ingest -> index -> ask -> eval."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from distiller.cli.main import app
from distiller.evaluation.golden import GoldenItem, save_golden
from distiller.optional_deps import is_available
from distiller.training.qlora import TRAINING_MODULES

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

runner = CliRunner()
BOOK_ID = "the-lantern-keeper"


def invoke_cli(arguments: list[str]):
    """Run the CLI and fail the test with context when the command errors."""
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, (
        f"{arguments} failed:\n{result.output}\n{result.exception!r}"
    )
    return result


def flat_output(result) -> str:
    """Normalize captured CLI output so wrapped panel/table lines stay searchable."""
    return " ".join(result.output.replace("│", " ").split())


def test_full_offline_pipeline(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """The five CLI stages work end to end without models or network."""
    epub_path = epub_factory(tmp_path / "lantern.epub")

    # 1. ingest
    result = invoke_cli(["ingest", str(epub_path)])
    assert BOOK_ID in result.output

    # 2. index
    result = invoke_cli(["index", BOOK_ID])
    assert "hash:512" in result.output

    # 3. ask (plain + json)
    result = invoke_cli(["ask", BOOK_ID, "How many steps did the keeper count?"])
    assert "Chapter One" in result.output

    result = invoke_cli(
        ["ask", BOOK_ID, "How many steps did the keeper count?", "--json"]
    )
    payload = json.loads(result.output)
    assert payload["citations"], "expected citations in JSON answer"
    assert payload["citations"][0]["chapter"] == "Chapter One"

    # 4. eval with a golden set
    golden_path = tmp_path / "golden.yaml"
    save_golden(
        golden_path,
        [
            GoldenItem(
                question="How many steps did the keeper count?",
                expected_chapters=["Chapter One"],
                expected_answer_contains=["three hundred"],
            ),
            GoldenItem(question="Who won the village sailing race?", answerable=False),
        ],
    )
    result = invoke_cli(["eval", BOOK_ID, "--golden", str(golden_path)])
    assert "retrieval_hit_rate" in result.output

    report_path = offline_env / BOOK_ID / "eval" / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["metrics"]["item_count"] == 2
    assert report["metrics"]["retrieval_hit_rate"] == 1.0

    # 5. listing and info
    assert BOOK_ID in invoke_cli(["books"]).output
    assert "The Lantern Keeper" in invoke_cli(["info", BOOK_ID]).output


def test_cli_reports_domain_errors_without_tracebacks(
    offline_env: Path,
    tmp_path: Path,
) -> None:
    """Expected user errors are friendly messages, never raw tracebacks."""
    unsupported = tmp_path / "bad.mobi"
    unsupported.write_text("content", encoding="utf-8")

    result = runner.invoke(app, ["ingest", str(unsupported)])
    assert result.exit_code != 0
    assert "Unsupported file type" in result.output
    assert "Traceback" not in result.output

    result = runner.invoke(app, ["index", "no-such-book"])
    assert result.exit_code != 0
    assert "No ingested book" in result.output
    assert "Traceback" not in result.output

    malformed_golden = tmp_path / "bad-golden.yaml"
    malformed_golden.write_text(
        "items:\n  - question: What?\n    unknown_field: oops\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", BOOK_ID, "--golden", str(malformed_golden)])
    assert result.exit_code != 0
    assert "Invalid golden set" in result.output
    assert "Traceback" not in result.output


def test_contextual_index_end_to_end(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """`index --contextual` enriches; answers stay quotable; eval records the index.

    Covers REQ-CR-006 (original text in prompts), REQ-CR-008 (CLI flag),
    REQ-CR-011 (eval report index block) and REQ-CR-012 (offline fake LLM).
    """
    epub_path = epub_factory(tmp_path / "lantern.epub")
    invoke_cli(["ingest", str(epub_path)])
    invoke_cli(["index", BOOK_ID, "--contextual"])

    metadata = json.loads(
        (offline_env / BOOK_ID / "index" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["contextual"] is True
    assert metadata["enriched_chunks"] > 0

    result = invoke_cli(
        ["ask", BOOK_ID, "What happened to the lantern during the storm?"]
    )
    # the synthetic context never leaks into prompts or answers
    assert "A passage about" not in result.output
    assert "Chapter Two" in result.output

    golden_path = tmp_path / "golden.yaml"
    save_golden(
        golden_path,
        [
            GoldenItem(
                question="What happened to the lantern during the storm?",
                expected_chapters=["Chapter Two"],
            )
        ],
    )
    invoke_cli(["eval", BOOK_ID, "--golden", str(golden_path)])

    report = json.loads(
        (offline_env / BOOK_ID / "eval" / "report.json").read_text(encoding="utf-8")
    )
    assert report["index"]["contextual"] is True
    assert report["index"]["embedder"] == "hash:512"

    # --no-contextual resets the flags on a fresh build (AC8)
    invoke_cli(["index", BOOK_ID, "--no-contextual"])
    reset = json.loads(
        (offline_env / BOOK_ID / "index" / "metadata.json").read_text(encoding="utf-8")
    )
    assert reset["contextual"] is False
    assert reset["enriched_chunks"] == 0


def _prepare_ablation_inputs(
    epub_factory: Callable[..., Path],
    tmp_path: Path,
    *,
    top_k: str | None = None,
) -> list[str]:
    """Ingest, index and write a small golden set; return the ablation argv tail."""
    epub_path = epub_factory(tmp_path / "lantern.epub")
    invoke_cli(["ingest", str(epub_path)])
    invoke_cli(["index", BOOK_ID])

    golden_path = tmp_path / "golden.yaml"
    save_golden(
        golden_path,
        [
            GoldenItem(
                question="What happened to the lantern during the storm?",
                expected_chapters=["Chapter Two"],
                expected_answer_contains=["cracked"],
            ),
            GoldenItem(question="Who won the village sailing race?", answerable=False),
        ],
    )
    arguments = ["ablation", BOOK_ID, "--golden", str(golden_path)]
    if top_k is not None:
        arguments += ["--top-k", top_k]
    return arguments


def test_ablation_command_skips_missing_reranker(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Without the embed extra, rerank variants are skipped, not failed.

    Covers REQ-RA-001, REQ-RA-004 and REQ-RA-006.
    """
    monkeypatch.setattr("distiller.cli.context.is_available", lambda module: False)
    arguments = _prepare_ablation_inputs(epub_factory, tmp_path)

    result = invoke_cli(arguments)

    assert "skipped" in result.output
    report = json.loads(
        (offline_env / BOOK_ID / "eval" / "ablation.json").read_text(encoding="utf-8")
    )
    assert report["baseline"] == "hybrid"
    assert [variant["variant"]["name"] for variant in report["variants"]] == [
        "hybrid",
        "hybrid+rerank",
    ]
    assert report["variants"][0]["metrics"]["item_count"] == 2
    assert "embed" in report["variants"][1]["skipped_reason"]
    assert report["index"]["embedder"] == "hash:512"


def test_ablation_command_runs_rerank_variants_with_sweep(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
    monkeypatch,
) -> None:
    """With a reranker available, the sweep matrix runs and JSON matches the file.

    Covers REQ-RA-003, REQ-RA-007 and REQ-RA-010.
    """

    class StubReranker:
        @property
        def name(self) -> str:
            return "stub-reranker"

        def rerank(self, query: str, items: list, top_k: int) -> list:
            return list(reversed(items))[:top_k]

    monkeypatch.setattr("distiller.cli.context.is_available", lambda module: True)
    monkeypatch.setattr(
        "distiller.cli.context.get_reranker", lambda settings: StubReranker()
    )
    arguments = _prepare_ablation_inputs(epub_factory, tmp_path, top_k="4,8")

    result = invoke_cli(arguments)
    assert "Δ hit" in result.output

    payload = json.loads(
        (offline_env / BOOK_ID / "eval" / "ablation.json").read_text(encoding="utf-8")
    )
    names = [variant["variant"]["name"] for variant in payload["variants"]]
    assert names == ["hybrid-k4", "hybrid+rerank-k4", "hybrid-k8", "hybrid+rerank-k8"]
    assert all(variant["skipped_reason"] is None for variant in payload["variants"])
    assert all(variant["metrics"]["item_count"] == 2 for variant in payload["variants"])

    json_result = invoke_cli([*arguments, "--json"])
    assert json.loads(json_result.output)["variants"] == payload["variants"]


def test_ablation_command_honours_limit(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
    monkeypatch,
) -> None:
    """--limit evaluates the same first N questions in every variant (REQ-RA-008)."""
    monkeypatch.setattr("distiller.cli.context.is_available", lambda module: False)
    arguments = _prepare_ablation_inputs(epub_factory, tmp_path)
    arguments += ["--limit", "1"]

    invoke_cli(arguments)

    report = json.loads(
        (offline_env / BOOK_ID / "eval" / "ablation.json").read_text(encoding="utf-8")
    )
    assert report["item_count"] == 1
    assert report["variants"][0]["metrics"]["item_count"] == 1


def test_synth_command_end_to_end(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """`synth` generates, filters, formats, reuses the cache and prints JSON.

    Covers REQ-SQ-010, REQ-SQ-013 and REQ-SQ-014.
    """
    epub_path = epub_factory(tmp_path / "lantern.epub")
    invoke_cli(["ingest", str(epub_path)])
    invoke_cli(["index", BOOK_ID])

    result = invoke_cli(
        [
            "synth",
            BOOK_ID,
            "--questions-per-chunk",
            "2",
            "--distractors",
            "1",
            "--seed",
            "5",
        ]
    )
    assert "Synthesized" in result.output

    dataset = offline_env / BOOK_ID / "dataset"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "generated"
    assert manifest["sampled_chunks"] == 3
    assert manifest["generated_pairs"] == 6
    assert manifest["kept_pairs"] == 6
    assert manifest["rejected"] == {}
    assert manifest["example_count"] == 6
    assert (
        manifest["answerable_examples"] + manifest["unanswerable_examples"]
        == manifest["example_count"]
    )

    qa_rows = (dataset / "qa.jsonl").read_text(encoding="utf-8").strip().splitlines()
    raft_rows = (
        (dataset / "raft.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(qa_rows) == 6
    assert len(raft_rows) == manifest["example_count"]

    # a second run reuses the cached pairs instead of calling the LLM
    invoke_cli(["synth", BOOK_ID])
    cached = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert cached["source"] == "cache"
    assert cached["generated_pairs"] == 6

    payload = json.loads(invoke_cli(["synth", BOOK_ID, "--json"]).output)
    assert payload["book_id"] == BOOK_ID
    assert payload["source"] == "cache"


def test_synth_requires_an_index(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """An unindexed book fails with an actionable message (REQ-SQ-015)."""
    epub_path = epub_factory(tmp_path / "lantern.epub")
    invoke_cli(["ingest", str(epub_path)])

    result = runner.invoke(app, ["synth", BOOK_ID])

    assert result.exit_code != 0
    assert "distiller index" in result.output
    assert "Traceback" not in result.output


def _prepare_training_inputs(epub_factory: Callable[..., Path], tmp_path: Path) -> None:
    """Ingest, index and synthesize a small dataset that contains negatives."""
    epub_path = epub_factory(tmp_path / "lantern.epub")
    invoke_cli(["ingest", str(epub_path)])
    invoke_cli(["index", BOOK_ID])
    invoke_cli(
        [
            "synth",
            BOOK_ID,
            "--questions-per-chunk",
            "2",
            "--distractors",
            "1",
            "--negative-ratio",
            "0.5",
            "--seed",
            "5",
        ]
    )


def _write_training_golden(tmp_path: Path) -> Path:
    """Write a two-item golden set for the training comparison."""
    golden_path = tmp_path / "training-golden.yaml"
    save_golden(
        golden_path,
        [
            GoldenItem(
                question="What happened to the lantern during the storm?",
                expected_chapters=["Chapter Two"],
                expected_answer_contains=["cracked"],
            ),
            GoldenItem(question="Who won the village sailing race?", answerable=False),
        ],
    )
    return golden_path


def _register_fake_adapter(
    offline_env: Path,
    tmp_path: Path,
    adapter_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """Register a fake adapter matching the book's manifest; return the manifest."""
    manifest = json.loads(
        (offline_env / BOOK_ID / "training" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source = adapter_factory(
        tmp_path / "adapter-download",
        book_id=BOOK_ID,
        base_model="Qwen/Qwen3-4B",
        dataset_hash=manifest["dataset_hash"],
    )
    monkeypatch.setenv("DISTILLER_ADAPTER__MODEL", "fake")
    invoke_cli(["train", BOOK_ID, "--register", str(source)])
    return manifest


def test_train_command_writes_training_artifacts(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """`train` formats, validates, splits and emits the notebook (REQ-TR-005)."""
    _prepare_training_inputs(epub_factory, tmp_path)

    result = invoke_cli(["train", BOOK_ID, "--seed", "5", "--val-ratio", "0.34"])
    assert "Training data" in result.output

    training = offline_env / BOOK_ID / "training"
    manifest = json.loads((training / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 5
    assert manifest["val_ratio"] == 0.34
    assert (
        manifest["train_count"] + manifest["validation_count"]
        == manifest["source_count"]
    )
    assert manifest["unanswerable_count"] >= 1
    assert (training / "train.jsonl").exists()
    assert (training / "validation.jsonl").exists()
    assert (training / "qlora.json").exists()
    notebook = json.loads((training / "train_t4.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4

    payload = json.loads(
        invoke_cli(
            ["train", BOOK_ID, "--seed", "5", "--val-ratio", "0.34", "--json"]
        ).output
    )
    assert payload == json.loads(
        (training / "manifest.json").read_text(encoding="utf-8")
    )


def test_train_requires_dataset(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A missing RAFT dataset names `distiller synth` (REQ-TR-015)."""
    epub_path = epub_factory(tmp_path / "lantern.epub")
    invoke_cli(["ingest", str(epub_path)])
    invoke_cli(["index", BOOK_ID])

    result = runner.invoke(app, ["train", BOOK_ID])

    assert result.exit_code != 0
    assert "distiller synth" in flat_output(result)
    assert "Traceback" not in result.output


def test_train_reports_validation_errors(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Malformed datasets abort with the error list and no traceback (REQ-TR-004)."""
    _prepare_training_inputs(epub_factory, tmp_path)
    raft_path = offline_env / BOOK_ID / "dataset" / "raft.jsonl"
    rows = raft_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(rows[0])
    payload["contexts"] = []
    rows[0] = json.dumps(payload)
    raft_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["train", BOOK_ID])

    assert result.exit_code != 0
    assert "validation failed" in flat_output(result)
    assert not (offline_env / BOOK_ID / "training").exists()
    assert "Traceback" not in result.output


def test_train_check_runtime_requires_the_extra(offline_env: Path) -> None:
    """`--check-runtime` raises the missing-extra error naming `training`.

    Covers REQ-TR-013.
    """
    if all(is_available(module) for module in TRAINING_MODULES):
        pytest.skip("training stack is installed in this environment")

    result = runner.invoke(app, ["train", BOOK_ID, "--check-runtime"])

    assert result.exit_code != 0
    assert "--extra training" in flat_output(result)
    assert "Traceback" not in result.output


def test_train_register_rejects_a_wrong_adapter(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
    adapter_factory: Callable[..., Path],
) -> None:
    """Registration validates the book id and copies nothing on failure (REQ-TR-008)."""
    _prepare_training_inputs(epub_factory, tmp_path)
    source = adapter_factory(tmp_path / "wrong-adapter", book_id="another-book")

    # before `train` there is no configuration to validate against
    result = runner.invoke(app, ["train", BOOK_ID, "--register", str(source)])
    assert result.exit_code != 0
    assert "distiller train" in flat_output(result)
    assert "Traceback" not in result.output

    invoke_cli(["train", BOOK_ID])
    result = runner.invoke(app, ["train", BOOK_ID, "--register", str(source)])
    assert result.exit_code != 0
    assert "another-book" in flat_output(result)
    assert "Traceback" not in result.output
    assert not (offline_env / BOOK_ID / "training" / "adapter").exists()


def test_ask_and_eval_require_a_registered_adapter(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
    adapter_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter commands fail actionably without registration or endpoint.

    Covers REQ-TR-009 and REQ-TR-015.
    """
    _prepare_training_inputs(epub_factory, tmp_path)
    monkeypatch.setenv("DISTILLER_ADAPTER__MODEL", "fake")
    golden_path = _write_training_golden(tmp_path)

    result = runner.invoke(app, ["ask", BOOK_ID, "What happened?", "--adapter"])
    assert result.exit_code != 0
    assert "--register" in flat_output(result)
    assert "Traceback" not in result.output

    result = runner.invoke(
        app, ["eval", BOOK_ID, "--golden", str(golden_path), "--adapter"]
    )
    assert result.exit_code != 0
    assert "--register" in flat_output(result)
    assert "Traceback" not in result.output

    result = runner.invoke(app, ["train-eval", BOOK_ID, "--golden", str(golden_path)])
    assert result.exit_code != 0
    assert "--register" in flat_output(result)
    assert "Traceback" not in result.output

    # with a registration but no endpoint, the configuration hint is shown
    invoke_cli(["train", BOOK_ID])
    manifest = json.loads(
        (offline_env / BOOK_ID / "training" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source = adapter_factory(
        tmp_path / "adapter-download",
        book_id=BOOK_ID,
        dataset_hash=manifest["dataset_hash"],
    )
    invoke_cli(["train", BOOK_ID, "--register", str(source)])
    monkeypatch.delenv("DISTILLER_ADAPTER__MODEL", raising=False)

    result = runner.invoke(app, ["ask", BOOK_ID, "What happened?", "--adapter"])
    assert result.exit_code != 0
    assert "DISTILLER_ADAPTER__MODEL" in flat_output(result)
    assert "Traceback" not in result.output


def test_ask_and_eval_select_a_registered_adapter(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
    adapter_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--adapter` answers with the adapter endpoint and records provenance.

    Covers REQ-TR-009.
    """
    _prepare_training_inputs(epub_factory, tmp_path)
    invoke_cli(["train", BOOK_ID, "--seed", "5"])
    manifest = _register_fake_adapter(
        offline_env, tmp_path, adapter_factory, monkeypatch
    )

    result = invoke_cli(["ask", BOOK_ID, "What happened to the lantern?", "--adapter"])
    assert "Chapter" in result.output

    golden_path = _write_training_golden(tmp_path)
    invoke_cli(["eval", BOOK_ID, "--golden", str(golden_path), "--adapter"])

    report = json.loads(
        (offline_env / BOOK_ID / "eval" / "report.json").read_text(encoding="utf-8")
    )
    assert report["generator"]["kind"] == "adapter"
    assert report["generator"]["model"] == report["model"]
    assert report["generator"]["adapter"]["dataset_hash"] == manifest["dataset_hash"]
    assert report["generator"]["adapter"]["base_model"] == "Qwen/Qwen3-4B"


def test_eval_records_generator_identity(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A plain eval records `kind: base` next to the legacy model key (REQ-TR-010)."""
    epub_path = epub_factory(tmp_path / "lantern.epub")
    invoke_cli(["ingest", str(epub_path)])
    invoke_cli(["index", BOOK_ID])
    golden_path = _write_training_golden(tmp_path)

    invoke_cli(["eval", BOOK_ID, "--golden", str(golden_path)])

    report = json.loads(
        (offline_env / BOOK_ID / "eval" / "report.json").read_text(encoding="utf-8")
    )
    assert report["generator"] == {"kind": "base", "model": "fake", "adapter": None}
    assert report["model"] == report["generator"]["model"] == "fake"


def test_train_eval_command_end_to_end(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
    adapter_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`train-eval` writes a shared-identity comparison with deltas (REQ-TR-011/012)."""
    _prepare_training_inputs(epub_factory, tmp_path)
    invoke_cli(["train", BOOK_ID, "--seed", "5", "--val-ratio", "0.34"])
    manifest = _register_fake_adapter(
        offline_env, tmp_path, adapter_factory, monkeypatch
    )
    golden_path = _write_training_golden(tmp_path)

    result = invoke_cli(["train-eval", BOOK_ID, "--golden", str(golden_path)])
    assert "Δ contains" in flat_output(result)

    report = json.loads(
        (offline_env / BOOK_ID / "eval" / "training.json").read_text(encoding="utf-8")
    )
    assert report["baseline"] == "base"
    assert [variant["variant"]["name"] for variant in report["variants"]] == [
        "base",
        "adapter",
    ]
    assert all(variant["skipped_reason"] is None for variant in report["variants"])
    assert all(variant["metrics"]["item_count"] == 2 for variant in report["variants"])
    assert report["index"]["embedder"] == "hash:512"
    assert report["retrieval"]["top_k_final"] == 8
    assert (
        report["variants"][1]["variant"]["adapter"]["dataset_hash"]
        == manifest["dataset_hash"]
    )
    assert report["deltas"]["adapter"]["contains_rate"] == 0.0

    json_result = invoke_cli(
        ["train-eval", BOOK_ID, "--golden", str(golden_path), "--json"]
    )
    assert json.loads(json_result.output) == report


def test_training_pipeline_end_to_end(
    offline_env: Path,
    epub_factory: Callable[..., Path],
    tmp_path: Path,
    adapter_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full offline chain completes without GPU, network or heavy deps.

    Covers REQ-TR-014.
    """
    _prepare_training_inputs(epub_factory, tmp_path)
    invoke_cli(["train", BOOK_ID])
    manifest = _register_fake_adapter(
        offline_env, tmp_path, adapter_factory, monkeypatch
    )
    golden_path = _write_training_golden(tmp_path)

    invoke_cli(["train-eval", BOOK_ID, "--golden", str(golden_path)])

    report = json.loads(
        (offline_env / BOOK_ID / "eval" / "training.json").read_text(encoding="utf-8")
    )
    assert report["book_id"] == BOOK_ID
    assert {variant["variant"]["kind"] for variant in report["variants"]} == {
        "base",
        "adapter",
    }
    assert report["item_count"] == 2
    assert report["deltas"]["adapter"]["contains_rate"] == 0.0
    assert manifest["unanswerable_count"] >= 1
