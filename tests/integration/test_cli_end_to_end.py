"""Integration test: the full offline CLI pipeline, ingest -> index -> ask -> eval."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from distiller.cli.main import app
from distiller.evaluation.golden import GoldenItem, save_golden

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
