"""Unit tests for the emitted T4 notebook and the runbook."""

from __future__ import annotations

import json
from pathlib import Path

from distiller.training.notebook import RUNBOOK_PATH, emit_notebook
from distiller.training.qlora import QLoRAConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_emit_notebook_is_valid_json_with_expected_cells() -> None:
    """The notebook is valid nbformat JSON with every runbook cell (REQ-TR-007)."""
    notebook = emit_notebook(QLoRAConfig(), book_id="the-lantern-keeper")

    payload = json.loads(json.dumps(notebook))
    assert payload["nbformat"] == 4
    assert payload["cells"]

    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    markdown_cells = [
        cell for cell in payload["cells"] if cell["cell_type"] == "markdown"
    ]
    assert markdown_cells
    assert all(
        cell["outputs"] == [] and cell["execution_count"] is None for cell in code_cells
    )

    sources = "\n".join("".join(cell["source"]) for cell in payload["cells"])
    for needle in (
        "pip install",
        "qlora.json",
        "train.jsonl",
        "validation.jsonl",
        "from_pretrained",
        "SFTTrainer",
        "save_pretrained",
        "run.json",
        "manifest.json",
        RUNBOOK_PATH,
    ):
        assert needle in sources, f"missing {needle!r} in the emitted notebook"


def test_qlora_runbook_documents_the_manual_steps() -> None:
    """The runbook exists, documents register/compare, and is linked (REQ-TR-016)."""
    runbook = REPO_ROOT / "docs" / "qlora-runbook.md"
    assert runbook.exists()

    text = runbook.read_text(encoding="utf-8")
    assert "distiller train" in text
    assert "--register" in text
    assert "train-eval" in text

    docs_index = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "qlora-runbook.md" in docs_index
