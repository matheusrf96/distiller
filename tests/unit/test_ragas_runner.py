"""Unit tests for the optional RAGAS runner (no LLM calls)."""

from __future__ import annotations

import pytest

from distiller.evaluation.ragas_runner import ragas_available, run_ragas

SAMPLE = {"question": "q", "answer": "a", "contexts": ["c"]}


def test_ragas_available_returns_bool() -> None:
    """Availability probing never raises."""
    assert isinstance(ragas_available(), bool)


def test_run_ragas_empty_samples_returns_error_payload() -> None:
    """An empty sample list never raises and reports an actionable error."""
    result = run_ragas([], model="judge")

    assert "error" in result


def test_run_ragas_with_samples_without_context_or_answer() -> None:
    """Samples missing answers or contexts are filtered out."""
    result = run_ragas(
        [{"question": "q", "answer": "", "contexts": ["c"]}],
        model="judge",
    )

    assert "error" in result


def test_run_ragas_reports_missing_extra_when_not_installed() -> None:
    """Without the eval extra the error names the extra to install."""
    if ragas_available():
        pytest.skip("ragas is installed in this environment")

    result = run_ragas([SAMPLE], model="judge")

    assert "ragas is not installed" in result["error"]
