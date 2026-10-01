"""Unit tests for the optional RAGAS runner (no LLM calls)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from distiller.evaluation.ragas_runner import ragas_available, run_ragas
from distiller.exceptions import MissingDependencyError

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


class FakeSeries:
    """Minimal stand-in for a pandas Series."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def mean(self) -> float:
        return sum(self._values) / len(self._values)


class FakeFrame:
    """Minimal stand-in for the pandas frame RAGAS returns."""

    def __init__(self, metrics: dict[str, list[float]]) -> None:
        self._metrics = metrics

    def select_dtypes(self, include: str) -> FakeFrame:
        return self

    @property
    def columns(self) -> list[str]:
        return list(self._metrics)

    def __len__(self) -> int:
        return len(next(iter(self._metrics.values())))

    def __getitem__(self, column: str) -> FakeSeries:
        return FakeSeries(self._metrics[column])


def fake_ragas(
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm_factory: Any = None,
    evaluate: Any = None,
    failing_module: str | None = None,
) -> None:
    """Route ragas imports to fakes and mark the package as available."""
    frame = FakeFrame({"faithfulness": [0.5, 1.0], "precision": [0.25, 0.75]})
    modules: dict[str, Any] = {
        "ragas": SimpleNamespace(
            EvaluationDataset=lambda samples: samples,
            evaluate=evaluate
            or (lambda **kwargs: SimpleNamespace(to_pandas=lambda: frame)),
        ),
        "ragas.dataset_schema": SimpleNamespace(
            SingleTurnSample=lambda **kwargs: kwargs
        ),
        "ragas.metrics": SimpleNamespace(
            Faithfulness=lambda: "faithfulness",
            LLMContextPrecisionWithoutReference=lambda: "precision",
        ),
        "ragas.llms": SimpleNamespace(
            llm_factory=llm_factory or (lambda model, client=None: f"judge:{model}")
        ),
    }

    def fake_require(name: str, **kwargs: object) -> object:
        if name == failing_module:
            raise MissingDependencyError(name, "eval", "RAGAS evaluation")
        return modules[name]

    monkeypatch.setattr(
        "distiller.evaluation.ragas_runner.is_available", lambda name: True
    )
    monkeypatch.setattr("distiller.evaluation.ragas_runner.require", fake_require)


def test_run_ragas_returns_averaged_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful run averages the numeric columns and reports provenance."""
    fake_ragas(monkeypatch)

    result = run_ragas([SAMPLE], model="judge-model")

    assert result["model"] == "judge-model"
    assert result["sample_count"] == 2
    assert result["metrics"] == {"faithfulness": 0.75, "precision": 0.5}


def test_run_ragas_reports_judge_setup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing judge factory returns an error payload instead of raising."""

    def broken_llm_factory(model: str, client: object = None) -> object:
        raise RuntimeError("no judge")

    fake_ragas(monkeypatch, llm_factory=broken_llm_factory)

    result = run_ragas([SAMPLE], model="judge-model")

    assert "judge setup failed: no judge" in result["error"]


def test_run_ragas_reports_evaluation_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing evaluate call returns an error payload instead of raising."""

    def broken_evaluate(**kwargs: object) -> object:
        raise RuntimeError("boom")

    fake_ragas(monkeypatch, evaluate=broken_evaluate)

    result = run_ragas([SAMPLE], model="judge-model")

    assert "ragas evaluation failed: RuntimeError: boom" in result["error"]


def test_run_ragas_reports_missing_modules_midway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing ragas submodule is reported as an error payload."""
    fake_ragas(monkeypatch, failing_module="ragas.metrics")

    result = run_ragas([SAMPLE], model="judge-model")

    assert "ragas.metrics" in result["error"]


def test_run_ragas_rejects_unusable_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Samples without answers or contexts are rejected after the extra check."""
    fake_ragas(monkeypatch)

    result = run_ragas(
        [{"question": "q", "answer": "", "contexts": []}], model="judge-model"
    )

    assert "no usable samples" in result["error"]
