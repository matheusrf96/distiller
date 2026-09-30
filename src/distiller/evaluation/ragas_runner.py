"""Optional RAGAS integration (LLM-judged metrics).

Kept defensive: RAGAS' API evolves quickly, so every failure path returns an
``{"error": ...}`` payload instead of raising, and the deterministic metrics in
``metrics.py`` remain the primary CI signal.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from ..optional_deps import MissingDependencyError, is_available, require

_PURPOSE = "RAGAS evaluation"


def ragas_available() -> bool:
    """Return True when the optional RAGAS package is installed."""
    return is_available("ragas")


def run_ragas(
    samples: list[dict[str, Any]],
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    max_samples: int = 50,
) -> dict[str, Any]:
    """Evaluate grounded answers with RAGAS.

    Args:
        samples: Rows of ``{"question": str, "answer": str, "contexts": list[str]}``.
        model: Judge model name (any OpenAI-compatible endpoint).
        base_url: Judge endpoint base URL.
        api_key: Judge API key, when required.
        max_samples: Maximum number of rows to evaluate.

    Returns:
        Metric dictionary, or an ``{"error": ...}`` payload on any failure.
    """
    if not ragas_available():
        return {"error": "ragas is not installed (uv sync --extra eval)"}

    rows = [s for s in samples if s.get("answer") and s.get("contexts")][
        : max(1, max_samples)
    ]
    if not rows:
        return {"error": "no usable samples (need non-empty answers and contexts)"}

    try:
        ragas = require("ragas", extra="eval", purpose=_PURPOSE)
        schema = require("ragas.dataset_schema", extra="eval", purpose=_PURPOSE)
        metrics = require("ragas.metrics", extra="eval", purpose=_PURPOSE)
        llms = require("ragas.llms", extra="eval", purpose=_PURPOSE)
    except MissingDependencyError as exc:
        return {"error": str(exc)}

    try:
        judge = llms.llm_factory(
            model,
            client=OpenAI(
                base_url=base_url or "http://localhost:11434/v1",
                api_key=api_key or "not-needed",
                timeout=180.0,
            ),
        )
    except Exception as exc:
        return {"error": f"judge setup failed: {exc}"}

    dataset = ragas.EvaluationDataset(
        samples=[
            schema.SingleTurnSample(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=[str(context) for context in row["contexts"]],
            )
            for row in rows
        ]
    )

    try:
        result = ragas.evaluate(
            dataset=dataset,
            metrics=[
                metrics.Faithfulness(),
                metrics.LLMContextPrecisionWithoutReference(),
            ],
            llm=judge,
        )
        frame = result.to_pandas()
        numeric = frame.select_dtypes("number")
        return {
            "model": model,
            "sample_count": len(frame),
            "metrics": {
                str(column): round(float(numeric[column].mean()), 4)
                for column in numeric.columns
            },
        }
    except Exception as exc:
        return {"error": f"ragas evaluation failed: {type(exc).__name__}: {exc}"}
