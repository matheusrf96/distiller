"""Unit tests for settings bounds validation."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from distiller.config import (
    ChunkingSettings,
    GgufSettings,
    SynthesisSettings,
    ThematicSettings,
)


@pytest.mark.parametrize(
    ("settings_type", "overrides", "message"),
    [
        (ChunkingSettings, {"target_chars": 0}, "target_chars"),
        (ChunkingSettings, {"max_chars": 100, "target_chars": 200}, "max_chars"),
        (ChunkingSettings, {"overlap_chars": 60, "target_chars": 100}, "overlap_chars"),
        (SynthesisSettings, {"max_chunks": 0}, "max_chunks"),
        (SynthesisSettings, {"questions_per_chunk": 0}, "questions_per_chunk"),
        (SynthesisSettings, {"distractors": -1}, "distractors"),
        (SynthesisSettings, {"negative_ratio": 1.5}, "negative_ratio"),
        (ThematicSettings, {"window_size": 1}, "window_size"),
        (ThematicSettings, {"map_top_k": 0}, "map_top_k"),
        (ThematicSettings, {"max_source_chars": 0}, "max_source_chars"),
        (ThematicSettings, {"max_summary_chars": 0}, "max_summary_chars"),
        (GgufSettings, {"timeout": 0}, "timeout"),
        (GgufSettings, {"model": "   "}, "model"),
    ],
)
def test_settings_reject_out_of_bounds_values(
    settings_type: type[BaseModel],
    overrides: dict[str, object],
    message: str,
) -> None:
    """Each validator rejects its out-of-bounds value with a named field."""
    with pytest.raises(ValidationError, match=message):
        settings_type(**overrides)
