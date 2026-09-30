"""A single search hit: chunk id + score."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Hot-path result object (created per candidate per query), hence a dataclass."""

    id: str
    score: float
