"""Deterministic quality filters for generated QA pairs (Phase 2, step 2)."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from pydantic import Field

from ..models import DomainModel
from .qa import QAPair

if TYPE_CHECKING:
    from ..models import Chunk

MIN_QUESTION_CHARS = 12
MIN_ANSWER_CHARS = 20
MIN_QUOTE_CHARS = 15
ECHO_RATIO = 0.7


class RejectedPair(DomainModel):
    """A pair that failed filtering.

    Attributes:
        pair_id: Id of the rejected pair.
        question: The rejected question (for inspection).
        reason: Machine-readable rejection reason.
    """

    pair_id: str
    question: str
    reason: str


class FilterOutcome(DomainModel):
    """Kept and rejected pairs from one filtering pass.

    Attributes:
        kept: Pairs that passed every rule.
        rejected: Rejected pairs with reasons.
    """

    kept: list[QAPair] = Field(default_factory=list)
    rejected: list[RejectedPair] = Field(default_factory=list)

    def rejection_counts(self) -> dict[str, int]:
        """Count rejections per reason, for the manifest."""
        counts: dict[str, int] = {}
        for rejected in self.rejected:
            counts[rejected.reason] = counts.get(rejected.reason, 0) + 1
        return counts


def filter_pairs(
    pairs: list[QAPair], *, chunks_by_id: dict[str, Chunk]
) -> FilterOutcome:
    """Apply deterministic quality rules to generated pairs.

    Rules, in order:

    1. quotes must appear verbatim in the source chunk (normalized whitespace,
       case-insensitive) and meet ``MIN_QUOTE_CHARS``; failing quotes are dropped;
    2. a pair with no verified quotes left is rejected;
    3. question and answer must meet minimum lengths;
    4. questions must be unique (normalized);
    5. questions must not nearly echo their own answer (``ECHO_RATIO``).

    Args:
        pairs: Generated pairs.
        chunks_by_id: Source chunks by id, for quote verification.

    Returns:
        Outcome with kept pairs (quotes pruned to the verified ones) and
        rejection reasons.
    """
    kept: list[QAPair] = []
    rejected: list[RejectedPair] = []
    seen_questions: set[str] = set()

    for pair in pairs:
        verified = _verified_quotes(pair, chunks_by_id.get(pair.chunk_id))
        if not verified:
            rejected.append(_reject(pair, "no verified quotes"))
            continue
        if (
            len(pair.question.strip()) < MIN_QUESTION_CHARS
            or len(pair.answer.strip()) < MIN_ANSWER_CHARS
        ):
            rejected.append(_reject(pair, "too short"))
            continue

        normalized_question = _normalize(pair.question)
        if normalized_question in seen_questions:
            rejected.append(_reject(pair, "duplicate question"))
            continue
        if (
            SequenceMatcher(None, normalized_question, _normalize(pair.answer)).ratio()
            >= ECHO_RATIO
        ):
            rejected.append(_reject(pair, "question echoes answer"))
            continue

        seen_questions.add(normalized_question)
        kept.append(
            pair
            if verified == pair.quotes
            else pair.model_copy(update={"quotes": verified})
        )

    return FilterOutcome(kept=kept, rejected=rejected)


def _verified_quotes(pair: QAPair, chunk: Chunk | None) -> list[str]:
    if chunk is None:
        return []
    haystack = _normalize(chunk.text)
    return [
        quote
        for quote in pair.quotes
        if len(quote) >= MIN_QUOTE_CHARS and _normalize(quote) in haystack
    ]


def _reject(pair: QAPair, reason: str) -> RejectedPair:
    return RejectedPair(pair_id=pair.id, question=pair.question, reason=reason)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
