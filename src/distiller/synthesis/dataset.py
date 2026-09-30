"""Synthesis orchestration: sampling, manifest and the full pipeline."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from ..models import DomainModel
from ..utils import read_jsonl
from .filtering import FilterOutcome, filter_pairs
from .qa import QAPair, generate_pairs
from .raft import RaftExample, build_examples

if TYPE_CHECKING:
    from pathlib import Path

    from ..config import SynthesisSettings
    from ..llm.base import LLMClient
    from ..models import BookDocument, Chunk

logger = logging.getLogger(__name__)


class DatasetManifest(DomainModel):
    """Provenance and counts for one synthesized dataset.

    Attributes:
        book_id: Book the dataset was built from.
        created_at: UTC timestamp of the run.
        model: Teacher model (from the cache when pairs were reused).
        source: Whether pairs were generated now or reused from ``qa.jsonl``.
        index: Index identity at generation time (embedder, contextual, ...).
        config: Synthesis settings used for the run.
        chunk_count: Total chunks in the book.
        sampled_chunks: Chunks that were sampled for generation.
        generated_pairs: Pairs produced by the teacher (pre-filter).
        kept_pairs: Pairs that passed filtering.
        rejected: Rejection counts by reason.
        example_count: RAFT examples written.
        answerable_examples: Positive examples.
        unanswerable_examples: Negative ("not in the book") examples.
    """

    book_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str
    source: Literal["generated", "cache"]
    index: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    chunk_count: int = 0
    sampled_chunks: int = 0
    generated_pairs: int = 0
    kept_pairs: int = 0
    rejected: dict[str, int] = Field(default_factory=dict)
    example_count: int = 0
    answerable_examples: int = 0
    unanswerable_examples: int = 0


@dataclass(frozen=True, slots=True)
class SynthesisRun:
    """Everything one synth run produced (runtime container).

    Attributes:
        manifest: Dataset provenance and counts.
        pairs: All generated (or cached) pairs, pre-filter.
        outcome: Filter outcome (kept and rejected pairs).
        examples: RAFT examples ready for training.
    """

    manifest: DatasetManifest
    pairs: list[QAPair]
    outcome: FilterOutcome
    examples: list[RaftExample]


def sample_chunks(chunks: list[Chunk], *, max_chunks: int, seed: int) -> list[Chunk]:
    """Sample chunks deterministically, returned in reading order.

    Args:
        chunks: All chunks of the book.
        max_chunks: Maximum number of chunks to sample.
        seed: Random seed.

    Returns:
        The sampled chunks sorted by ordinal.
    """
    if max_chunks >= len(chunks):
        return list(chunks)
    rng = random.Random(seed)  # noqa: S311 - reproducibility, not cryptography
    return sorted(rng.sample(chunks, k=max_chunks), key=lambda chunk: chunk.ordinal)


def load_cached_pairs(path: Path) -> list[QAPair] | None:
    """Load cached QA pairs, or None when the file is missing or corrupt.

    Args:
        path: ``qa.jsonl`` written by a previous run.

    Returns:
        Validated pairs, or None to signal "regenerate".
    """
    if not path.exists():
        return None
    try:
        return [QAPair.model_validate(row) for row in read_jsonl(path)]
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable QA cache %s: %s", path, exc)
        return None


def synthesize(
    book: BookDocument,
    chunks: list[Chunk],
    llm: LLMClient,
    settings: SynthesisSettings,
    *,
    index_identity: dict[str, Any] | None = None,
    cached_pairs: list[QAPair] | None = None,
) -> SynthesisRun:
    """Run the full synthesis pipeline: sample, generate, filter, format.

    Args:
        book: Parsed book.
        chunks: All chunks of the book.
        llm: Teacher client (not called when ``cached_pairs`` is provided).
        settings: Synthesis settings (sampling, filtering, formatting).
        index_identity: Index identity recorded in the manifest.
        cached_pairs: Pairs from a previous run, skipping generation.

    Returns:
        The run: manifest, pairs, filter outcome and RAFT examples.
    """
    sampled = sample_chunks(chunks, max_chunks=settings.max_chunks, seed=settings.seed)

    if cached_pairs is not None:
        pairs = list(cached_pairs)
        source: Literal["generated", "cache"] = "cache"
        model = cached_pairs[0].model if cached_pairs else llm.name
    else:
        pairs = generate_pairs(
            llm, book, sampled, questions_per_chunk=settings.questions_per_chunk
        )
        source = "generated"
        model = llm.name

    outcome = filter_pairs(pairs, chunks_by_id={chunk.id: chunk for chunk in chunks})
    examples = build_examples(
        book.title,
        outcome.kept,
        chunks,
        distractors=settings.distractors,
        negative_ratio=settings.negative_ratio,
        seed=settings.seed,
    )

    manifest = DatasetManifest(
        book_id=book.book_id,
        model=model,
        source=source,
        index=dict(index_identity or {}),
        config=settings.model_dump(),
        chunk_count=len(chunks),
        sampled_chunks=len(sampled),
        generated_pairs=len(pairs),
        kept_pairs=len(outcome.kept),
        rejected=outcome.rejection_counts(),
        example_count=len(examples),
        answerable_examples=sum(1 for example in examples if example.answerable),
        unanswerable_examples=sum(1 for example in examples if not example.answerable),
    )
    return SynthesisRun(
        manifest=manifest, pairs=pairs, outcome=outcome, examples=examples
    )
