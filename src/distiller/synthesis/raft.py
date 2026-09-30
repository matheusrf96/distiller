"""RAFT training-example formatting (Phase 2, step 3)."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from pydantic import Field

from ..models import DomainModel, refusal_text

if TYPE_CHECKING:
    from ..models import Chunk
    from .qa import QAPair


class RaftContext(DomainModel):
    """One context document inside a RAFT example.

    Attributes:
        chunk_id: Source chunk id.
        text: Chunk text as shown to the model.
        is_golden: True for the chunk that answers the question.
    """

    chunk_id: str
    text: str
    is_golden: bool = False


class RaftExample(DomainModel):
    """One supervised training example in RAFT format.

    Attributes:
        id: Stable example id (derived from the QA pair).
        book_id: Owning book.
        question: Training question.
        contexts: Golden and distractor contexts, in presentation order.
        answer: Target answer (quoted evidence plus answer, or the refusal).
        answerable: False for "not in the book" negatives.
    """

    id: str
    book_id: str
    question: str
    contexts: list[RaftContext] = Field(default_factory=list)
    answer: str
    answerable: bool = True


def build_examples(
    book_title: str,
    pairs: list[QAPair],
    chunks: list[Chunk],
    *,
    distractors: int,
    negative_ratio: float,
    seed: int,
) -> list[RaftExample]:
    """Turn filtered pairs into RAFT examples with distractors and negatives.

    Args:
        book_title: Book title (used in the refusal targets).
        pairs: Filtered QA pairs.
        chunks: All chunks of the book (the distractor pool).
        distractors: Distractor contexts per example.
        negative_ratio: Probability that a pair becomes an unanswerable example.
        seed: Random seed for distractor choice and context order.

    Returns:
        One example per pair, in input order; deterministic for a given seed.
    """
    rng = random.Random(seed)  # noqa: S311 - reproducibility, not cryptography
    examples: list[RaftExample] = []
    for pair in pairs:
        golden = next((chunk for chunk in chunks if chunk.id == pair.chunk_id), None)
        pool = [chunk for chunk in chunks if chunk.id != pair.chunk_id]
        picked = rng.sample(pool, k=min(distractors, len(pool)))

        if golden is not None and rng.random() >= negative_ratio:
            examples.append(_positive_example(pair, golden, picked, rng))
        else:
            examples.append(_negative_example(pair, book_title, picked))
    return examples


def _positive_example(
    pair: QAPair,
    golden: Chunk,
    distractors: list[Chunk],
    rng: random.Random,
) -> RaftExample:
    contexts = [RaftContext(chunk_id=golden.id, text=golden.text, is_golden=True)]
    contexts.extend(
        RaftContext(chunk_id=chunk.id, text=chunk.text) for chunk in distractors
    )
    rng.shuffle(contexts)

    citation_index = next(
        index for index, context in enumerate(contexts, start=1) if context.is_golden
    )
    primary_quote = pair.quotes[0] if pair.quotes else ""
    answer = (
        f'The relevant passage says: "{primary_quote}" '
        f"[{citation_index}]\n\n{pair.answer}"
    )
    return RaftExample(
        id=f"{pair.id}:raft",
        book_id=pair.book_id,
        question=pair.question,
        contexts=contexts,
        answer=answer,
        answerable=True,
    )


def _negative_example(
    pair: QAPair,
    book_title: str,
    distractors: list[Chunk],
) -> RaftExample:
    contexts = [
        RaftContext(chunk_id=chunk.id, text=chunk.text) for chunk in distractors
    ]
    return RaftExample(
        id=f"{pair.id}:raft",
        book_id=pair.book_id,
        question=pair.question,
        contexts=contexts,
        answer=refusal_text(book_title),
        answerable=False,
    )
