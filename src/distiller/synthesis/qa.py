"""Grounded question-answer generation from book chunks (Phase 2, step 1)."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..models import DomainModel
from ..utils import stable_hash_hex

if TYPE_CHECKING:
    from ..llm.base import LLMClient
    from ..models import BookDocument, Chunk

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = (
    "You write grounded question-answer training data from book passages. "
    "Every answer must be fully supported by the passage. "
    "Respond with a JSON array only: no prose, no markdown fences."
)

__all__ = [
    "QA_SYSTEM_PROMPT",
    "QA_USER_PROMPT",
    "QAPair",
    "generate_pairs",
    "parse_pairs",
]

QA_USER_PROMPT = """Book: {book_title}
Chapter: {chapter}
Section: {section}

Passage:
<chunk>
{chunk_text}
</chunk>

Write exactly {count} question(s) that the passage above answers, each with a
concise answer and the verbatim quotes from the passage that support it.
Respond with a JSON array of objects:
[{{"question": "...", "answer": "...", "quotes": ["verbatim span"]}}]"""


class QAPair(DomainModel):
    """One generated question-answer pair grounded in a chunk.

    Attributes:
        id: Stable id derived from the chunk id and the question.
        book_id: Owning book.
        chunk_id: Chunk the pair was generated from (the golden chunk).
        chapter: Chapter title for provenance.
        heading_path: Heading stack at the source chunk.
        question: Generated question.
        answer: Generated answer, supported by ``quotes``.
        quotes: Verbatim spans from the source chunk.
        model: Teacher model that produced the pair.
    """

    id: str
    book_id: str
    chunk_id: str
    chapter: str
    heading_path: list[str] = Field(default_factory=list)
    question: str
    answer: str
    quotes: list[str] = Field(default_factory=list)
    model: str


def parse_pairs(text: str) -> list[dict[str, Any]]:
    """Extract pair dictionaries from an LLM completion.

    Args:
        text: Raw completion; expected to contain a JSON array.

    Returns:
        Valid items (non-empty ``question`` and ``answer``); an empty list when
        the completion cannot be parsed at all.
    """
    payload = _extract_json_array(text)
    if payload is None:
        logger.warning("Skipping unparseable QA completion: %s", _preview(text))
        return []
    return [item for item in payload if _is_valid_item(item)]


def generate_pairs(
    llm: LLMClient,
    book: BookDocument,
    chunks: list[Chunk],
    *,
    questions_per_chunk: int,
) -> list[QAPair]:
    """Generate grounded pairs for every chunk (one LLM call per chunk).

    Args:
        llm: Teacher client.
        book: Parsed book (title for the prompt).
        chunks: Chunks to generate from.
        questions_per_chunk: Questions requested per chunk.

    Returns:
        Pairs in chunk order; chunks whose call fails or whose completion is
        unparseable are skipped with a warning.
    """
    pairs: list[QAPair] = []
    for chunk in chunks:
        user = QA_USER_PROMPT.format(
            book_title=book.title,
            chapter=chunk.chapter,
            section=chunk.heading_path[-1] if chunk.heading_path else chunk.chapter,
            chunk_text=chunk.text,
            count=questions_per_chunk,
        )
        try:
            raw = llm.complete(
                system=QA_SYSTEM_PROMPT, user=user, temperature=0.2, max_tokens=800
            )
        except Exception as exc:
            logger.warning("QA generation failed for %s: %s", chunk.id, exc)
            continue

        for item in parse_pairs(raw):
            question = str(item["question"]).strip()
            pairs.append(
                QAPair(
                    id=f"{chunk.id}:{stable_hash_hex(question, 12)}",
                    book_id=chunk.book_id,
                    chunk_id=chunk.id,
                    chapter=chunk.chapter,
                    heading_path=list(chunk.heading_path),
                    question=question,
                    answer=str(item["answer"]).strip(),
                    quotes=_clean_quotes(item.get("quotes")),
                    model=llm.name,
                )
            )
    return pairs


def _clean_quotes(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [
        str(quote).strip()
        for quote in raw
        if isinstance(quote, str) and str(quote).strip()
    ]


def _extract_json_array(text: str) -> list[Any] | None:
    try:
        parsed = json.loads(text)
    except ValueError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except ValueError:
            return None
    return parsed if isinstance(parsed, list) else None


def _is_valid_item(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("question"), str)
        and bool(item["question"].strip())
        and isinstance(item.get("answer"), str)
        and bool(item["answer"].strip())
    )


def _preview(text: str, limit: int = 80) -> str:
    return " ".join(text.split())[:limit]
