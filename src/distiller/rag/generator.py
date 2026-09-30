"""Grounded answer generation with citation extraction and refusal handling."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..models import Answer, Citation
from .prompts import build_system_prompt, build_user_prompt, refusal_text

if TYPE_CHECKING:
    from ..llm.base import LLMClient
    from ..models import RetrievedChunk

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")
_REFUSAL_MARK = "couldn't find"


class Generator:
    """Turns retrieved contexts into a cited, grounded answer.

    Attributes:
        llm: Chat client used for generation.
        book_title: Title used in prompts and refusal messages.
    """

    def __init__(
        self,
        llm: LLMClient,
        book_title: str,
        *,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> None:
        """Configure the generator.

        Args:
            llm: Chat client used for generation.
            book_title: Title used in prompts and refusal messages.
            max_tokens: Maximum tokens per answer.
            temperature: Optional sampling temperature override.
        """
        self.llm = llm
        self.book_title = book_title
        self.max_tokens = max_tokens
        self.temperature = temperature

    def answer(self, question: str, contexts: list[RetrievedChunk]) -> Answer:
        """Answer a question from the supplied contexts.

        Args:
            question: User question.
            contexts: Retrieved chunks acting as the only allowed evidence.

        Returns:
            Grounded answer with citations, or an explicit refusal when there
            are no contexts.
        """
        if not contexts:
            return Answer(
                question=question,
                text=refusal_text(self.book_title),
                citations=[],
                contexts=[],
                model=self.llm.name,
                refused=True,
            )

        text = self.llm.complete(
            system=build_system_prompt(self.book_title),
            user=build_user_prompt(question, contexts),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = text.strip()
        citations = extract_citations(text, contexts)
        return Answer(
            question=question,
            text=text,
            citations=citations,
            contexts=contexts,
            model=self.llm.name,
            # A hedged answer that still cites the book is grounded, not a refusal.
            refused=is_refusal(text, self.book_title) and not citations,
        )


def extract_citations(text: str, contexts: list[RetrievedChunk]) -> list[Citation]:
    """Map ``[n]`` markers in the answer back to chunk provenance.

    Args:
        text: Generated answer text.
        contexts: Contexts in the order they were shown to the model.

    Returns:
        Citations for every valid marker, in ascending marker order.
    """
    seen = sorted({int(marker) for marker in _CITATION_RE.findall(text)})
    citations: list[Citation] = []
    for index in seen:
        if 1 <= index <= len(contexts):
            chunk = contexts[index - 1].chunk
            citations.append(
                Citation(
                    index=index,
                    chunk_id=chunk.id,
                    chapter=chunk.chapter,
                    heading_path=list(chunk.heading_path),
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                )
            )
    return citations


def is_refusal(text: str, book_title: str) -> bool:
    """Detect refusal wording in an answer (text-only heuristic).

    Callers combine this with citation presence: an answer that hedges but
    cites the book is grounded, not a refusal.
    """
    head = " ".join(text.split())[:200].lower()
    if _REFUSAL_MARK in head:
        return True
    return refusal_text(book_title).lower() in text.lower()
