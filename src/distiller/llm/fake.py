"""Deterministic offline LLM used for tests, demos and CI.

Set ``DISTILLER_LLM__MODEL=fake`` to run the whole pipeline without any backend.
The default behavior answers with the first retrieved document (so citations and
end-to-end tests are meaningful) or refuses when no documents were provided.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_DOC_RE = re.compile(r'<doc id="1"[^>]*>\n(.*?)\n</doc>', re.DOTALL)
_EXCERPT_RE = re.compile(r"<excerpt>\n(.*?)\n</excerpt>", re.DOTALL)
_CHUNK_RE = re.compile(r"<chunk>\n(.*?)\n</chunk>", re.DOTALL)
_SOURCE_RE = re.compile(r"<source>\n(.*?)\n</source>", re.DOTALL)
_QUESTION_COUNT_RE = re.compile(r"Write exactly (\d+) question")
_REFUSAL_TEMPLATE = "I couldn't find that in {title}."

_QUESTION_TEMPLATES = (
    "What does the passage say about {keyword}?",
    "Why does the passage mention {keyword}?",
    "How is {keyword} described in the book?",
    "What happens in the part about {keyword}?",
    "Which details does the passage give about {keyword}?",
)


class FakeLLM:
    """Deterministic offline LLM used for tests, demos and CI."""

    def __init__(
        self,
        response: str | Callable[[str], str] | None = None,
        *,
        book_title: str = "the book",
    ) -> None:
        """Optionally pin the response text or a callable producing it.

        Args:
            response: Fixed answer, or callable receiving the user prompt.
            book_title: Title used in the refusal sentence.
        """
        self._response = response
        self._book_title = book_title

    @property
    def name(self) -> str:
        """Stable identifier, recorded on answers."""
        return "fake"

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Answer with the first retrieved document, or refuse when none exist.

        Deterministic branches:

        * ``<excerpt>`` blocks (contextual enrichment) → one situating sentence;
        * ``<chunk>`` blocks (synthetic QA) → a JSON array of grounded pairs;
        * ``<source>`` blocks (thematic summaries) → one summary sentence;
        * ``<doc>`` blocks (book QA) → the first document with a ``[1]`` citation;
        * anything else → the refusal sentence.

        Args:
            system: System prompt (unused by the fake).
            user: User prompt containing one of the block markers above.
            temperature: Ignored.
            max_tokens: Ignored.

        Returns:
            A deterministic completion for the detected prompt type.
        """
        if callable(self._response):
            return self._response(user)
        if self._response is not None:
            return self._response

        excerpt = _EXCERPT_RE.search(user)
        if excerpt:
            snippet = " ".join(excerpt.group(1).split())[:80].rstrip(" .")
            return f"A passage about: {snippet}."

        chunk = _CHUNK_RE.search(user)
        if chunk:
            return self._qa_json(" ".join(chunk.group(1).split()), user)

        source = _SOURCE_RE.search(user)
        if source:
            snippet = " ".join(source.group(1).split())[:160].rstrip(" .")
            return f"This part of the book covers: {snippet}."

        match = _DOC_RE.search(user)
        if not match:
            return _REFUSAL_TEMPLATE.format(title=self._book_title)
        snippet = " ".join(match.group(1).split())
        if len(snippet) > 240:
            snippet = snippet[:240].rsplit(" ", 1)[0] + "…"
        return f'According to the book: "{snippet}" [1]'

    def _qa_json(self, text: str, user: str) -> str:
        """Build a deterministic JSON array of grounded QA pairs."""
        count_match = _QUESTION_COUNT_RE.search(user)
        count = int(count_match.group(1)) if count_match else 1
        keyword = text[:30]
        quote = text[:60]
        payload = [
            {
                "question": _QUESTION_TEMPLATES[
                    index % len(_QUESTION_TEMPLATES)
                ].format(keyword=keyword),
                "answer": f"According to the passage: {quote}",
                "quotes": [quote],
            }
            for index in range(count)
        ]
        return json.dumps(payload)
