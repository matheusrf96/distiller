"""Deterministic offline LLM used for tests, demos and CI.

Set ``DISTILLER_LLM__MODEL=fake`` to run the whole pipeline without any backend.
The default behavior answers with the first retrieved document (so citations and
end-to-end tests are meaningful) or refuses when no documents were provided.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_DOC_RE = re.compile(r'<doc id="1"[^>]*>\n(.*?)\n</doc>', re.DOTALL)
_EXCERPT_RE = re.compile(r"<excerpt>\n(.*?)\n</excerpt>", re.DOTALL)
_REFUSAL_TEMPLATE = "I couldn't find that in {title}."


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

        Contextual-enrichment prompts (``<excerpt>`` blocks) receive a
        deterministic short sentence so offline index builds exercise enrichment.

        Args:
            system: System prompt (unused by the fake).
            user: User prompt containing ``<doc>`` or ``<excerpt>`` blocks.
            temperature: Ignored.
            max_tokens: Ignored.

        Returns:
            Snippet answer with a ``[1]`` citation, a situating sentence, or the
            refusal sentence.
        """
        if callable(self._response):
            return self._response(user)
        if self._response is not None:
            return self._response

        excerpt = _EXCERPT_RE.search(user)
        if excerpt:
            snippet = " ".join(excerpt.group(1).split())[:80].rstrip(" .")
            return f"A passage about: {snippet}."

        match = _DOC_RE.search(user)
        if not match:
            return _REFUSAL_TEMPLATE.format(title=self._book_title)
        snippet = " ".join(match.group(1).split())
        if len(snippet) > 240:
            snippet = snippet[:240].rsplit(" ", 1)[0] + "…"
        return f'According to the book: "{snippet}" [1]'
