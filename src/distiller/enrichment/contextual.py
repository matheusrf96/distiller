"""LLM-generated situating context for chunks (contextual retrieval).

Implements the indexing-time half of Contextual Retrieval: before embedding and
lexical indexing, each chunk receives a 1-2 sentence context situating it in the
book. The synthetic context is used **only** for ``Chunk.index_text``; prompts,
answers and citations keep using the original ``Chunk.text``.

Contexts are cached per book (``artifacts/<book>/enrichment.jsonl``) so a re-index
performs no repeat LLM calls. The cache key is the chunk id, which embeds a
content hash, so re-chunking safely invalidates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import read_jsonl

if TYPE_CHECKING:
    from ..llm.base import LLMClient
    from ..models import BookDocument, Chapter, Chunk

logger = logging.getLogger(__name__)

DEFAULT_MAX_DOCUMENT_CHARS = 6000
DEFAULT_MAX_CONTEXT_CHARS = 500
DEFAULT_MIN_CONTEXT_CHARS = 10

SYSTEM_PROMPT = (
    "You situate excerpts of a book for a search index. "
    "Reply with one or two short sentences only: no preamble, no markdown, "
    "no quoting of the excerpt."
)

USER_PROMPT = """Book: {book_title}
Chapter: {chapter}
Section: {section}

Excerpt to situate:
<excerpt>
{chunk_text}
</excerpt>

Excerpt of the surrounding chapter:
<chapter>
{chapter_excerpt}
</chapter>

In one or two sentences, situate the excerpt in the book: name the chapter and
what the excerpt is about, so a search engine can match it to related questions.
Reply with those sentences only."""

_UNUSABLE_MARKERS = ("couldn't find", "cannot find", "i'm sorry", "as an ai")


class ContextualEnricher:
    """Adds cached, LLM-generated contexts to chunks before indexing.

    Attributes:
        llm: Chat client used for context generation.
        book: Book the chunks belong to (used for chapter excerpts).
        cache_path: Optional append-only cache file for generated contexts.
    """

    def __init__(
        self,
        llm: LLMClient,
        book: BookDocument,
        *,
        cache_path: Path | None = None,
        max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        min_context_chars: int = DEFAULT_MIN_CONTEXT_CHARS,
    ) -> None:
        """Prepare the enricher for one book.

        Args:
            llm: Chat client used for context generation.
            book: Parsed book providing chapter excerpts.
            cache_path: Optional enrichment cache file (read and appended).
            max_document_chars: Cap for the chapter excerpt sent to the LLM.
            max_context_chars: Cap for the generated context prefix.
            min_context_chars: Completions shorter than this are discarded.
        """
        self.llm = llm
        self.book = book
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.max_context_chars = max(1, max_context_chars)
        self.min_context_chars = max(1, min_context_chars)
        self._chapter_excerpts = {
            chapter.title: _truncate(_chapter_text(chapter), max_document_chars)
            for chapter in book.chapters
        }
        self._cache = self._load_cache()

    def enrich(self, chunks: list[Chunk]) -> list[Chunk]:
        """Return chunk copies with ``context`` filled where possible.

        Cache hits never call the LLM; failures leave the chunk unchanged and
        are logged, so one bad completion cannot abort an index build.

        Args:
            chunks: Chunks produced by the chunker.

        Returns:
            Chunks in input order, each with ``context`` set when usable.
        """
        enriched: list[Chunk] = []
        for chunk in chunks:
            context = self._cache.get(chunk.id)
            if context is None:
                context = self._generate(chunk)
                if context:
                    self._cache[chunk.id] = context
                    self._append_cache(chunk.id, context)
            enriched.append(
                chunk.model_copy(update={"context": context}) if context else chunk
            )
        return enriched

    def _generate(self, chunk: Chunk) -> str | None:
        user = USER_PROMPT.format(
            book_title=self.book.title,
            chapter=chunk.chapter,
            section=chunk.heading_path[-1] if chunk.heading_path else chunk.chapter,
            chunk_text=chunk.text,
            chapter_excerpt=self._chapter_excerpts.get(chunk.chapter, ""),
        )
        try:
            raw = self.llm.complete(
                system=SYSTEM_PROMPT, user=user, temperature=0.0, max_tokens=200
            )
        except Exception as exc:
            logger.warning("Context generation failed for %s: %s", chunk.id, exc)
            return None
        return self._normalize(raw)

    def _normalize(self, raw: str) -> str | None:
        text = " ".join(raw.split())
        if len(text) < self.min_context_chars:
            return None
        lowered = text.lower()
        if any(marker in lowered for marker in _UNUSABLE_MARKERS):
            return None
        return text[: self.max_context_chars].rstrip()

    def _load_cache(self) -> dict[str, str]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        try:
            return {
                str(row["chunk_id"]): str(row["context"])
                for row in read_jsonl(self.cache_path)
                if row.get("chunk_id") and row.get("context")
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Ignoring unreadable enrichment cache %s: %s", self.cache_path, exc
            )
            return {}

    def _append_cache(self, chunk_id: str, context: str) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"chunk_id": chunk_id, "context": context}) + "\n"
                )
        except OSError as exc:
            logger.warning(
                "Could not write enrichment cache %s: %s", self.cache_path, exc
            )


def _chapter_text(chapter: Chapter) -> str:
    return "\n\n".join(block.text for block in chapter.blocks)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip()
