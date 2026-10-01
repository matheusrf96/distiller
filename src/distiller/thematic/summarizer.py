"""Cached, per-node summarization for the thematic summary tree.

Summaries are cached per book (``artifacts/<book>/thematic/summaries.jsonl``) so
a re-run with unchanged inputs performs no LLM calls. The cache key embeds the
node content, the model and the prompt version, so re-chunking, switching models
or changing the summary budget regenerates cleanly. Only successful summaries
are cached, which is why a re-run retries exactly the holes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import read_jsonl, stable_hash_hex
from .prompts import PROMPT_VERSION, SUMMARY_SYSTEM_PROMPT, build_summary_prompt

if TYPE_CHECKING:
    from ..llm.base import LLMClient

logger = logging.getLogger(__name__)

MIN_SUMMARY_CHARS = 10
_UNUSABLE_MARKERS = ("i'm sorry", "as an ai", "cannot summarize", "couldn't find")


def summary_cache_key(
    node_id: str,
    source_text: str,
    model: str,
    *,
    max_source_chars: int,
    max_summary_chars: int,
) -> str:
    """Build the cache key for one node summary.

    Args:
        node_id: Deterministic node id (part of the content identity).
        source_text: Exact text sent to the model (already truncated).
        model: LLM name that produced (or would produce) the summary.
        max_source_chars: Source budget used for the prompt.
        max_summary_chars: Summary budget used for the prompt.

    Returns:
        Content-addressed cache key.
    """
    payload = "\x1f".join(
        [
            PROMPT_VERSION,
            model,
            node_id,
            str(max_source_chars),
            str(max_summary_chars),
            source_text,
        ]
    )
    return stable_hash_hex(payload)


def load_summary_cache(path: Path | None) -> dict[str, str]:
    """Load the append-only summary cache, ignoring corrupt files.

    Args:
        path: ``summaries.jsonl`` written by previous builds (or None).

    Returns:
        Mapping cache key -> summary; empty when the file is missing or corrupt.
    """
    if path is None or not path.exists():
        return {}
    try:
        cache: dict[str, str] = {}
        for row in read_jsonl(path):
            key = row.get("key")
            summary = row.get("summary")
            if key and summary:
                cache[str(key)] = str(summary)
        return cache
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Ignoring unreadable summary cache %s: %s", path, exc)
        return {}


class Summarizer:
    """Generates and caches one summary per tree node.

    Attributes:
        llm: Chat client used for summarization.
        cache_path: Optional append-only cache file for generated summaries.
        max_source_chars: Cap for the text sent to a summarization call.
        max_summary_chars: Cap for one generated summary.
        generated: Summaries produced in this run.
        reused: Summaries served from the cache in this run.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        cache_path: Path | None = None,
        max_source_chars: int = 6000,
        max_summary_chars: int = 1200,
    ) -> None:
        """Prepare the summarizer for one tree build.

        Args:
            llm: Chat client used for summarization.
            cache_path: Optional summary cache file (read and appended).
            max_source_chars: Cap for the text sent to a summarization call.
            max_summary_chars: Cap for one generated summary.
        """
        self.llm = llm
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.max_source_chars = max(1, max_source_chars)
        self.max_summary_chars = max(1, max_summary_chars)
        self.generated = 0
        self.reused = 0
        self._cache = load_summary_cache(self.cache_path)

    def summarize(
        self,
        node_id: str,
        *,
        book_title: str,
        unit_title: str,
        source_text: str,
        regenerate: bool = False,
    ) -> str | None:
        """Summarize one node, reusing the cache unless ``regenerate`` is set.

        Cache hits never call the LLM; failures (a raising call or an unusable
        completion) return None and are logged, so one bad completion cannot
        abort a build.

        Args:
            node_id: Deterministic node id.
            book_title: Title used in the prompt.
            unit_title: Chapter/window/root title used in the prompt.
            source_text: Chapter text or child summaries to distill.
            regenerate: Ignore the cache and call the LLM again.

        Returns:
            The summary, or None when the node failed.
        """
        source = _truncate(source_text.strip(), self.max_source_chars)
        if not source:
            logger.warning(
                "No source text for node %s; recording it as failed", node_id
            )
            return None

        key = summary_cache_key(
            node_id,
            source,
            self.llm.name,
            max_source_chars=self.max_source_chars,
            max_summary_chars=self.max_summary_chars,
        )
        if not regenerate:
            cached = self._cache.get(key)
            if cached is not None:
                self.reused += 1
                return cached

        user = build_summary_prompt(
            book_title,
            unit_title,
            source,
            max_summary_chars=self.max_summary_chars,
        )
        try:
            raw = self.llm.complete(
                system=SUMMARY_SYSTEM_PROMPT,
                user=user,
                temperature=0.0,
                max_tokens=500,
            )
        except Exception as exc:
            logger.warning("Summarization failed for %s: %s", node_id, exc)
            return None

        summary = self._normalize(raw)
        if summary is None:
            logger.warning("Unusable summary for %s; recording it as failed", node_id)
            return None
        self._cache[key] = summary
        self.generated += 1
        self._append_cache(key, node_id, summary)
        return summary

    def _normalize(self, raw: str) -> str | None:
        text = " ".join(raw.split())
        if len(text) < MIN_SUMMARY_CHARS:
            return None
        lowered = text.lower()
        if any(marker in lowered for marker in _UNUSABLE_MARKERS):
            return None
        return text[: self.max_summary_chars].rstrip()

    def _append_cache(self, key: str, node_id: str, summary: str) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"key": key, "node_id": node_id, "summary": summary})
                    + "\n"
                )
        except OSError as exc:
            logger.warning("Could not write summary cache %s: %s", self.cache_path, exc)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip()
