"""Global answering: summary selection plus one map-reduce pass.

Whole-book questions are answered by mapping over the most relevant tree
summaries (the top ``map_top_k`` by embedding similarity when the index
embedder is available, otherwise every chapter summary in reading order) and
reducing the partial answers into one grounded answer. The system prompt,
``<doc>`` evidence blocks and refusal sentence are shared with the local RAG
pipeline.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..models import Answer, SummaryCitation, refusal_text
from ..rag.generator import is_refusal
from ..rag.prompts import build_system_prompt
from .prompts import build_map_prompt, build_reduce_prompt

if TYPE_CHECKING:
    import numpy as np

    from ..config import ThematicSettings
    from ..indexing.embedder import Embedder
    from ..llm.base import LLMClient
    from .tree import SummaryNode, SummaryTree

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")

__all__ = ["GlobalPipeline", "extract_summary_citations"]


class GlobalPipeline:
    """Answer whole-book questions by map-reducing over the summary tree.

    Attributes:
        tree: Summary tree to select from.
        llm: Chat client used for map and reduce calls.
        settings: Thematic settings (``map_top_k``).
        embedder: Optional index embedder used for summary selection.
        book_title: Title used in prompts and the refusal sentence.
        max_tokens: Maximum tokens per map/reduce completion.
    """

    def __init__(
        self,
        tree: SummaryTree,
        llm: LLMClient,
        settings: ThematicSettings,
        *,
        embedder: Embedder | None = None,
        book_title: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        """Assemble the global pipeline for one book.

        Args:
            tree: Summary tree built by ``distiller tree build``.
            llm: Chat client used for map and reduce calls.
            settings: Thematic settings (``map_top_k``).
            embedder: Optional index embedder; without it every chapter
                summary is mapped in reading order.
            book_title: Title override; defaults to the root node title.
            max_tokens: Maximum tokens per map/reduce completion.
        """
        self.tree = tree
        self.llm = llm
        self.settings = settings
        self.embedder = embedder
        self.book_title = book_title or tree.root.title
        self.max_tokens = max_tokens

    def ask(self, question: str) -> Answer:
        """Answer one whole-book question from the selected summaries.

        Args:
            question: Natural-language question.

        Returns:
            Global answer with summary citations, or the shared refusal when
            no summary-bearing node is available.
        """
        nodes = self.select(question)
        if not nodes:
            return Answer(
                question=question,
                text=refusal_text(self.book_title),
                model=self.llm.name,
                refused=True,
                mode="global",
            )

        partials = [self._map(question, node) for node in nodes]
        text = self._reduce(question, nodes, partials).strip()
        citations = extract_summary_citations(text, nodes)
        return Answer(
            question=question,
            text=text,
            summary_citations=citations,
            model=self.llm.name,
            refused=is_refusal(text, self.book_title) and not citations,
            mode="global",
        )

    def select(self, question: str) -> list[SummaryNode]:
        """Select the summary nodes mapped over for a question.

        With an embedder, the top ``map_top_k`` summary-bearing nodes across
        all levels by similarity; without one, every summary-bearing chapter
        node in reading order.

        Args:
            question: Natural-language question.

        Returns:
            Selected nodes in map order; empty when no node has a summary.
        """
        candidates = [node for node in self.tree.nodes if node.summary]
        if not candidates:
            return []
        if self.embedder is None:
            return [node for node in candidates if node.level == 1]

        query_vector = self.embedder.embed_query(question)
        summary_vectors = self.embedder.embed_documents(
            [node.summary or "" for node in candidates]
        )
        scored = sorted(
            zip(candidates, summary_vectors, strict=True),
            key=lambda item: _similarity(query_vector, item[1]),
            reverse=True,
        )
        return [node for node, _ in scored[: self.settings.map_top_k]]

    def _map(self, question: str, node: SummaryNode) -> str:
        return self.llm.complete(
            system=build_system_prompt(self.book_title),
            user=build_map_prompt(question, node),
            max_tokens=self.max_tokens,
        ).strip()

    def _reduce(
        self, question: str, nodes: list[SummaryNode], partials: list[str]
    ) -> str:
        return self.llm.complete(
            system=build_system_prompt(self.book_title),
            user=build_reduce_prompt(question, nodes, partials),
            max_tokens=self.max_tokens,
        )


def extract_summary_citations(
    text: str, nodes: list[SummaryNode]
) -> list[SummaryCitation]:
    """Map ``[n]`` markers in a global answer back to summary nodes.

    Args:
        text: Generated answer text.
        nodes: Selected nodes in the order they were shown to the model.

    Returns:
        Citations for every valid marker, in ascending marker order.
    """
    seen = sorted({int(marker) for marker in _CITATION_RE.findall(text)})
    citations: list[SummaryCitation] = []
    for index in seen:
        if 1 <= index <= len(nodes):
            node = nodes[index - 1]
            citations.append(
                SummaryCitation(
                    index=index, node_id=node.id, title=node.title, level=node.level
                )
            )
    return citations


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity for normalized vectors (a dot product)."""
    return float(sum(float(a) * float(b) for a, b in zip(left, right, strict=False)))
