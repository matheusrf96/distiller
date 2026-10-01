"""Prompt construction for the thematic summary tree and global answering.

Summarization, map and reduce calls reuse the local RAG system prompt
(:func:`distiller.rag.prompts.build_system_prompt`), so grounding, ``[n]``
citations and the shared refusal sentence stay identical to local answers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tree import SummaryNode

__all__ = [
    "MAP_USER_PROMPT",
    "PROMPT_VERSION",
    "REDUCE_USER_PROMPT",
    "SUMMARY_SYSTEM_PROMPT",
    "SUMMARY_USER_PROMPT",
    "build_map_prompt",
    "build_reduce_prompt",
    "build_summary_prompt",
    "render_summary_documents",
]

# Bump when the summarization prompt changes so cached summaries regenerate.
PROMPT_VERSION = "thematic-v1"

SUMMARY_SYSTEM_PROMPT = (
    "You summarize parts of a book for a hierarchical summary tree. "
    "Reply with one concise paragraph only: no preamble, no markdown, "
    "no quoting of the source text."
)

SUMMARY_USER_PROMPT = """Book: {book_title}
Unit: {unit_title}

Text to summarize:
<source>
{source_text}
</source>

Summarize the text above in one concise paragraph of at most
{max_summary_chars} characters. Preserve the key claims, names and numbers a
reader would need to answer whole-book questions. Reply with the summary only."""

MAP_USER_PROMPT = """Summary documents:
{documents}

Question: {question}

Answer the question using only the summary documents above, with citations:"""

REDUCE_USER_PROMPT = """Partial answers:
{documents}

Question: {question}

Combine the partial answers into one final answer using only them, with
citations:"""


def build_summary_prompt(
    book_title: str,
    unit_title: str,
    source_text: str,
    *,
    max_summary_chars: int,
) -> str:
    """Build the user prompt asking for one node summary.

    Args:
        book_title: Title of the book being summarized.
        unit_title: Title of the chapter/window/root being summarized.
        source_text: Chapter text or child summaries to distill.
        max_summary_chars: Length budget stated in the instruction.

    Returns:
        User prompt containing the source as a ``<source>`` block.
    """
    return SUMMARY_USER_PROMPT.format(
        book_title=book_title,
        unit_title=unit_title,
        source_text=source_text,
        max_summary_chars=max_summary_chars,
    )


def render_summary_documents(nodes: list[SummaryNode]) -> str:
    """Render node summaries as ``<doc>`` blocks with node provenance."""
    blocks: list[str] = []
    for index, node in enumerate(nodes, start=1):
        attrs = [
            f'id="{index}"',
            f'node="{node.id}"',
            f'level="{node.level}"',
            f'title="{node.title}"',
        ]
        blocks.append(f"<doc {' '.join(attrs)}>\n{node.summary or ''}\n</doc>")
    return "\n\n".join(blocks)


def build_map_prompt(question: str, node: SummaryNode) -> str:
    """Build the user prompt asking for one partial answer from one summary."""
    return MAP_USER_PROMPT.format(
        documents=render_summary_documents([node]), question=question
    )


def build_reduce_prompt(
    question: str, nodes: list[SummaryNode], partials: list[str]
) -> str:
    """Build the user prompt reducing partial answers into a final answer.

    Partials are rendered as ``<doc>`` blocks in selection order, so marker
    ``[n]`` in the final answer maps back to ``nodes[n - 1]``.
    """
    blocks: list[str] = []
    for index, (node, partial) in enumerate(zip(nodes, partials, strict=True), start=1):
        attrs = [
            f'id="{index}"',
            f'node="{node.id}"',
            f'level="{node.level}"',
            f'title="{node.title}"',
        ]
        blocks.append(f"<doc {' '.join(attrs)}>\n{partial}\n</doc>")
    return REDUCE_USER_PROMPT.format(documents="\n\n".join(blocks), question=question)
