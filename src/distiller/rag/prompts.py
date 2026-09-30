"""Prompt construction for grounded, citation-enforcing book QA."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import refusal_text

if TYPE_CHECKING:
    from ..models import RetrievedChunk

__all__ = [
    "SYSTEM_TEMPLATE",
    "build_system_prompt",
    "build_user_prompt",
    "refusal_text",
    "render_documents",
]

SYSTEM_TEMPLATE = """You are a meticulous research assistant for the book "{title}".

Rules:
1. Answer ONLY using the provided documents. Never use outside knowledge.
2. Cite every document you rely on with its id in square brackets, e.g. [1]. \
Place the citation directly after the claim it supports.
3. When the book's exact wording matters, quote it verbatim inside double quotes.
4. If the documents do not contain the answer, reply exactly: "{refusal}"
5. Be concise and precise. Prefer short paragraphs or bullet points over long prose."""


def build_system_prompt(book_title: str) -> str:
    """Build the system prompt enforcing grounding, citations and refusal."""
    return SYSTEM_TEMPLATE.format(title=book_title, refusal=refusal_text(book_title))


def render_documents(contexts: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as ``<doc>`` blocks with provenance attributes."""
    blocks: list[str] = []
    for index, context in enumerate(contexts, start=1):
        chunk = context.chunk
        attrs = [f'id="{index}"', f'chapter="{chunk.chapter}"']
        if chunk.heading_path:
            attrs.append(f'section="{chunk.heading_path[-1]}"')
        if chunk.page_start:
            pages = (
                str(chunk.page_start)
                if chunk.page_start == chunk.page_end
                else f"{chunk.page_start}-{chunk.page_end}"
            )
            attrs.append(f'pages="{pages}"')
        blocks.append(f"<doc {' '.join(attrs)}>\n{chunk.text}\n</doc>")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    """Build the user prompt: documents, question and citation instruction."""
    return (
        "Documents:\n"
        f"{render_documents(contexts)}\n\n"
        f"Question: {question}\n\n"
        "Answer with citations:"
    )
