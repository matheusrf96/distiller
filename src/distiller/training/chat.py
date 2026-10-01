"""RAFT -> chat formatting for supervised fine-tuning (Phase 3, step 1).

Training data must match inference behaviour, so the system prompt, the
document rendering and the user prompt come from the shared RAG prompt
contract (``distiller.rag.prompts``) instead of being duplicated here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from ..exceptions import TrainingError
from ..models import DomainModel, RetrievedChunk
from ..rag import build_system_prompt, build_user_prompt
from ..utils import stable_hash_hex

if TYPE_CHECKING:
    from ..models import Chunk
    from ..synthesis import RaftExample

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")

__all__ = [
    "ChatMessage",
    "TrainingExample",
    "citation_indices",
    "format_example",
    "format_examples",
    "prompt_contract_hash",
]


class ChatMessage(DomainModel):
    """One chat turn in a training example.

    Attributes:
        role: Chat role (system, user or assistant).
        content: Message text.
    """

    role: Literal["system", "user", "assistant"]
    content: str


class TrainingExample(DomainModel):
    """One chat-formatted supervised example.

    Attributes:
        id: Stable example id carried over from the RAFT example.
        book_id: Owning book.
        question: Training question.
        messages: System, user and assistant turns, in order.
        answerable: False for "not in the book" negatives.
    """

    id: str
    book_id: str
    question: str
    messages: list[ChatMessage] = Field(default_factory=list)
    answerable: bool = True

    @property
    def target(self) -> str:
        """The assistant message content (empty when malformed)."""
        return self.messages[-1].content if self.messages else ""


def format_example(
    book_title: str,
    example: RaftExample,
    chunks_by_id: dict[str, Chunk],
) -> TrainingExample:
    """Format one RAFT example as a chat example.

    Contexts are resolved against ``chunks.jsonl`` so the ``<doc>`` blocks carry
    the chapter/section/page provenance the generator sees at inference time;
    their order (and therefore the ``[n]`` citation indices) is preserved.

    Args:
        book_title: Book title used by the shared prompt contract.
        example: RAFT example to format.
        chunks_by_id: Indexed chunks keyed by id.

    Returns:
        Chat example with system, user and assistant messages.

    Raises:
        TrainingError: When a context chunk id is absent from ``chunks.jsonl``.
    """
    contexts: list[RetrievedChunk] = []
    for context in example.contexts:
        chunk = chunks_by_id.get(context.chunk_id)
        if chunk is None:
            raise TrainingError(
                f"Training context '{context.chunk_id}' is not in chunks.jsonl. "
                f"Re-run `distiller index` and `distiller synth` to rebuild the "
                f"dataset against the current chunks."
            )
        contexts.append(RetrievedChunk(chunk=chunk, score=0.0))

    return TrainingExample(
        id=example.id,
        book_id=example.book_id,
        question=example.question,
        messages=[
            ChatMessage(role="system", content=build_system_prompt(book_title)),
            ChatMessage(
                role="user", content=build_user_prompt(example.question, contexts)
            ),
            ChatMessage(role="assistant", content=example.answer),
        ],
        answerable=example.answerable,
    )


def format_examples(
    book_title: str,
    examples: list[RaftExample],
    chunks: list[Chunk],
) -> list[TrainingExample]:
    """Format every RAFT example (see :func:`format_example`).

    Args:
        book_title: Book title used by the shared prompt contract.
        examples: RAFT examples to format.
        chunks: All chunks of the book.

    Returns:
        Chat examples in input order.
    """
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    return [format_example(book_title, example, chunks_by_id) for example in examples]


def prompt_contract_hash(book_title: str) -> str:
    """Hash the RAG prompt contract used for training.

    The system prompt for this book plus the static skeleton of the user prompt
    are hashed, so drift between the training prompt and the inference prompt is
    visible in the dataset manifest.

    Args:
        book_title: Book title substituted into the system prompt.

    Returns:
        Stable hash of the shared prompt contract.
    """
    contract = f"{build_system_prompt(book_title)}\n\n{build_user_prompt('', [])}"
    return stable_hash_hex(contract, 40)


def citation_indices(text: str) -> list[int]:
    """Return the ``[n]`` citation indices present in an answer, in order.

    Args:
        text: Answer text to scan.

    Returns:
        Citation indices as integers (duplicates preserved).
    """
    return [int(marker) for marker in _CITATION_RE.findall(text)]
