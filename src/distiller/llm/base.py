"""LLM client protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat-completion interface satisfied by any OpenAI-compatible backend."""

    @property
    def name(self) -> str: ...

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a single completion.

        Args:
            system: System prompt.
            user: User prompt.
            temperature: Sampling temperature override.
            max_tokens: Maximum tokens override.

        Returns:
            The generated text.
        """
        ...
