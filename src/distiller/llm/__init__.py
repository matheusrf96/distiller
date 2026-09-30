"""LLM factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import LLMClient
from .fake import FakeLLM
from .openai_compat import client_from_settings

if TYPE_CHECKING:
    from ..config import Settings


def get_llm(settings: Settings) -> LLMClient:
    """Return the configured LLM client.

    Args:
        settings: Pipeline settings.

    Returns:
        The deterministic offline client when ``llm.model == "fake"``,
        otherwise an OpenAI-compatible client.
    """
    if settings.llm.model == "fake":
        return FakeLLM()
    return client_from_settings(settings.llm)


__all__ = ["FakeLLM", "LLMClient", "get_llm"]
