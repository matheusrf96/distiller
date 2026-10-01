"""LLM factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import LLMSettings
from ..exceptions import ConfigurationError
from .base import LLMClient
from .fake import FakeLLM
from .openai_compat import client_from_settings

if TYPE_CHECKING:
    from ..config import Settings

_ADAPTER_HINT = (
    "No adapter endpoint configured. Set DISTILLER_ADAPTER__MODEL and "
    "DISTILLER_ADAPTER__BASE_URL to the served LoRA adapter "
    "(or DISTILLER_ADAPTER__MODEL=fake for offline runs)."
)


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


def get_adapter_llm(settings: Settings) -> LLMClient:
    """Return the client for the configured adapter endpoint.

    Args:
        settings: Pipeline settings.

    Returns:
        The deterministic offline client when ``adapter.model == "fake"``,
        otherwise an OpenAI-compatible client pointed at the served adapter.

    Raises:
        ConfigurationError: When no adapter endpoint is configured.
    """
    adapter = settings.adapter
    if adapter.model is None:
        raise ConfigurationError(_ADAPTER_HINT)
    if adapter.model == "fake":
        return FakeLLM()
    return client_from_settings(
        LLMSettings(
            base_url=adapter.base_url,
            api_key=adapter.api_key,
            model=adapter.model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout=adapter.timeout,
        )
    )


__all__ = ["FakeLLM", "LLMClient", "get_adapter_llm", "get_llm"]
