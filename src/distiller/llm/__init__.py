"""LLM factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import LLMSettings
from ..exceptions import ConfigurationError
from .base import LLMClient
from .fake import FakeLLM
from .openai_compat import client_from_settings

if TYPE_CHECKING:
    from ..config import GgufSettings, Settings

_ADAPTER_HINT = (
    "No adapter endpoint configured. Set DISTILLER_ADAPTER__MODEL and "
    "DISTILLER_ADAPTER__BASE_URL to the served LoRA adapter "
    "(or DISTILLER_ADAPTER__MODEL=fake for offline runs)."
)

_GGUF_HINT = (
    "No GGUF endpoint configured. Set DISTILLER_GGUF__BASE_URL to the served "
    "model, e.g. http://localhost:11434/v1 (Ollama) or "
    "http://localhost:8080/v1 (llama-server). Use "
    "DISTILLER_GGUF__MODEL=fake for offline runs."
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


def get_gguf_llm(settings: GgufSettings, model_name: str) -> LLMClient:
    """Return the client for the configured GGUF endpoint.

    Args:
        settings: GGUF endpoint settings.
        model_name: Resolved served model name (the registered
            ``distiller-<book-id>`` or the ``DISTILLER_GGUF__MODEL`` override).

    Returns:
        The deterministic offline client when ``model_name == "fake"``,
        otherwise an OpenAI-compatible client pointed at the served GGUF.

    Raises:
        ConfigurationError: When no GGUF endpoint is configured.
    """
    if model_name == "fake":
        return FakeLLM()
    if settings.base_url is None:
        raise ConfigurationError(_GGUF_HINT)
    return client_from_settings(
        LLMSettings(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=model_name,
            timeout=settings.timeout,
        )
    )


__all__ = [
    "FakeLLM",
    "LLMClient",
    "get_adapter_llm",
    "get_gguf_llm",
    "get_llm",
]
