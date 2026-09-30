"""OpenAI-compatible LLM client (cloud APIs, Ollama, llama.cpp)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from ..config import LLMSettings


class OpenAICompatClient:
    """Chat client for any OpenAI-compatible endpoint.

    Attributes:
        model: Model name sent with each request.
        base_url: Endpoint base URL.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        """Configure the client.

        Args:
            model: Model name.
            base_url: Endpoint base URL; defaults to a local Ollama server.
            api_key: API key, when the endpoint requires one.
            temperature: Default sampling temperature.
            max_tokens: Default maximum tokens per completion.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.base_url = base_url or "http://localhost:11434/v1"
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key or "not-needed",
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        """Stable identifier of the client and model, for reports."""
        return f"openai-compat:{self.model}"

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a completion for a system + user message pair.

        Args:
            system: System prompt.
            user: User prompt.
            temperature: Sampling temperature override.
            max_tokens: Maximum tokens override.

        Returns:
            The generated text.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.default_temperature
            if temperature is None
            else temperature,
            max_tokens=self.default_max_tokens if max_tokens is None else max_tokens,
        )
        return response.choices[0].message.content or ""


def client_from_settings(settings: LLMSettings) -> OpenAICompatClient:
    """Build a client from LLM settings.

    Args:
        settings: LLM settings (model, base URL, key, limits).

    Returns:
        Configured client.
    """
    return OpenAICompatClient(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
    )
