"""Unit tests for ingestion error paths and the LLM clients."""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import pytest

from distiller.config import GgufSettings, LLMSettings, Settings
from distiller.exceptions import (
    BookNotFoundError,
    ConfigurationError,
    IngestError,
    MissingDependencyError,
)
from distiller.ingest import ingest_book
from distiller.llm import get_gguf_llm, get_llm
from distiller.llm.fake import FakeLLM
from distiller.llm.openai_compat import OpenAICompatClient, client_from_settings
from distiller.optional_deps import is_available

if TYPE_CHECKING:
    from pathlib import Path


def test_ingest_missing_file_raises(settings: Settings, tmp_path: Path) -> None:
    """A missing path is reported as BookNotFoundError."""
    with pytest.raises(BookNotFoundError, match="not found"):
        ingest_book(tmp_path / "missing.epub", settings=settings)


def test_ingest_unsupported_suffix_raises(settings: Settings, tmp_path: Path) -> None:
    """Unsupported formats are rejected with the supported list."""
    source = tmp_path / "book.mobi"
    source.write_text("content", encoding="utf-8")

    with pytest.raises(IngestError, match="Unsupported file type"):
        ingest_book(source, settings=settings)


def test_ingest_empty_text_raises(settings: Settings, tmp_path: Path) -> None:
    """A file with no content raises IngestError."""
    source = tmp_path / "empty.txt"
    source.write_text("\n\n   \n", encoding="utf-8")

    with pytest.raises(IngestError, match="No content"):
        ingest_book(source, settings=settings)


def test_explicit_pdf_backend_requires_extra(
    settings: Settings, sample_pdf: Path
) -> None:
    """Choosing an uninstalled PDF backend names the extra to install."""
    if is_available("docling"):
        pytest.skip("docling is installed in this environment")

    settings.ingest.pdf_backend = "docling"
    with pytest.raises(MissingDependencyError, match="--extra pdf-ai"):
        ingest_book(sample_pdf, settings=settings)


def test_fake_llm_supports_fixed_and_callable_responses() -> None:
    """The fake LLM answers with fixed text or a callable of the prompt."""
    fixed = FakeLLM(response="canned answer")
    assert fixed.name == "fake"
    assert fixed.complete(system="s", user="u") == "canned answer"

    callable_response = FakeLLM(response=lambda user: f"len={len(user)}")
    assert callable_response.complete(system="s", user="abcd") == "len=4"


def test_fake_llm_refuses_without_documents() -> None:
    """No <doc> blocks in the prompt produce the refusal sentence."""
    answer = FakeLLM(book_title="The Book").complete(system="s", user="Question: q?")

    assert answer == "I couldn't find that in The Book."


def test_get_llm_selects_fake_or_client() -> None:
    """The sentinel model name selects the offline client."""
    fake_settings = Settings(llm=LLMSettings(model="fake"))
    assert isinstance(get_llm(fake_settings), FakeLLM)

    real_settings = Settings(llm=LLMSettings(model="gpt-4o-mini"))
    client = get_llm(real_settings)
    assert isinstance(client, OpenAICompatClient)
    assert client.name == "openai-compat:gpt-4o-mini"
    assert client.base_url == "http://localhost:11434/v1"


def test_get_gguf_llm_selects_fake_or_client() -> None:
    """The GGUF factory resolves the sentinel, name and endpoint (AC5)."""
    assert isinstance(get_gguf_llm(GgufSettings(), "fake"), FakeLLM)

    with pytest.raises(ConfigurationError, match="DISTILLER_GGUF__BASE_URL"):
        get_gguf_llm(GgufSettings(), "distiller-book")

    client = get_gguf_llm(
        GgufSettings(base_url="http://localhost:8080/v1"), "distiller-book"
    )
    assert isinstance(client, OpenAICompatClient)
    assert client.model == "distiller-book"
    assert client.base_url == "http://localhost:8080/v1"


def test_client_from_settings_maps_fields() -> None:
    """Settings fields flow into the client configuration."""
    client = client_from_settings(
        LLMSettings(
            model="local-model", base_url="http://localhost:1234/v1", api_key="k"
        )
    )

    assert client.model == "local-model"
    assert client.base_url == "http://localhost:1234/v1"


def test_connection_errors_become_configuration_errors() -> None:
    """An unreachable endpoint becomes a ConfigurationError with a hint (AC14)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    client = OpenAICompatClient(
        model="local-model", base_url=f"http://127.0.0.1:{port}/v1"
    )

    with pytest.raises(ConfigurationError) as excinfo:
        client.complete(system="s", user="u")

    message = str(excinfo.value)
    assert f"127.0.0.1:{port}" in message
    assert "ollama serve" in message
    assert "serve.sh" in message
