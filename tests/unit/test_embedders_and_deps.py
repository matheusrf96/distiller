"""Unit tests for embedders and optional-dependency loading."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from distiller.config import EmbeddingSettings
from distiller.exceptions import ConfigurationError, MissingDependencyError
from distiller.indexing.embedder import (
    QWEN3_QUERY_PROMPT,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)
from distiller.optional_deps import is_available, require


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    """Vectors are stable across calls and unit-normalized."""
    embedder = HashingEmbedder(dim=64)
    first = embedder.embed_query("the lantern cracked during the storm")
    second = embedder.embed_query("the lantern cracked during the storm")

    assert np.allclose(first, second)
    assert np.isclose(float(np.linalg.norm(first)), 1.0)
    assert embedder.name == "hash:64"
    assert embedder.dim == 64


def test_hashing_embedder_similar_text_scores_higher() -> None:
    """Lexically overlapping text is closer than unrelated text."""
    embedder = HashingEmbedder(dim=256)
    query = embedder.embed_query("lantern storm")
    related = embedder.embed_documents(["the lantern cracked in the storm"])[0]
    unrelated = embedder.embed_documents(["sailing race results"])[0]

    assert float(query @ related) > float(query @ unrelated)


def test_hashing_embedder_handles_empty_input() -> None:
    """Empty document lists produce an empty matrix."""
    embedder = HashingEmbedder(dim=64)
    matrix = embedder.embed_documents([])

    assert matrix.shape == (0, 64)


def test_get_embedder_honours_backend() -> None:
    """The hash backend builds the offline embedder with the configured dims."""
    embedder = get_embedder(EmbeddingSettings(backend="hash", hash_dim=128))

    assert isinstance(embedder, HashingEmbedder)
    assert embedder.dim == 128


def test_get_embedder_requires_embed_extra() -> None:
    """Without sentence-transformers the error names the extra to install."""
    if is_available("sentence_transformers"):
        pytest.skip("sentence-transformers is installed in this environment")

    with pytest.raises(MissingDependencyError, match="--extra embed"):
        get_embedder(EmbeddingSettings(backend="sentence-transformers"))


def test_is_available_distinguishes_modules() -> None:
    """Availability checks never import: stdlib true, nonsense false."""
    assert is_available("json") is True
    assert is_available("definitely_not_a_real_module_xyz") is False


def test_require_caches_and_reports_missing_dependency() -> None:
    """require() caches imports and raises a fully described error."""
    first = require("json", extra="none", purpose="testing")
    second = require("json", extra="none", purpose="testing")
    assert first is second

    with pytest.raises(MissingDependencyError) as excinfo:
        require(
            "definitely_not_a_real_module_xyz", extra="dev", purpose="Testing purpose"
        )

    assert excinfo.value.module == "definitely_not_a_real_module_xyz"
    assert excinfo.value.extra == "dev"
    assert "Testing purpose" in str(excinfo.value)


class FakeSentenceTransformer:
    """Deterministic stand-in for sentence_transformers.SentenceTransformer."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def get_sentence_embedding_dimension(self) -> int:
        return 8

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        base = np.arange(1.0, 9.0, dtype=np.float32)
        return np.tile(base, (len(texts), 1))


def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the embedder's require() to the fake module."""
    module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setattr(
        "distiller.indexing.embedder.require", lambda name, **kwargs: module
    )


def test_hashing_embedder_empty_text_is_all_zeros() -> None:
    """Text without tokens has no direction and stays unnormalized."""
    vector = HashingEmbedder(dim=64).embed_query("")

    assert not vector.any()


def test_sentence_transformer_embedder_loads_and_truncates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loaded model encodes documents and queries with MRL truncation."""
    fake_sentence_transformers(monkeypatch)
    embedder = SentenceTransformerEmbedder(
        "Qwen/Qwen3-Embedding-0.6B", dim=4, batch_size=2
    )

    assert embedder.name == "st:Qwen/Qwen3-Embedding-0.6B:4"
    assert embedder.query_prompt == QWEN3_QUERY_PROMPT
    documents = embedder.embed_documents(["one", "two"])
    assert documents.shape == (2, 4)
    assert np.allclose(np.linalg.norm(documents, axis=1), 1.0)
    assert embedder.embed_documents([]).shape == (0, 4)
    assert embedder.embed_query("lantern").shape == (4,)


def test_sentence_transformer_embedder_keeps_native_dim_without_a_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-Qwen model keeps its native dim and has no query prompt."""
    fake_sentence_transformers(monkeypatch)
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3")

    assert embedder.dim == 8
    assert embedder.query_prompt is None
    assert embedder.embed_documents(["one"]).shape == (1, 8)
    assert embedder.embed_query("lantern").shape == (8,)


def test_sentence_transformer_embedder_rejects_oversized_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requested dim above the native dim is a configuration error."""
    fake_sentence_transformers(monkeypatch)

    with pytest.raises(ConfigurationError, match="exceeds native dim"):
        SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-0.6B", dim=16)
