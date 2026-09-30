"""Unit tests for embedders and optional-dependency loading."""

from __future__ import annotations

import numpy as np
import pytest

from distiller.config import EmbeddingSettings
from distiller.exceptions import MissingDependencyError
from distiller.indexing.embedder import HashingEmbedder, get_embedder
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
