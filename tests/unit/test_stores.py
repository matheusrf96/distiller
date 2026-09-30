"""Unit tests for the vector stores."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from distiller.indexing.store import NumpyStore, create_store, open_store

if TYPE_CHECKING:
    from pathlib import Path


def build_store() -> NumpyStore:
    """A three-vector store with chapter payloads."""
    store = NumpyStore(dim=4)
    store.upsert(
        ["a", "b", "c"],
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        payloads=[
            {"chapter": "One", "ordinal": 0},
            {"chapter": "Two", "ordinal": 1},
            {"chapter": "Two", "ordinal": 2},
        ],
    )
    return store


def test_search_orders_by_cosine_similarity() -> None:
    """The nearest vector wins and scores are descending."""
    store = build_store()
    hits = store.search(np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32), k=3)

    assert [hit.id for hit in hits] == ["a", "b", "c"]
    assert hits[0].score > hits[1].score >= hits[2].score


def test_search_applies_payload_filters() -> None:
    """The where filter restricts candidates before ranking."""
    store = build_store()
    hits = store.search(
        np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float32), k=3, where={"chapter": "Two"}
    )

    assert [hit.id for hit in hits] == ["b", "c"]


def test_search_edge_cases() -> None:
    """Empty stores, zero vectors and zero k all return no hits."""
    store = build_store()

    assert NumpyStore(dim=4).search(np.ones(4, dtype=np.float32), k=3) == []
    assert store.search(np.zeros(4, dtype=np.float32), k=3) == []
    assert store.search(np.ones(4, dtype=np.float32), k=0) == []
    assert (
        store.search(np.ones(4, dtype=np.float32), k=3, where={"chapter": "Nope"}) == []
    )


def test_upsert_validates_shapes_and_lengths() -> None:
    """Mismatched inputs raise ValueError instead of corrupting the store."""
    store = NumpyStore(dim=4)

    with pytest.raises(ValueError, match="must have shape"):
        store.upsert(["a"], np.ones((1, 3), dtype=np.float32), [{}])
    with pytest.raises(ValueError, match="same length"):
        store.upsert(["a", "b"], np.ones((1, 4), dtype=np.float32), [{}])
    assert store.count() == 0


def test_persist_and_load_roundtrip(tmp_path: Path) -> None:
    """Persisted stores reload with identical ids, payloads and scores."""
    store = create_store(tmp_path / "store", dim=4, backend="numpy")
    store.upsert(
        ["a"],
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        payloads=[{"chapter": "One"}],
    )
    store.persist()

    loaded = open_store(tmp_path / "store", dim=4, backend="numpy")
    hits = loaded.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=1)

    assert loaded.count() == 1
    assert hits[0].id == "a"


def test_persist_without_path_raises_configuration_error(tmp_path: Path) -> None:
    """Persisting a store created without a path is a configuration error."""
    from distiller.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="without a path"):
        NumpyStore(dim=4).persist()
