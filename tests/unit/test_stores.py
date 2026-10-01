"""Unit tests for the vector stores."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from distiller.indexing.store import (
    NumpyStore,
    QdrantStore,
    create_store,
    open_store,
)

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


def test_numpy_store_empty_batch_is_a_noop() -> None:
    """An empty upsert leaves the store unchanged."""
    store = NumpyStore(dim=4)

    store.upsert([], np.zeros((0, 4), dtype=np.float32), [])

    assert store.count() == 0
    assert store.dim == 4


class _FakeVectorParams:
    def __init__(self, size: int, distance: str) -> None:
        self.size = size
        self.distance = distance


class _FakePointStruct:
    def __init__(
        self, id: str, vector: list[float], payload: dict[str, object]
    ) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


class _FakeFilter:
    def __init__(self, must: list[object]) -> None:
        self.must = must


class _FakeFieldCondition:
    def __init__(self, key: str, match: object) -> None:
        self.key = key
        self.match = match


class _FakeMatchValue:
    def __init__(self, value: object) -> None:
        self.value = value


class FakeQdrantClient:
    """In-memory stand-in for the qdrant_client client."""

    def __init__(self, path: str | None = None, url: str | None = None) -> None:
        self.path = path
        self.url = url
        self.collections: dict[str, _FakeVectorParams] = {}
        self.points: list[_FakePointStruct] = []
        self.last_filter: _FakeFilter | None = None

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, vectors_config: _FakeVectorParams) -> None:
        self.collections[name] = vectors_config

    def count(self, name: str, exact: bool = False) -> SimpleNamespace:
        return SimpleNamespace(count=len(self.points))

    def upsert(self, name: str, points: list[_FakePointStruct]) -> None:
        self.points.extend(points)

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        limit: int,
        query_filter: _FakeFilter | None,
        with_payload: bool,
    ) -> SimpleNamespace:
        self.last_filter = query_filter
        return SimpleNamespace(
            points=[
                SimpleNamespace(payload=point.payload, id=point.id, score=0.5)
                for point in self.points[:limit]
            ]
        )


class ExistingCollectionClient(FakeQdrantClient):
    """Client whose collection already exists, so creation is skipped."""

    def __init__(self, path: str | None = None, url: str | None = None) -> None:
        super().__init__(path=path, url=url)
        self.collections[QdrantStore.COLLECTION] = _FakeVectorParams(0, "cosine")


def fake_qdrant(
    monkeypatch: pytest.MonkeyPatch,
    client_class: type[FakeQdrantClient] = FakeQdrantClient,
) -> None:
    """Route the store's require() calls to the fake client and models."""
    models = SimpleNamespace(
        Distance=SimpleNamespace(COSINE="cosine"),
        VectorParams=_FakeVectorParams,
        PointStruct=_FakePointStruct,
        Filter=_FakeFilter,
        FieldCondition=_FakeFieldCondition,
        MatchValue=_FakeMatchValue,
    )

    def fake_require(name: str, **kwargs: object) -> object:
        if name == "qdrant_client":
            return SimpleNamespace(QdrantClient=client_class)
        if name == "qdrant_client.models":
            return models
        raise AssertionError(f"unexpected require: {name}")

    monkeypatch.setattr("distiller.indexing.store.require", fake_require)


def test_qdrant_store_upserts_searches_and_filters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Qdrant store creates its collection and maps points to hits."""
    fake_qdrant(monkeypatch)
    store = QdrantStore(dim=4, path=tmp_path / "qdrant")

    assert store.dim == 4
    store.upsert(
        ["a", "b"],
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        payloads=[{"chapter": "One"}, {"chapter": "Two"}],
    )
    store.upsert([], np.zeros((0, 4), dtype=np.float32), [])
    assert store.count() == 2

    hits = store.search(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        k=2,
        where={"chapter": "Two"},
    )

    assert [hit.id for hit in hits] == ["a", "b"]
    assert hits[0].score == 0.5
    client = store._client
    assert isinstance(client, FakeQdrantClient)
    assert client.last_filter is not None
    assert client.last_filter.must[0].key == "chapter"
    assert store.search(np.ones(4, dtype=np.float32), k=1).__len__() == 1
    assert client.last_filter is None
    assert store.persist() is None


def test_qdrant_store_skips_existing_collection_and_supports_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-existing collection is not recreated; URL mode skips the path."""
    fake_qdrant(monkeypatch, client_class=ExistingCollectionClient)
    store = QdrantStore(dim=4, url="http://example:6333")

    client = store._client
    assert isinstance(client, ExistingCollectionClient)
    assert client.url == "http://example:6333"
    assert client.path is None
    assert client.collections[QdrantStore.COLLECTION].size == 0


def test_create_and_open_store_dispatch_to_qdrant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The factory functions build Qdrant stores for the qdrant backend."""
    fake_qdrant(monkeypatch)

    assert isinstance(
        create_store(tmp_path / "store", dim=4, backend="qdrant"), QdrantStore
    )
    assert isinstance(
        open_store(tmp_path / "store", dim=4, backend="qdrant"), QdrantStore
    )
