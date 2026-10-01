"""Vector stores: NumPy (exact, zero-infra, default) and Qdrant (local mode)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from ..exceptions import ConfigurationError
from ..optional_deps import require
from ..utils import read_json, write_json
from .hits import SearchHit

if TYPE_CHECKING:
    from collections.abc import Sequence

Payload = dict[str, Any]


@runtime_checkable
class VectorStore(Protocol):
    """Minimal interface every vector store implementation provides."""

    @property
    def dim(self) -> int:
        """Embedding dimension the store was created with."""
        ...  # pragma: no cover - protocol stub

    def upsert(
        self,
        ids: Sequence[str],
        vectors: np.ndarray,
        payloads: Sequence[Payload],
    ) -> None:
        """Add vectors and their payloads to the store."""
        ...  # pragma: no cover - protocol stub

    def search(
        self, vector: np.ndarray, k: int, where: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Return the top ``k`` chunks by similarity, with optional payload filters."""
        ...  # pragma: no cover - protocol stub

    def count(self) -> int:
        """Number of stored vectors."""
        ...  # pragma: no cover - protocol stub

    def persist(self) -> None:
        """Flush any in-memory state to the store's storage location."""
        ...  # pragma: no cover - protocol stub


class NumpyStore:
    """Exact cosine search over an in-memory matrix (perfect for book scale).

    Persists as ``vectors.npy`` plus ``records.json`` under the given path.

    Attributes:
        dim: Embedding dimension.
    """

    def __init__(self, dim: int, path: Path | None = None) -> None:
        """Create an empty store.

        Args:
            dim: Embedding dimension.
            path: Directory used by :meth:`persist` and :meth:`load`.
        """
        self._dim = int(dim)
        self._path = Path(path) if path is not None else None
        self._ids: list[str] = []
        self._payloads: list[Payload] = []
        self._matrix = np.zeros((0, self._dim), dtype=np.float32)

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    def count(self) -> int:
        """Number of stored vectors."""
        return len(self._ids)

    def upsert(
        self, ids: Sequence[str], vectors: np.ndarray, payloads: Sequence[Payload]
    ) -> None:
        """Add vectors and payloads, validating shapes and lengths.

        Raises:
            ValueError: When shapes or lengths are inconsistent.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(f"vectors must have shape (n, {self._dim})")
        if not (len(ids) == len(payloads) == vectors.shape[0]):
            raise ValueError("ids, vectors and payloads must have the same length")
        if vectors.shape[0] == 0:
            return
        self._ids.extend(str(identifier) for identifier in ids)
        self._payloads.extend(dict(payload) for payload in payloads)
        self._matrix = (
            vectors if self._matrix.size == 0 else np.vstack([self._matrix, vectors])
        )

    def search(
        self, vector: np.ndarray, k: int, where: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Return the top ``k`` chunks by cosine similarity.

        Args:
            vector: Query embedding.
            k: Maximum number of hits.
            where: Optional payload equality filters.

        Returns:
            Hits, best first.
        """
        if not self._ids or k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)[: self._dim]
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        scores = self._matrix @ (query / norm)

        valid = np.arange(len(self._ids))
        if where:
            valid = np.array(
                [
                    index
                    for index in valid
                    if all(
                        self._payloads[index].get(key) == value
                        for key, value in where.items()
                    )
                ],
                dtype=int,
            )
            if valid.size == 0:
                return []

        similarities = scores[valid]
        order = np.argsort(-similarities)[: min(k, len(similarities))]
        return [
            SearchHit(self._ids[int(valid[i])], float(similarities[i])) for i in order
        ]

    def persist(self) -> None:
        """Write the matrix and records to disk.

        Raises:
            ConfigurationError: When the store was created without a path.
        """
        if self._path is None:
            raise ConfigurationError(
                "NumpyStore was created without a path; nothing to persist to"
            )
        self._path.mkdir(parents=True, exist_ok=True)
        np.save(self._path / "vectors.npy", self._matrix)
        write_json(
            self._path / "records.json",
            [
                {"id": identifier, "payload": payload}
                for identifier, payload in zip(self._ids, self._payloads, strict=True)
            ],
        )

    @classmethod
    def load(cls, path: Path) -> NumpyStore:
        """Load a persisted store.

        Args:
            path: Directory previously written by :meth:`persist`.

        Returns:
            Store with the persisted vectors and records.
        """
        path = Path(path)
        matrix = np.load(path / "vectors.npy").astype(np.float32)
        records = read_json(path / "records.json")
        store = cls(dim=matrix.shape[1], path=path)
        store._ids = [str(record["id"]) for record in records]
        store._payloads = [dict(record["payload"]) for record in records]
        store._matrix = matrix
        return store


class QdrantStore:
    """Qdrant in local (on-disk, no server) mode.

    Switch with ``DISTILLER_STORE__BACKEND=qdrant``; requires the ``index`` extra.
    """

    COLLECTION = "distiller_chunks"
    _PURPOSE = "The Qdrant vector store"

    def __init__(
        self, dim: int, *, path: Path | None = None, url: str | None = None
    ) -> None:
        """Open or create the collection.

        Args:
            dim: Embedding dimension.
            path: Local on-disk directory (no server required).
            url: Server URL, used when ``path`` is None.

        Raises:
            MissingDependencyError: When the ``index`` extra is not installed.
        """
        qdrant = require("qdrant_client", extra="index", purpose=self._PURPOSE)
        models = require("qdrant_client.models", extra="index", purpose=self._PURPOSE)
        self._models = models
        self._dim = int(dim)

        if path is not None:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = qdrant.QdrantClient(path=str(path))
        else:
            self._client = qdrant.QdrantClient(url=url or "http://localhost:6333")

        if not self._client.collection_exists(self.COLLECTION):
            self._client.create_collection(
                self.COLLECTION,
                vectors_config=models.VectorParams(
                    size=self._dim, distance=models.Distance.COSINE
                ),
            )

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim

    def count(self) -> int:
        """Number of stored vectors."""
        return int(self._client.count(self.COLLECTION, exact=True).count)

    def upsert(
        self,
        ids: Sequence[str],
        vectors: np.ndarray,
        payloads: Sequence[Payload],
    ) -> None:
        """Add vectors and payloads as Qdrant points.

        Chunk ids are mapped to deterministic UUID point ids.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        points = []
        for chunk_id, vector, payload in zip(ids, vectors, payloads, strict=True):
            enriched = dict(payload)
            enriched["chunk_id"] = str(chunk_id)
            points.append(
                self._models.PointStruct(
                    id=_point_id(chunk_id), vector=vector.tolist(), payload=enriched
                )
            )
        if points:
            self._client.upsert(self.COLLECTION, points=points)

    def search(
        self, vector: np.ndarray, k: int, where: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Return the top ``k`` chunks by similarity, with optional payload filters."""
        query_filter = None
        if where:
            query_filter = self._models.Filter(
                must=[
                    self._models.FieldCondition(
                        key=key, match=self._models.MatchValue(value=value)
                    )
                    for key, value in where.items()
                ]
            )
        result = self._client.query_points(
            collection_name=self.COLLECTION,
            query=np.asarray(vector, dtype=np.float32).tolist(),
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in result.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    str(payload.get("chunk_id", point.id)), float(point.score or 0.0)
                )
            )
        return hits

    def persist(self) -> None:
        """No-op: Qdrant writes are durable as they happen."""
        return


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def create_store(path: Path, dim: int, backend: str) -> VectorStore:
    """Create an empty store ready for ``upsert``.

    Args:
        path: Storage directory for the store.
        dim: Embedding dimension.
        backend: ``numpy`` or ``qdrant``.

    Returns:
        Empty store bound to ``path``.
    """
    if backend == "qdrant":
        return QdrantStore(dim, path=path)
    return NumpyStore(dim, path=path)


def open_store(path: Path, dim: int, backend: str) -> VectorStore:
    """Open a previously built store.

    Args:
        path: Storage directory written at build time.
        dim: Embedding dimension.
        backend: ``numpy`` or ``qdrant``.

    Returns:
        Store with the persisted vectors.
    """
    if backend == "qdrant":
        return QdrantStore(dim, path=path)
    return NumpyStore.load(path)
