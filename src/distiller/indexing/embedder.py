"""Embedders: local sentence-transformers models, plus an offline hashing backend."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np

from ..exceptions import ConfigurationError
from ..optional_deps import require
from ..utils import stable_hash_hex

if TYPE_CHECKING:
    from ..config import EmbeddingSettings

QWEN3_QUERY_PROMPT = (
    "Instruct: Given a web search query, "
    "retrieve relevant passages that answer the query\n"
    "Query: "
)
_TOKEN = re.compile(r"\w+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    @property
    def name(self) -> str: ...  # pragma: no cover - protocol stub

    @property
    def dim(self) -> int: ...  # pragma: no cover - protocol stub

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...  # pragma: no cover
    def embed_query(self, text: str) -> np.ndarray: ...  # pragma: no cover


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. No models, no downloads.

    Used by tests/CI and for fully offline smoke runs. Quality is intentionally
    low; it only needs to make lexical overlaps discoverable.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = max(64, int(dim))

    @property
    def name(self) -> str:
        return f"hash:{self.dim}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            digest = int(stable_hash_hex(token, 16), 16)
            vec[digest % self.dim] += 1.0 if (digest >> 63) & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class SentenceTransformerEmbedder:
    """Local embedding model via sentence-transformers (e.g. Qwen3-Embedding-0.6B)."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        dim: int | None = None,
        batch_size: int = 16,
        query_prompt: str | None = None,
    ) -> None:
        module = require(
            "sentence_transformers", extra="embed", purpose="Local embedding models"
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = module.SentenceTransformer(model_name, device=device)
        native_dim = int(self._model.get_sentence_embedding_dimension())
        if dim is not None and dim > native_dim:
            raise ConfigurationError(
                f"Requested dim={dim} exceeds native dim={native_dim} for {model_name}"
            )
        self.dim = int(dim or native_dim)
        self.query_prompt = (
            query_prompt
            if query_prompt is not None
            else _default_query_prompt(model_name)
        )

    @property
    def name(self) -> str:
        return f"st:{self.model_name}:{self.dim}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        encoded: Any = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
        )
        matrix = cast("np.ndarray", np.asarray(encoded, dtype=np.float32))
        return self._truncate(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        query = f"{self.query_prompt}{text}" if self.query_prompt else text
        encoded: Any = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        matrix = cast("np.ndarray", np.asarray(encoded, dtype=np.float32))
        query_vector: np.ndarray = self._truncate(matrix)[0]
        return query_vector

    def _truncate(self, vectors: np.ndarray) -> np.ndarray:
        if self.dim >= vectors.shape[1]:
            return vectors
        truncated = vectors[:, : self.dim]
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit_vectors: np.ndarray = truncated / norms
        return unit_vectors


def _default_query_prompt(model_name: str) -> str | None:
    """Return the built-in query instruction for known model families."""
    if "Qwen3-Embedding" in model_name:
        return QWEN3_QUERY_PROMPT
    return None


def get_embedder(settings: EmbeddingSettings) -> Embedder:
    """Build the embedder selected by the configuration.

    Args:
        settings: Embedding settings (backend, model, device, dimensions).

    Returns:
        A hashing embedder for offline runs, otherwise a local model embedder.
    """
    if settings.backend == "hash":
        return HashingEmbedder(dim=settings.hash_dim)
    return SentenceTransformerEmbedder(
        settings.model,
        device=settings.device,
        dim=settings.dim,
        batch_size=settings.batch_size,
    )
