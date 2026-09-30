"""Configuration for the distiller pipeline.

Settings load from environment variables (``DISTILLER_*``, ``__`` nests keys)
and from an optional ``distiller.toml`` in the working directory.

Example::

    DISTILLER_LLM__MODEL=gpt-4o-mini
    DISTILLER_LLM__BASE_URL=http://localhost:11434/v1
    DISTILLER_EMBEDDING__BACKEND=hash          # offline tests
    DISTILLER_ARTIFACTS_DIR=artifacts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class LLMSettings(BaseModel):
    """Generation backend settings (any OpenAI-compatible endpoint).

    Attributes:
        base_url: Endpoint base URL; defaults to a local server.
        api_key: API key, when the endpoint requires one.
        model: Model name; the sentinel ``"fake"`` selects the offline client.
        temperature: Sampling temperature for answers.
        max_tokens: Maximum tokens per generated answer.
        timeout: Request timeout in seconds.
    """

    base_url: str | None = None
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: float = 120.0


class EmbeddingSettings(BaseModel):
    """Embedding backend settings.

    Attributes:
        backend: ``sentence-transformers`` for real models, ``hash`` for offline tests.
        model: Hugging Face model name for the sentence-transformers backend.
        device: Torch device override (e.g. ``"cpu"``, ``"cuda"``).
        batch_size: Encoding batch size.
        dim: Optional MRL truncation of the embedding dimension.
        hash_dim: Dimension used only by the offline ``hash`` backend.
    """

    backend: Literal["sentence-transformers", "hash"] = "sentence-transformers"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str | None = None
    batch_size: int = 16
    dim: int | None = None
    hash_dim: int = 512


class IngestSettings(BaseModel):
    """Book parsing settings.

    Attributes:
        pdf_backend: Which PDF extractor to use; ``auto`` prefers the best available.
    """

    pdf_backend: Literal["auto", "docling", "pymupdf4llm", "pymupdf"] = "auto"


class ChunkingSettings(BaseModel):
    """Chunk sizing settings.

    Attributes:
        target_chars: Preferred chunk size; chunks flush once they reach it.
        overlap_chars: Whole-paragraph overlap between neighbouring chunks.
        max_chars: Hard ceiling for a chunk; oversized blocks are split.
    """

    target_chars: int = 1800
    overlap_chars: int = 250
    max_chars: int = 3600

    @model_validator(mode="after")
    def _validate_bounds(self) -> ChunkingSettings:
        if self.target_chars <= 0:
            raise ValueError("chunking.target_chars must be > 0")
        if self.max_chars < self.target_chars:
            raise ValueError("chunking.max_chars must be >= chunking.target_chars")
        if not 0 <= self.overlap_chars <= self.target_chars // 2:
            raise ValueError(
                "chunking.overlap_chars must be between 0 and "
                "half of chunking.target_chars"
            )
        return self


class RetrievalSettings(BaseModel):
    """Hybrid retrieval and reranking settings.

    Attributes:
        top_k_dense: Number of dense candidates fetched before fusion.
        top_k_sparse: Number of BM25 candidates fetched before fusion.
        top_k_final: Number of chunks handed to the generator (and reranker output).
        rrf_k: Reciprocal-rank-fusion smoothing constant.
        rerank: Whether cross-encoder reranking is enabled by default.
        rerank_model: Cross-encoder model name.
        rerank_pool: Candidate pool size fed to the reranker.
    """

    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 8
    rrf_k: int = 60
    rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_pool: int = 40


class StoreSettings(BaseModel):
    """Vector store settings.

    Attributes:
        backend: ``numpy`` (exact, zero-infra) or ``qdrant`` (local mode).
    """

    backend: Literal["numpy", "qdrant"] = "numpy"


class SynthesisSettings(BaseModel):
    """Synthetic QA and RAFT dataset settings.

    Attributes:
        max_chunks: Maximum chunks sampled for generation (cost control).
        questions_per_chunk: Questions requested per chunk in one LLM call.
        distractors: Distractor chunks included per RAFT example.
        negative_ratio: Fraction of examples built as unanswerable negatives.
        seed: Seed for sampling and example shuffling (reproducibility).
    """

    max_chunks: int = 100
    questions_per_chunk: int = 2
    distractors: int = 4
    negative_ratio: float = 0.15
    seed: int = 13

    @model_validator(mode="after")
    def _validate_bounds(self) -> SynthesisSettings:
        if self.max_chunks < 1:
            raise ValueError("synthesis.max_chunks must be >= 1")
        if self.questions_per_chunk < 1:
            raise ValueError("synthesis.questions_per_chunk must be >= 1")
        if self.distractors < 0:
            raise ValueError("synthesis.distractors must be >= 0")
        if not 0.0 <= self.negative_ratio <= 1.0:
            raise ValueError("synthesis.negative_ratio must be between 0 and 1")
        return self


class EnrichmentSettings(BaseModel):
    """Contextual retrieval settings.

    Enrichment asks the configured LLM for a short situating context per chunk;
    the context is used only for embedding and lexical indexing.

    Attributes:
        enabled: Generate per-chunk contexts before indexing (off by default).
        max_document_chars: Chapter excerpt budget fed to the LLM for situating.
        max_context_chars: Hard cap for the generated context prefix.
    """

    enabled: bool = False
    max_document_chars: int = 6000
    max_context_chars: int = 500


class EvaluationSettings(BaseModel):
    """Evaluation settings.

    Attributes:
        judge_model: LLM used for RAGAS judging; defaults to the generation model.
        max_samples: Maximum samples sent to RAGAS per run.
    """

    judge_model: str | None = None
    max_samples: int = 50


class Settings(BaseSettings):
    """Root configuration: environment variables plus optional TOML file.

    Attributes:
        artifacts_dir: Root directory for per-book artifacts.
        llm: Generation backend settings.
        embedding: Embedding backend settings.
        ingest: Parsing settings.
        chunking: Chunk sizing settings.
        retrieval: Retrieval and reranking settings.
        store: Vector store settings.
        evaluation: Evaluation settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="DISTILLER_",
        env_nested_delimiter="__",
        extra="ignore",
        toml_file="distiller.toml",
    )

    artifacts_dir: Path = Path("artifacts")
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    enrichment: EnrichmentSettings = Field(default_factory=EnrichmentSettings)
    synthesis: SynthesisSettings = Field(default_factory=SynthesisSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Apply source precedence: init args > environment > distiller.toml."""
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def load_settings(**overrides: Any) -> Settings:
    """Build settings fresh (no caching, so tests can monkeypatch environment).

    Args:
        **overrides: Highest-precedence values for specific fields.

    Returns:
        Fully validated settings instance.
    """
    return Settings(**overrides)
