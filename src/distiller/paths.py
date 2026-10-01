"""Artifact layout: everything for one book lives under ``artifacts/<book_id>/``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BookPaths:
    """Immutable filesystem layout for one book's artifacts (a helper, not data).

    Attributes:
        root: Artifact directory for the book (``artifacts/<book_id>``).
    """

    root: Path

    @classmethod
    def for_book(cls, artifacts_dir: Path | str, book_id: str) -> BookPaths:
        """Build the layout for one book.

        Args:
            artifacts_dir: Root artifacts directory.
            book_id: Book slug printed by ``distiller ingest``.

        Returns:
            Layout rooted at ``<artifacts_dir>/<book_id>``.
        """
        return cls(root=Path(artifacts_dir) / book_id)

    @property
    def book_json(self) -> Path:
        """Structured parse of the book (``book.json``)."""
        return self.root / "book.json"

    @property
    def parsed_md(self) -> Path:
        """Human-readable Markdown rendering of the parse (``parsed.md``)."""
        return self.root / "parsed.md"

    @property
    def chunks_jsonl(self) -> Path:
        """Retrieval chunks with provenance (``chunks.jsonl``)."""
        return self.root / "chunks.jsonl"

    @property
    def index_dir(self) -> Path:
        """Index directory (``index/``)."""
        return self.root / "index"

    @property
    def index_metadata(self) -> Path:
        """Index metadata: embedder identity, dimensions, counts."""
        return self.index_dir / "metadata.json"

    @property
    def store_dir(self) -> Path:
        """Vector store data directory (``index/store/``)."""
        return self.index_dir / "store"

    @property
    def eval_dir(self) -> Path:
        """Evaluation reports directory (``eval/``)."""
        return self.root / "eval"

    @property
    def golden_yaml(self) -> Path:
        """Default golden question set (``golden.yaml``)."""
        return self.root / "golden.yaml"

    @property
    def enrichment_jsonl(self) -> Path:
        """LLM-generated chunk contexts cache (``enrichment.jsonl``)."""
        return self.root / "enrichment.jsonl"

    @property
    def dataset_dir(self) -> Path:
        """Synthetic QA and RAFT dataset directory (``dataset/``)."""
        return self.root / "dataset"

    @property
    def dataset_qa_jsonl(self) -> Path:
        """Generated QA pairs, pre-filter (``dataset/qa.jsonl``)."""
        return self.dataset_dir / "qa.jsonl"

    @property
    def dataset_rejected_jsonl(self) -> Path:
        """Pairs rejected by the filters, with reasons (``dataset/rejected.jsonl``)."""
        return self.dataset_dir / "rejected.jsonl"

    @property
    def dataset_raft_jsonl(self) -> Path:
        """RAFT training examples (``dataset/raft.jsonl``)."""
        return self.dataset_dir / "raft.jsonl"

    @property
    def dataset_manifest(self) -> Path:
        """Dataset provenance and counts (``dataset/manifest.json``)."""
        return self.dataset_dir / "manifest.json"

    @property
    def training_dir(self) -> Path:
        """Training artifacts directory (``training/``)."""
        return self.root / "training"

    @property
    def training_train_jsonl(self) -> Path:
        """Chat-formatted training split (``training/train.jsonl``)."""
        return self.training_dir / "train.jsonl"

    @property
    def training_validation_jsonl(self) -> Path:
        """Chat-formatted held-out split (``training/validation.jsonl``)."""
        return self.training_dir / "validation.jsonl"

    @property
    def training_manifest(self) -> Path:
        """Training dataset provenance and counts (``training/manifest.json``)."""
        return self.training_dir / "manifest.json"

    @property
    def training_qlora_json(self) -> Path:
        """Pinned QLoRA configuration (``training/qlora.json``)."""
        return self.training_dir / "qlora.json"

    @property
    def training_notebook(self) -> Path:
        """Self-contained T4 training notebook (``training/train_t4.ipynb``)."""
        return self.training_dir / "train_t4.ipynb"

    @property
    def adapter_dir(self) -> Path:
        """Registered LoRA adapter directory (``training/adapter/``)."""
        return self.training_dir / "adapter"

    @property
    def adapter_run_json(self) -> Path:
        """Training report of the registered adapter (``training/adapter/run.json``)."""
        return self.adapter_dir / "run.json"

    @property
    def gguf_dir(self) -> Path:
        """Registered GGUF directory (``training/gguf/``)."""
        return self.training_dir / "gguf"

    @property
    def gguf_file(self) -> Path:
        """Registered GGUF copy (``training/gguf/model.gguf``)."""
        return self.gguf_dir / "model.gguf"

    @property
    def gguf_report_json(self) -> Path:
        """GGUF registration report (``training/gguf/gguf.json``)."""
        return self.gguf_dir / "gguf.json"

    @property
    def gguf_modelfile(self) -> Path:
        """Ollama Modelfile (``training/gguf/Modelfile``)."""
        return self.gguf_dir / "Modelfile"

    @property
    def gguf_serve_script(self) -> Path:
        """llama-server/Ollama serving script (``training/gguf/serve.sh``)."""
        return self.gguf_dir / "serve.sh"

    @property
    def eval_training_json(self) -> Path:
        """Base-vs-adapter comparison report (``eval/training.json``)."""
        return self.eval_dir / "training.json"

    def ensure(self) -> BookPaths:
        """Create the artifact directory if it does not exist yet.

        Returns:
            The same layout, for chaining.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        return self
