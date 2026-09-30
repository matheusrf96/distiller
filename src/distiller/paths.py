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

    def ensure(self) -> BookPaths:
        """Create the artifact directory if it does not exist yet.

        Returns:
            The same layout, for chaining.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        return self
