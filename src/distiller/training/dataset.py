"""Training dataset preparation: split, validation, manifest and writing.

Preparation is deliberately validate-then-write: nothing touches the artifacts
when an example is malformed or the dataset has no refusal negatives.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import Field

from ..exceptions import TrainingError
from ..models import DomainModel, refusal_text
from ..synthesis import RaftExample
from ..utils import read_jsonl, stable_hash_hex, write_json, write_jsonl
from .chat import citation_indices, format_examples, prompt_contract_hash
from .notebook import emit_notebook
from .qlora import QLoRAConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ..config import TrainingSettings
    from ..models import BookDocument, Chunk
    from ..paths import BookPaths
    from .chat import TrainingExample

_DOC_MARKER = "<doc "

__all__ = [
    "TrainingBundle",
    "TrainingDatasetManifest",
    "ValidationReport",
    "load_raft_examples",
    "prepare_dataset",
    "split_examples",
    "validate_examples",
]


class TrainingDatasetManifest(DomainModel):
    """Provenance and counts for one chat-formatted training dataset.

    Attributes:
        book_id: Book the dataset was prepared for.
        created_at: UTC timestamp of the preparation run.
        seed: Seed used for the train/validation split.
        val_ratio: Requested validation ratio.
        source_count: RAFT examples read from ``dataset/raft.jsonl``.
        train_count: Examples written to ``training/train.jsonl``.
        validation_count: Examples written to ``training/validation.jsonl``.
        answerable_count: Positive examples in the dataset.
        unanswerable_count: Refusal negatives in the dataset.
        source_hash: Hash of the source RAFT examples.
        dataset_hash: Hash of the written chat examples (train then validation).
        prompt_hash: Hash of the shared RAG prompt contract.
    """

    book_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    seed: int = 13
    val_ratio: float = 0.1
    source_count: int = 0
    train_count: int = 0
    validation_count: int = 0
    answerable_count: int = 0
    unanswerable_count: int = 0
    source_hash: str = ""
    dataset_hash: str = ""
    prompt_hash: str = ""


class ValidationReport(DomainModel):
    """Result of the pre-training dataset checks.

    Attributes:
        example_count: Examples checked.
        answerable_count: Positive examples.
        unanswerable_count: Refusal negatives.
        errors: Fatal problems; any error aborts preparation.
        warnings: Non-fatal observations.
    """

    example_count: int = 0
    answerable_count: int = 0
    unanswerable_count: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when preparation may proceed (no errors)."""
        return not self.errors


@dataclass(frozen=True, slots=True)
class TrainingBundle:
    """Everything one preparation run produced (runtime container).

    Attributes:
        manifest: Dataset provenance and counts.
        examples: All formatted chat examples.
        train: Training split written to ``train.jsonl``.
        validation: Held-out split written to ``validation.jsonl``.
        config: Pinned QLoRA configuration for the run.
        validation_report: Outcome of the pre-write checks.
    """

    manifest: TrainingDatasetManifest
    examples: list[TrainingExample]
    train: list[TrainingExample]
    validation: list[TrainingExample]
    config: QLoRAConfig
    validation_report: ValidationReport


def load_raft_examples(path: Path) -> list[RaftExample]:
    """Load RAFT examples from ``dataset/raft.jsonl``.

    Args:
        path: RAFT dataset written by ``distiller synth``.

    Returns:
        Validated examples in file order.

    Raises:
        TrainingError: When the file is missing or unreadable.
    """
    if not path.exists():
        raise TrainingError(f"No RAFT dataset at {path}. Run `distiller synth` first.")
    try:
        return [RaftExample.model_validate(row) for row in read_jsonl(path)]
    except (OSError, ValueError) as exc:
        raise TrainingError(
            f"RAFT dataset {path} is unreadable or corrupt: {exc}. "
            f"Re-run `distiller synth` to rebuild it."
        ) from exc


def split_examples(
    examples: list[TrainingExample],
    *,
    seed: int,
    val_ratio: float,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split examples deterministically into train and validation sets.

    The validation size is ``max(1, floor(n * val_ratio))`` (capped at ``n``) and
    the train split gets the rest. RAFT datasets have unique questions, so the
    disjoint example sets are disjoint question sets.

    Args:
        examples: Chat-formatted examples.
        seed: Random seed for the shuffle.
        val_ratio: Fraction of examples held out for validation.

    Returns:
        ``(train, validation)`` in deterministic order.
    """
    if not examples:
        return [], []
    validation_size = min(max(1, math.floor(len(examples) * val_ratio)), len(examples))
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)  # noqa: S311 - reproducibility, not crypto
    validation_indices = set(order[:validation_size])
    train = [
        example
        for index, example in enumerate(examples)
        if index not in validation_indices
    ]
    validation = [
        example for index, example in enumerate(examples) if index in validation_indices
    ]
    return train, validation


def validate_examples(
    examples: list[TrainingExample],
    book_title: str,
) -> ValidationReport:
    """Check every example before anything is written.

    Args:
        examples: Chat-formatted examples.
        book_title: Book title for the shared refusal sentence.

    Returns:
        Report with errors (fatal), warnings and counts.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not examples:
        errors.append(
            "the dataset contains no examples; run `distiller synth` to rebuild it"
        )
    refusal = refusal_text(book_title)
    for example in examples:
        errors.extend(_example_errors(example, refusal))
    if examples and not any(not example.answerable for example in examples):
        errors.append(
            "the dataset has no unanswerable examples; the refusal negatives are "
            "required for the model to learn when the book does not answer"
        )
    return ValidationReport(
        example_count=len(examples),
        answerable_count=sum(1 for example in examples if example.answerable),
        unanswerable_count=sum(1 for example in examples if not example.answerable),
        errors=errors,
        warnings=warnings,
    )


def prepare_dataset(
    book: BookDocument,
    examples: list[RaftExample],
    chunks: list[Chunk],
    settings: TrainingSettings,
    *,
    paths: BookPaths | None = None,
    qlora: QLoRAConfig | None = None,
) -> TrainingBundle:
    """Format, validate, split and (optionally) write one training dataset.

    Validation happens before any file is written: a malformed example, an
    unknown context chunk or a dataset without refusal negatives raises without
    touching the artifacts.

    Args:
        book: Parsed book.
        examples: RAFT examples from ``dataset/raft.jsonl``.
        chunks: Indexed chunks (provenance for the ``<doc>`` blocks).
        settings: Training settings (seed, validation ratio).
        paths: When given, write the artifacts into this book layout.
        qlora: Optional configuration override; defaults to the pinned config
            with the run seed.

    Returns:
        Everything the run produced: manifest, examples, splits and config.

    Raises:
        TrainingError: On unknown context chunks or validation errors.
    """
    formatted = format_examples(book.title, examples, chunks)
    report = validate_examples(formatted, book.title)
    if not report.ok:
        raise TrainingError(
            "Training data validation failed:\n"
            + "\n".join(f"- {error}" for error in report.errors)
        )

    train, validation = split_examples(
        formatted, seed=settings.seed, val_ratio=settings.val_ratio
    )
    config = qlora or QLoRAConfig(seed=settings.seed)
    manifest = TrainingDatasetManifest(
        book_id=book.book_id,
        seed=settings.seed,
        val_ratio=settings.val_ratio,
        source_count=len(examples),
        train_count=len(train),
        validation_count=len(validation),
        answerable_count=report.answerable_count,
        unanswerable_count=report.unanswerable_count,
        source_hash=_hash_rows(examples),
        dataset_hash=_hash_rows([*train, *validation]),
        prompt_hash=prompt_contract_hash(book.title),
    )
    bundle = TrainingBundle(
        manifest=manifest,
        examples=formatted,
        train=train,
        validation=validation,
        config=config,
        validation_report=report,
    )
    if paths is not None:
        _write_bundle(bundle, paths)
    return bundle


def _example_errors(example: TrainingExample, refusal: str) -> list[str]:
    errors: list[str] = []
    label = example.id
    if not example.messages:
        return [f"example {label}: no messages"]
    for message in example.messages:
        if not message.content.strip():
            errors.append(f"example {label}: empty {message.role} message")
    user = next(
        (message for message in example.messages if message.role == "user"), None
    )
    context_count = user.content.count(_DOC_MARKER) if user else 0
    if context_count == 0:
        errors.append(f"example {label}: no context documents")
    if example.answerable:
        if not any(
            1 <= index <= context_count for index in citation_indices(example.target)
        ):
            errors.append(
                f"example {label}: answerable answer has no valid [n] citation"
            )
    elif example.target != refusal:
        errors.append(
            f"example {label}: negative answer must be exactly the shared refusal "
            f"sentence ({refusal!r})"
        )
    return errors


def _hash_rows(rows: Sequence[DomainModel]) -> str:
    payload = "\n".join(row.model_dump_json() for row in rows)
    return stable_hash_hex(payload, 40)


def _write_bundle(bundle: TrainingBundle, paths: BookPaths) -> None:
    paths.training_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.training_train_jsonl, bundle.train)
    write_jsonl(paths.training_validation_jsonl, bundle.validation)
    _write_model(paths.training_manifest, bundle.manifest)
    _write_model(paths.training_qlora_json, bundle.config)
    write_json(
        paths.training_notebook,
        emit_notebook(bundle.config, book_id=bundle.manifest.book_id),
    )


def _write_model(path: Path, model: DomainModel) -> None:
    """Write a model with pydantic's JSON types (so ``--json`` matches the file)."""
    write_json(path, json.loads(model.model_dump_json()))
