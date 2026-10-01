"""Adapter registry: validate and install a T4 adapter directory (Phase 3).

The registry is filesystem-driven (mirroring ``index/metadata.json``):
``training/adapter/run.json`` carries the adapter identity, so re-training a
book needs no configuration churn. Phase 4's serving registry builds on the
same ``run.json``.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import Field

from ..exceptions import TrainingError
from ..models import DomainModel
from ..utils import read_json, write_json

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")

__all__ = [
    "ADAPTER_FILES",
    "TrainingReport",
    "load_adapter_report",
    "register_adapter",
]


class TrainingReport(DomainModel):
    """Identity, hashes and loss history of one adapter training run.

    Written as ``run.json`` by the T4 notebook and copied into
    ``training/adapter/`` by ``distiller train --register``.

    Attributes:
        book_id: Book the adapter was trained for.
        base_model: Base checkpoint the adapter was trained on.
        dataset_hash: Hash of the training dataset the adapter saw.
        config_hash: Hash of ``qlora.json`` at training time.
        created_at: UTC timestamp of the training run.
        train_count: Training examples used.
        validation_count: Held-out examples used.
        epochs: Training epochs completed.
        learning_rate: Optimizer learning rate.
        seed: Training seed.
        loss_history: Per-step training loss values.
        hardware: GPU used for training, when reported.
        trainer: Trainer stack that produced the adapter.
        dataset_hash_matches: False when registration found the adapter was
            trained on a different dataset than the current one (recorded, not
            fatal).
    """

    book_id: str
    base_model: str
    dataset_hash: str
    config_hash: str
    created_at: datetime
    train_count: int = 0
    validation_count: int = 0
    epochs: int = 0
    learning_rate: float = 0.0
    seed: int = 13
    loss_history: list[float] = Field(default_factory=list)
    hardware: str | None = None
    trainer: str = "unsloth+SFTTrainer"
    dataset_hash_matches: bool = True


def load_adapter_report(adapter_dir: Path) -> TrainingReport:
    """Load and validate the ``run.json`` of an adapter directory.

    Args:
        adapter_dir: Directory that should contain ``run.json``.

    Returns:
        Validated training report.

    Raises:
        TrainingError: When ``run.json`` is missing or invalid.
    """
    run_json = adapter_dir / "run.json"
    if not run_json.exists():
        raise TrainingError(
            f"Not an adapter directory: {run_json} is missing. "
            f"Point --register at the folder downloaded from the T4 run."
        )
    try:
        return TrainingReport.model_validate(read_json(run_json))
    except (OSError, ValueError) as exc:
        raise TrainingError(f"Invalid adapter run.json at {run_json}: {exc}") from exc


def register_adapter(
    source_dir: Path,
    destination: Path,
    *,
    book_id: str,
    expected_base_model: str,
    expected_dataset_hash: str | None = None,
) -> TrainingReport:
    """Validate a T4 adapter directory and copy it into the book's artifacts.

    Args:
        source_dir: Adapter directory downloaded from the T4 run.
        destination: Registry directory (``training/adapter``).
        book_id: Book the adapter must belong to.
        expected_base_model: Base model pinned in this book's ``qlora.json``.
        expected_dataset_hash: Current dataset hash; a mismatch is recorded in
            the report, not fatal.

    Returns:
        The registered report (with ``dataset_hash_matches`` updated).

    Raises:
        TrainingError: When the directory is invalid, the book id or base model
            differs, or the adapter weight files are missing.
    """
    report = load_adapter_report(source_dir)
    if report.book_id != book_id:
        raise TrainingError(
            f"Adapter was trained for book '{report.book_id}', not '{book_id}'. "
            f"Re-run `distiller train {book_id}` and register the matching adapter."
        )
    if report.base_model != expected_base_model:
        raise TrainingError(
            f"Adapter base model '{report.base_model}' does not match this book's "
            f"configured base model '{expected_base_model}' (training/qlora.json)."
        )
    for name in ADAPTER_FILES:
        if not (source_dir / name).exists():
            raise TrainingError(
                f"Adapter directory {source_dir} is missing '{name}'. "
                f"Download the full adapter folder from the T4 run."
            )

    if (
        expected_dataset_hash is not None
        and report.dataset_hash != expected_dataset_hash
    ):
        logger.warning(
            "Adapter %s was trained on dataset %s, current dataset is %s",
            source_dir,
            report.dataset_hash,
            expected_dataset_hash,
        )
        report = report.model_copy(update={"dataset_hash_matches": False})

    if source_dir.resolve() != destination.resolve():
        shutil.copytree(source_dir, destination, dirs_exist_ok=True)
    write_json(destination / "run.json", json.loads(report.model_dump_json()))
    return report
