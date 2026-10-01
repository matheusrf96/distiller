"""Unit tests for the adapter registry."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from distiller.exceptions import TrainingError
from distiller.training.registry import load_adapter_report, register_adapter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    AdapterFactory = Callable[..., Path]


def test_register_adapter_copies_and_validates(
    tmp_path: Path, adapter_factory: AdapterFactory
) -> None:
    """A valid adapter is copied and its identity recorded (REQ-TR-008)."""
    source = adapter_factory(tmp_path / "downloaded")
    destination = tmp_path / "artifacts" / "book" / "training" / "adapter"

    report = register_adapter(
        source,
        destination,
        book_id="the-lantern-keeper",
        expected_base_model="Qwen/Qwen3-4B",
        expected_dataset_hash="dataset-hash",
    )

    assert (destination / "run.json").exists()
    assert (destination / "adapter_config.json").exists()
    assert (destination / "adapter_model.safetensors").exists()
    assert report.book_id == "the-lantern-keeper"
    assert report.base_model == "Qwen/Qwen3-4B"
    assert report.dataset_hash_matches is True
    assert load_adapter_report(destination) == report

    written = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert written["loss_history"] == [1.2, 0.8, 0.5]
    assert written["hardware"] == "Tesla T4"


def test_register_adapter_skips_copying_onto_itself(
    tmp_path: Path, adapter_factory: AdapterFactory
) -> None:
    """Registering an adapter directory in place skips the copy step."""
    source = adapter_factory(tmp_path / "adapter")

    report = register_adapter(
        source,
        source,
        book_id="the-lantern-keeper",
        expected_base_model="Qwen/Qwen3-4B",
    )

    assert report.book_id == "the-lantern-keeper"


def test_register_records_a_dataset_hash_mismatch(
    tmp_path: Path, adapter_factory: AdapterFactory
) -> None:
    """A dataset-hash mismatch is recorded, not fatal (open-question decision)."""
    source = adapter_factory(tmp_path / "downloaded", dataset_hash="stale")
    destination = tmp_path / "adapter"

    report = register_adapter(
        source,
        destination,
        book_id="the-lantern-keeper",
        expected_base_model="Qwen/Qwen3-4B",
        expected_dataset_hash="current",
    )

    assert report.dataset_hash_matches is False
    written = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert written["dataset_hash_matches"] is False


def test_register_rejects_wrong_book_or_base_model(
    tmp_path: Path, adapter_factory: AdapterFactory
) -> None:
    """Wrong book or base model fails friendly and copies nothing (REQ-TR-008)."""
    destination = tmp_path / "adapter"

    wrong_book = adapter_factory(tmp_path / "book", book_id="other-book")
    with pytest.raises(TrainingError, match="other-book"):
        register_adapter(
            wrong_book,
            destination,
            book_id="the-lantern-keeper",
            expected_base_model="Qwen/Qwen3-4B",
        )
    assert not destination.exists()

    wrong_model = adapter_factory(tmp_path / "model", base_model="Qwen/Qwen2.5-3B")
    with pytest.raises(TrainingError, match="base model"):
        register_adapter(
            wrong_model,
            destination,
            book_id="the-lantern-keeper",
            expected_base_model="Qwen/Qwen3-4B",
        )
    assert not destination.exists()


def test_register_rejects_invalid_adapter_directories(
    tmp_path: Path, adapter_factory: AdapterFactory
) -> None:
    """Missing/invalid run.json and missing weights fail friendly (REQ-TR-008)."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TrainingError, match=r"run\.json"):
        register_adapter(empty, tmp_path / "out", book_id="b", expected_base_model="m")

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "run.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(TrainingError, match=r"Invalid adapter run\.json"):
        register_adapter(
            invalid, tmp_path / "out", book_id="b", expected_base_model="m"
        )

    incomplete = adapter_factory(tmp_path / "incomplete")
    (incomplete / "adapter_model.safetensors").unlink()
    with pytest.raises(TrainingError, match=r"adapter_model\.safetensors"):
        register_adapter(
            incomplete,
            tmp_path / "out",
            book_id="the-lantern-keeper",
            expected_base_model="Qwen/Qwen3-4B",
        )
