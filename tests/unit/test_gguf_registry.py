"""Unit tests for the GGUF registry."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from distiller.exceptions import GGUFError
from distiller.gguf import load_gguf_report, register_gguf
from distiller.rag.prompts import build_system_prompt

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    GgufFactory = Callable[..., Path]


def test_register_copies_and_writes_the_report(
    tmp_path: Path, gguf_factory: GgufFactory
) -> None:
    """A valid GGUF is copied, hashed, reported; re-registration replaces (AC3)."""
    source = gguf_factory(tmp_path / "lantern-q4_k_m.gguf")
    destination = tmp_path / "artifacts" / "book" / "training" / "gguf"

    report = register_gguf(
        source,
        destination,
        book_id="the-lantern-keeper",
        book_title="The Lantern Keeper",
    )

    model_file = destination / "model.gguf"
    assert model_file.read_bytes() == source.read_bytes()
    assert report.model_name == "distiller-the-lantern-keeper"
    assert report.source_file == "lantern-q4_k_m.gguf"
    assert report.sha256 == hashlib.sha256(model_file.read_bytes()).hexdigest()
    assert report.size_bytes == model_file.stat().st_size
    assert report.metadata.quantization == "Q4_K_M"
    assert report.metadata.parameter_count == 128 * 64 + 32
    assert load_gguf_report(destination / "gguf.json") == report
    modelfile = (destination / "Modelfile").read_text(encoding="utf-8")
    assert build_system_prompt("The Lantern Keeper") in modelfile
    serve_script = (destination / "serve.sh").read_text(encoding="utf-8")
    assert "distiller-the-lantern-keeper" in serve_script

    written = json.loads((destination / "gguf.json").read_text(encoding="utf-8"))
    assert written["sha256"] == report.sha256
    assert written["metadata"]["architecture"] == "qwen3"

    replacement = gguf_factory(tmp_path / "second.gguf", tensors=[("a.weight", (2, 2))])
    second = register_gguf(
        replacement,
        destination,
        book_id="the-lantern-keeper",
        book_title="The Lantern Keeper",
    )
    assert second.sha256 != report.sha256
    assert model_file.read_bytes() == replacement.read_bytes()
    assert load_gguf_report(destination / "gguf.json") == second


def test_register_rejects_invalid_file_without_copying(tmp_path: Path) -> None:
    """An invalid or missing file fails before anything is copied (AC4)."""
    destination = tmp_path / "artifacts" / "book" / "training" / "gguf"

    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOT-GGUF")
    with pytest.raises(GGUFError, match=r"bad\.gguf"):
        register_gguf(bad, destination, book_id="book", book_title="Book")
    assert not destination.exists()

    with pytest.raises(GGUFError, match=r"missing\.gguf"):
        register_gguf(
            tmp_path / "missing.gguf", destination, book_id="book", book_title="Book"
        )
    assert not destination.exists()


def test_register_honours_the_model_name_and_adapter_identity(
    tmp_path: Path, gguf_factory: GgufFactory
) -> None:
    """The served name override and the adapter provenance are recorded (AC3)."""
    source = gguf_factory(tmp_path / "model.gguf")
    destination = tmp_path / "gguf"

    report = register_gguf(
        source,
        destination,
        book_id="the-lantern-keeper",
        book_title="The Lantern Keeper",
        model_name="custom-name",
        adapter={"base_model": "Qwen/Qwen3-4B", "dataset_hash": "abc"},
    )

    assert report.model_name == "custom-name"
    assert report.adapter["base_model"] == "Qwen/Qwen3-4B"
    assert "custom-name" in (destination / "serve.sh").read_text(encoding="utf-8")


def test_load_gguf_report_rejects_missing_and_corrupt_reports(tmp_path: Path) -> None:
    """Missing or corrupt ``gguf.json`` fail with GGUFError."""
    with pytest.raises(GGUFError, match="No GGUF registration"):
        load_gguf_report(tmp_path / "gguf.json")

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.raises(GGUFError, match="Invalid GGUF report"):
        load_gguf_report(corrupt)
