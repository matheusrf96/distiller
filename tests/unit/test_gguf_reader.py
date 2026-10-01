"""Unit tests for the dependency-free GGUF reader."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import pytest

from distiller.exceptions import GGUFError
from distiller.gguf import GgufMetadata, read_gguf

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    GgufFactory = Callable[..., Path]


def test_read_metadata_extracts_identity_and_counts(
    tmp_path: Path, gguf_factory: GgufFactory
) -> None:
    """A synthetic GGUF parses to its identity and counts (REQ-GG-001, AC1)."""
    path = gguf_factory(
        tmp_path / "model.gguf",
        tensors=[("token_embd.weight", (128, 64)), ("output.weight", (32,))],
        extra_metadata={
            "general.quantization_version": 2,
            "tokenizer.ggml.tokens": ["a", "b", "c"],
        },
    )

    metadata = read_gguf(path)

    assert isinstance(metadata, GgufMetadata)
    assert metadata.version == 3
    assert metadata.architecture == "qwen3"
    assert metadata.model_name == "distiller-lantern-q4_k_m"
    assert metadata.file_type == 15
    assert metadata.quantization == "Q4_K_M"
    assert metadata.tensor_count == 2
    assert metadata.parameter_count == 128 * 64 + 32


def test_rejects_non_gguf_bad_version_and_truncated_files(
    tmp_path: Path, gguf_factory: GgufFactory
) -> None:
    """Bad magic, bad version and truncation name the file and problem (AC2)."""
    not_gguf = tmp_path / "not.gguf"
    not_gguf.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(GGUFError, match=r"not\.gguf.*not a GGUF"):
        read_gguf(not_gguf)

    bad_version = gguf_factory(tmp_path / "v9.gguf", version=9)
    with pytest.raises(GGUFError, match=r"v9\.gguf.*version 9"):
        read_gguf(bad_version)

    truncated_header = gguf_factory(tmp_path / "short.gguf", truncate=12)
    with pytest.raises(GGUFError, match=r"short\.gguf.*truncated"):
        read_gguf(truncated_header)

    truncated_metadata = gguf_factory(tmp_path / "short-meta.gguf", truncate=30)
    with pytest.raises(GGUFError, match=r"short-meta\.gguf.*truncated"):
        read_gguf(truncated_metadata)

    missing = tmp_path / "missing.gguf"
    with pytest.raises(GGUFError, match=r"missing\.gguf"):
        read_gguf(missing)


def test_rejects_absurd_string_lengths(tmp_path: Path) -> None:
    """A corrupt string length is rejected before allocating memory (AC2)."""
    payload = b"GGUF" + struct.pack("<I", 3)
    payload += struct.pack("<QQ", 0, 1)
    payload += struct.pack("<Q", len("general.name")) + b"general.name"
    payload += struct.pack("<I", 8)  # string value type
    payload += struct.pack("<Q", 1 << 62)  # absurd declared length
    path = tmp_path / "huge.gguf"
    path.write_bytes(payload)

    with pytest.raises(GGUFError, match=r"huge\.gguf.*string"):
        read_gguf(path)
