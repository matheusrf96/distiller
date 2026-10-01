"""Dependency-free GGUF reader (Phase 4).

GGUF is llama.cpp's binary container: a fixed header, a key-value metadata
table and a tensor-info table. The repository only needs the identity of a
downloaded file (architecture, name, quantization, counts), so this module
parses exactly that with the standard library — no ``gguf`` package and no
llama.cpp binary.
"""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING, Any

from ..exceptions import GGUFError
from ..models import DomainModel

if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO

GGUF_MAGIC = 0x46554747  # b"GGUF" read as a little-endian u32.
SUPPORTED_VERSIONS = frozenset({2, 3})  # matches llama.cpp's gguf reader.
MAX_STRING_LENGTH = 1024 * 1024 * 1024
MAX_ARRAY_ITEMS = 1024 * 1024 * 1024
MAX_TENSOR_DIMENSIONS = 4  # GGUF tensors are at most 4-dimensional.

# llama.cpp `llama_ftype` values -> quantization labels (common subset).
FILE_TYPE_LABELS: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
}

# GGUF metadata value types (spec v2/v3, types 0..12).
(
    _UINT8,
    _INT8,
    _UINT16,
    _INT16,
    _UINT32,
    _INT32,
    _FLOAT32,
    _BOOL,
    _STRING,
    _ARRAY,
    _UINT64,
    _INT64,
    _FLOAT64,
) = range(13)

_SCALAR_FORMATS: dict[int, str] = {
    _UINT8: "B",
    _INT8: "b",
    _UINT16: "H",
    _INT16: "h",
    _UINT32: "I",
    _INT32: "i",
    _FLOAT32: "f",
    _BOOL: "?",
    _UINT64: "Q",
    _INT64: "q",
    _FLOAT64: "d",
}

__all__ = [
    "FILE_TYPE_LABELS",
    "GGUF_MAGIC",
    "SUPPORTED_VERSIONS",
    "GgufMetadata",
    "read_gguf",
]


class GgufMetadata(DomainModel):
    """Identity parsed from a GGUF header, metadata table and tensor table.

    Attributes:
        version: GGUF container version.
        architecture: ``general.architecture`` (e.g. ``qwen3``).
        model_name: ``general.name``, when the exporter wrote one.
        file_type: Raw ``general.file_type`` value.
        quantization: Human label for the file type (e.g. ``Q4_K_M``).
        tensor_count: Tensors declared in the tensor-info table.
        parameter_count: Sum of the products of every tensor's dimensions.
    """

    version: int
    architecture: str | None = None
    model_name: str | None = None
    file_type: int | None = None
    quantization: str | None = None
    tensor_count: int = 0
    parameter_count: int = 0


def read_gguf(path: Path) -> GgufMetadata:
    """Parse the header, metadata and tensor-info table of a GGUF file.

    Args:
        path: GGUF file downloaded from the training machine.

    Returns:
        The parsed identity: version, architecture, name, quantization label,
        tensor count and parameter count.

    Raises:
        GGUFError: When the file is missing, is not a GGUF file, uses an
            unsupported version or is truncated inside a parsed section.
    """
    if not path.exists():
        raise GGUFError(
            f"GGUF file not found: {path}. Download the Q4_K_M export from the "
            f"training machine first (see docs/gguf-runbook.md)."
        )
    try:
        with path.open("rb") as handle:
            return _parse(path, handle)
    except OSError as exc:
        raise GGUFError(f"Could not read GGUF file {path}: {exc}") from exc


def _parse(path: Path, handle: BinaryIO) -> GgufMetadata:
    """Parse one open GGUF file, validating the header first."""
    magic = _unpack("<I", handle, path, "header")[0]
    if magic != GGUF_MAGIC:
        raise GGUFError(f"{path} is not a GGUF file (bad magic 0x{magic:08X}).")
    version = _unpack("<I", handle, path, "header")[0]
    if version not in SUPPORTED_VERSIONS:
        raise GGUFError(
            f"GGUF file {path} uses unsupported version {version} "
            f"(supported: {sorted(SUPPORTED_VERSIONS)})."
        )
    tensor_count, metadata_count = _unpack("<QQ", handle, path, "header")
    metadata = _read_metadata(handle, metadata_count, path)
    parameter_count = _read_tensor_table(handle, tensor_count, path)
    file_type = _int_or_none(metadata.get("general.file_type"))
    return GgufMetadata(
        version=version,
        architecture=_string_or_none(metadata.get("general.architecture")),
        model_name=_string_or_none(metadata.get("general.name")),
        file_type=file_type,
        quantization=_quantization_label(file_type),
        tensor_count=tensor_count,
        parameter_count=parameter_count,
    )


def _read_metadata(handle: BinaryIO, count: int, path: Path) -> dict[str, object]:
    """Read every key-value metadata pair into a plain dictionary."""
    metadata: dict[str, object] = {}
    for _ in range(count):
        key = _read_string(handle, path, "metadata key")
        value_type = _unpack("<I", handle, path, "metadata value type")[0]
        metadata[key] = _read_value(handle, value_type, path, key)
    return metadata


def _read_value(handle: BinaryIO, value_type: int, path: Path, label: str) -> object:
    """Read one metadata value of any supported type (arrays recurse)."""
    if value_type == _STRING:
        return _read_string(handle, path, f"metadata value for '{label}'")
    if value_type == _ARRAY:
        return _read_array(handle, path, label)
    fmt = _SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise GGUFError(
            f"GGUF file {path} uses unknown metadata type {value_type} "
            f"for key '{label}'."
        )
    return _unpack(f"<{fmt}", handle, path, f"metadata value for '{label}'")[0]


def _read_array(handle: BinaryIO, path: Path, label: str) -> list[object]:
    """Read a metadata array: element type, length, then the elements."""
    element_type = _unpack("<I", handle, path, f"array type for '{label}'")[0]
    count = _unpack("<Q", handle, path, f"array length for '{label}'")[0]
    if count > MAX_ARRAY_ITEMS:
        raise GGUFError(
            f"GGUF file {path} declares an array of {count} elements for key "
            f"'{label}' (limit {MAX_ARRAY_ITEMS})."
        )
    return [_read_value(handle, element_type, path, label) for _ in range(count)]


def _read_tensor_table(handle: BinaryIO, count: int, path: Path) -> int:
    """Walk the tensor-info table and sum every tensor's parameter count."""
    parameter_count = 0
    for index in range(count):
        section = f"tensor {index}"
        _read_string(handle, path, f"{section} name")
        dimensions = _read_tensor_dimensions(handle, path, section)
        _unpack("<I", handle, path, f"{section} ggml type")
        _unpack("<Q", handle, path, f"{section} offset")
        parameter_count += math.prod(dimensions)
    return parameter_count


def _read_tensor_dimensions(
    handle: BinaryIO, path: Path, section: str
) -> tuple[int, ...]:
    """Read one tensor's dimension count and shape."""
    dimension_count = _unpack("<I", handle, path, f"{section} dimension count")[0]
    if dimension_count > MAX_TENSOR_DIMENSIONS:
        raise GGUFError(
            f"GGUF file {path} declares {dimension_count} dimensions for "
            f"{section} (at most {MAX_TENSOR_DIMENSIONS} are valid)."
        )
    return tuple(_unpack(f"<{dimension_count}Q", handle, path, f"{section} shape"))


def _read_string(handle: BinaryIO, path: Path, section: str) -> str:
    """Read a length-prefixed UTF-8 string."""
    length = _unpack("<Q", handle, path, f"{section} length")[0]
    if length > MAX_STRING_LENGTH:
        raise GGUFError(
            f"GGUF file {path} declares a {length}-byte string in the {section} "
            f"(limit {MAX_STRING_LENGTH})."
        )
    return _read_exact(handle, length, path, section).decode("utf-8", errors="replace")


def _unpack(fmt: str, handle: BinaryIO, path: Path, section: str) -> tuple[Any, ...]:
    """Read exactly ``struct.calcsize(fmt)`` bytes and unpack them."""
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(handle, size, path, section))


def _read_exact(handle: BinaryIO, size: int, path: Path, section: str) -> bytes:
    """Read ``size`` bytes or raise a truncation error naming the section."""
    data = handle.read(size)
    if len(data) != size:
        raise GGUFError(
            f"GGUF file {path} is truncated while reading the {section} "
            f"(expected {size} bytes, found {len(data)})."
        )
    return data


def _quantization_label(file_type: int | None) -> str | None:
    if file_type is None:
        return None
    return FILE_TYPE_LABELS.get(file_type, f"unknown({file_type})")


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
