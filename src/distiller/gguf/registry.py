"""GGUF registry: validate a downloaded export and install it (Phase 4).

Mirrors the adapter registry: the artifacts directory is self-contained
(``training/gguf/model.gguf``), ``gguf.json`` records the identity, and
re-registration replaces the previous copy and report. The report's hash is
computed over the registered copy, and the registered adapter identity
(``training/adapter/run.json``) is carried when available.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..exceptions import GGUFError
from ..models import DomainModel
from ..utils import read_json, write_json
from .reader import GgufMetadata, read_gguf
from .serving import ServingProfile, render_modelfile, render_serve_script

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_FILE_NAME = "model.gguf"

__all__ = ["MODEL_FILE_NAME", "GgufReport", "load_gguf_report", "register_gguf"]


class GgufReport(DomainModel):
    """Identity of one registered GGUF file (``gguf.json``).

    Attributes:
        book_id: Book the GGUF was registered for.
        model_name: Served model name (Ollama name and llama-server alias).
        source_file: Name of the downloaded file that was registered.
        sha256: SHA-256 of the registered copy.
        size_bytes: Size of the registered copy in bytes.
        created_at: UTC timestamp of the registration.
        metadata: Parsed GGUF identity (version, architecture, quantization...).
        adapter: Registered adapter provenance when available (SDD-0004).
    """

    book_id: str
    model_name: str
    source_file: str
    sha256: str
    size_bytes: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: GgufMetadata
    adapter: dict[str, Any] = Field(default_factory=dict)


def register_gguf(
    source: Path,
    destination: Path,
    *,
    book_id: str,
    book_title: str,
    model_name: str | None = None,
    adapter: dict[str, Any] | None = None,
    profile: ServingProfile | None = None,
) -> GgufReport:
    """Validate a downloaded GGUF and install it into the book's artifacts.

    The file is parsed before anything is copied, then copied to
    ``destination/model.gguf``; the report hash is computed over that copy.
    Re-registration replaces the previous model, report, Modelfile and script.

    Args:
        source: GGUF file downloaded from the training machine.
        destination: Registry directory (``training/gguf``).
        book_id: Book the GGUF was exported for.
        book_title: Title embedded in the Modelfile system prompt.
        model_name: Served model name; defaults to ``distiller-<book_id>``.
        adapter: Registered adapter provenance, when available.
        profile: Serving parameters; defaults to the pinned 4 GB profile.

    Returns:
        The registration report written to ``destination/gguf.json``.

    Raises:
        GGUFError: When the file is invalid or missing.
    """
    metadata = read_gguf(source)
    destination.mkdir(parents=True, exist_ok=True)
    registered = destination / MODEL_FILE_NAME
    if source.resolve() != registered.resolve():
        shutil.copy2(source, registered)
    served_name = model_name or f"distiller-{book_id}"
    report = GgufReport(
        book_id=book_id,
        model_name=served_name,
        source_file=source.name,
        sha256=_sha256(registered),
        size_bytes=registered.stat().st_size,
        metadata=metadata,
        adapter=dict(adapter or {}),
    )
    write_json(destination / "gguf.json", json.loads(report.model_dump_json()))
    serving_profile = profile or ServingProfile()
    (destination / "Modelfile").write_text(
        render_modelfile(book_title, serving_profile), encoding="utf-8"
    )
    (destination / "serve.sh").write_text(
        render_serve_script(served_name, serving_profile), encoding="utf-8"
    )
    return report


def load_gguf_report(report_path: Path) -> GgufReport:
    """Load and validate a ``gguf.json`` registration report.

    Args:
        report_path: Path of the report (``training/gguf/gguf.json``).

    Returns:
        Validated registration report.

    Raises:
        GGUFError: When the report is missing or invalid.
    """
    if not report_path.exists():
        raise GGUFError(
            f"No GGUF registration at {report_path}. "
            f"Run `distiller gguf register <book> <file.gguf>` first."
        )
    try:
        return GgufReport.model_validate(read_json(report_path))
    except (OSError, ValueError) as exc:
        raise GGUFError(f"Invalid GGUF report at {report_path}: {exc}") from exc


def _sha256(path: Path) -> str:
    """Hash a file with SHA-256, streaming in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
