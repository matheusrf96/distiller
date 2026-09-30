"""Small shared helpers: slugs, hashes, JSON I/O and text wrapping."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def slugify(text: str, max_len: int = 60) -> str:
    """Convert arbitrary text into a filesystem-safe slug.

    Args:
        text: Text to convert (book title or file stem).
        max_len: Maximum slug length; longer slugs are truncated.

    Returns:
        Lowercase, hyphen-separated ASCII slug (never empty).
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return (slug[:max_len].rstrip("-")) or "book"


def stable_hash_hex(text: str, length: int = 40) -> str:
    """Hash text with SHA-256 and return the first ``length`` hex characters.

    Used for content addressing (chunk ids, token bucketing), not for security.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def wrap_text(text: str, width: int = 88) -> list[str]:
    """Wrap text greedily at word boundaries.

    Args:
        text: Text to wrap.
        width: Maximum line width in characters.

    Returns:
        List of lines, always with at least one (possibly empty) entry.
    """
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` as pretty UTF-8 JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    """Write rows as JSON Lines, serializing pydantic models when needed.

    Returns:
        Number of rows written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump()
            file_handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed objects from a JSON Lines file, skipping blank lines."""
    with Path(path).open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)
