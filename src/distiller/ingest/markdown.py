"""Minimal Markdown -> Block parser.

Deliberately small and dependency-free: every ingestion backend (Docling,
PyMuPDF4LLM, EPUB HTML conversion) normalizes to Markdown, and this module turns
that Markdown into the structured ``Block`` list the chunker consumes.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ..models import Block

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST = re.compile(r"^\s{0,3}(?:[-*+]|\d{1,3}[.)])\s+")
_QUOTE = re.compile(r"^\s{0,3}>\s?")
_TABLE = re.compile(r"^\s*\|.*\|\s*$")
_HR = re.compile(r"^\s{0,3}(-{3,}|\*{3,}|_{3,})\s*$")

_Parsed = tuple[Block | None, int]
_Reader = Callable[[list[str], int], "_Parsed | None"]


def markdown_to_blocks(markdown_text: str, *, page: int | None = None) -> list[Block]:
    """Parse Markdown into structured blocks.

    Args:
        markdown_text: Markdown produced by an ingestion backend.
        page: Optional 1-based page number annotating every block (PDF backends).

    Returns:
        Ordered blocks (headings, paragraphs, lists, tables, code, quotes).
    """
    lines = markdown_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Block] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        block, index = _read_block(lines, index)
        if block is not None:
            block.page = page
            blocks.append(block)
    return blocks


def _read_block(lines: list[str], index: int) -> _Parsed:
    for reader in _READERS:
        parsed = reader(lines, index)
        if parsed is not None:
            return parsed
    return _read_paragraph(lines, index)


def _read_heading(lines: list[str], index: int) -> _Parsed | None:
    match = _HEADING.match(lines[index])
    if not match:
        return None
    return Block(
        type="heading", text=match.group(2).strip(), level=len(match.group(1))
    ), index + 1


def _read_rule(lines: list[str], index: int) -> _Parsed | None:
    if not _HR.match(lines[index]):
        return None
    return None, index + 1  # horizontal rules carry no content


def _read_fence(lines: list[str], index: int) -> _Parsed | None:
    if not _FENCE.match(lines[index]):
        return None
    fence = lines[index].strip()[:3]
    body, next_index = _consume(
        lines, index + 1, lambda line: not line.strip().startswith(fence)
    )
    next_index += 1  # skip the closing fence
    text = "\n".join(body).strip()
    return (Block(type="code", text=text) if text else None), next_index


def _read_table(lines: list[str], index: int) -> _Parsed | None:
    if not _TABLE.match(lines[index]):
        return None
    rows, next_index = _consume(lines, index, lambda line: bool(_TABLE.match(line)))
    return Block(type="table", text="\n".join(row.strip() for row in rows)), next_index


def _read_quote(lines: list[str], index: int) -> _Parsed | None:
    if not _QUOTE.match(lines[index]):
        return None
    rows, next_index = _consume(lines, index, lambda line: bool(_QUOTE.match(line)))
    text = "\n".join(_QUOTE.sub("", row).rstrip() for row in rows).strip()
    return (Block(type="quote", text=text) if text else None), next_index


def _read_list(lines: list[str], index: int) -> _Parsed | None:
    if not _LIST.match(lines[index]):
        return None
    rows, next_index = _consume(lines, index, _is_list_continuation)
    return Block(type="list", text="\n".join(row.strip() for row in rows)), next_index


def _read_paragraph(lines: list[str], index: int) -> _Parsed:
    rows, next_index = _consume(lines, index, _is_paragraph_line)
    text = " ".join(row.strip() for row in rows).strip()
    return (Block(type="paragraph", text=text) if text else None), next_index


def _consume(
    lines: list[str], start: int, keep_going: Callable[[str], bool]
) -> tuple[list[str], int]:
    collected: list[str] = []
    index = start
    while index < len(lines) and keep_going(lines[index]):
        collected.append(lines[index])
        index += 1
    return collected, index


def _is_list_continuation(line: str) -> bool:
    """List items continue until a blank line or a different block type starts."""
    return bool(line.strip()) and not _starts_other_block(line)


def _is_paragraph_line(line: str) -> bool:
    return bool(line.strip()) and not _starts_any_block(line)


def _starts_other_block(line: str) -> bool:
    return bool(
        _HEADING.match(line)
        or _FENCE.match(line)
        or _QUOTE.match(line)
        or _TABLE.match(line)
        or _HR.match(line)
    )


def _starts_any_block(line: str) -> bool:
    return bool(_LIST.match(line)) or _starts_other_block(line)


_READERS: tuple[_Reader, ...] = (
    _read_heading,
    _read_rule,
    _read_fence,
    _read_table,
    _read_quote,
    _read_list,
)


def blocks_to_markdown(blocks: list[Block]) -> str:
    """Reconstruct Markdown from blocks (used for the ``parsed.md`` artifact)."""
    parts: list[str] = []
    for block in blocks:
        if block.type == "heading" and block.level:
            parts.append(f"{'#' * block.level} {block.text}")
        elif block.type == "code":
            parts.append(f"```\n{block.text}\n```")
        elif block.type == "quote":
            parts.append("\n".join(f"> {line}" for line in block.text.splitlines()))
        else:
            parts.append(block.text)
    return "\n\n".join(part for part in parts if part.strip()) + "\n"
