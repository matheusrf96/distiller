"""Unit tests for the Markdown -> Block parser."""

from __future__ import annotations

from distiller.ingest.markdown import blocks_to_markdown, markdown_to_blocks


def test_parses_structured_blocks() -> None:
    """Headings, paragraphs, lists, quotes and code parse into typed blocks."""
    markdown = (
        "# Title\n"
        "\n"
        "Intro paragraph\n"
        "continues on a second line.\n"
        "\n"
        "## Section\n"
        "\n"
        "- item one\n"
        "- item two\n"
        "\n"
        "> a quoted passage\n"
        "\n"
        "```\n"
        "code line\n"
        "```\n"
    )
    blocks = markdown_to_blocks(markdown)

    assert [block.type for block in blocks] == [
        "heading",
        "paragraph",
        "heading",
        "list",
        "quote",
        "code",
    ]
    assert blocks[0].text == "Title" and blocks[0].level == 1
    assert blocks[1].text == "Intro paragraph continues on a second line."
    assert blocks[2].text == "Section" and blocks[2].level == 2
    assert blocks[3].text.splitlines() == ["- item one", "- item two"]
    assert blocks[4].text == "a quoted passage"
    assert blocks[5].text == "code line"


def test_parses_tables_and_skips_rules() -> None:
    """Tables survive parsing and horizontal rules are dropped."""
    markdown = "| a | b |\n| - | - |\n| 1 | 2 |\n\n---\n\nAfter the rule.\n"
    blocks = markdown_to_blocks(markdown)

    assert blocks[0].type == "table"
    assert blocks[0].text.splitlines()[0] == "| a | b |"
    assert blocks[1].type == "paragraph"
    assert blocks[1].text == "After the rule."


def test_roundtrip_keeps_headings() -> None:
    """Markdown rendering preserves headings and body text."""
    blocks = markdown_to_blocks("# One\n\nBody.\n\n## Two\n\nMore.\n")
    rendered = blocks_to_markdown(blocks)

    assert "# One" in rendered
    assert "## Two" in rendered
    assert "Body." in rendered


def test_page_number_annotates_blocks() -> None:
    """The optional page argument annotates every parsed block."""
    blocks = markdown_to_blocks("Body text.\n", page=7)

    assert blocks and all(block.page == 7 for block in blocks)
