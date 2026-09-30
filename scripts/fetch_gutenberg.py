#!/usr/bin/env python3
# Every URL used below comes from the fixed HTTPS allow-list defined in this module.
# ruff: noqa: S310
"""Download a Project Gutenberg book and build a matching EPUB + PDF fixture pair.

The EPUB is fetched from Gutenberg; the PDF is generated from the parsed text so
both formats carry identical content for parser comparisons.

Run from the repository root:

    uv run python scripts/fetch_gutenberg.py --id 1661 --out fixtures/
    uv run python scripts/fetch_gutenberg.py --id 1661 --out fixtures/ --slug sherlock
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pymupdf

from distiller.config import load_settings
from distiller.ingest import ingest_book
from distiller.utils import wrap_text

USER_AGENT = "distiller-fixture-fetcher/0.1 (local development)"

CANDIDATE_URLS = (
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.epub",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}-images.epub",
    "https://www.gutenberg.org/ebooks/{id}.epub.noimages",
)


def fetch_epub(book_id: int, dest: Path) -> Path:
    if dest.exists():
        print(f"EPUB already present: {dest}")
        return dest
    last_error: Exception | None = None
    for template in CANDIDATE_URLS:
        url = template.format(id=book_id)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            dest.write_bytes(payload)
            print(f"Fetched {len(payload):,} bytes from {url}")
            return dest
        except Exception as exc:
            last_error = exc
            print(f"  failed {url}: {exc}")
    raise SystemExit(f"Could not download Gutenberg #{book_id}: {last_error}")


def build_pdf(epub_path: Path, pdf_path: Path) -> None:
    doc = ingest_book(epub_path, settings=load_settings())

    out = pymupdf.open()
    for chapter in doc.chapters:
        page = out.new_page()
        y = 90.0
        page.insert_text((72, y), chapter.title, fontsize=18)
        y += 34
        for block in chapter.blocks:
            text = block.text.strip()
            if not text:
                continue
            size = 14 if block.type == "heading" else 11
            for line in wrap_text(text, 90):
                if y > 780:
                    page = out.new_page()
                    y = 90.0
                page.insert_text((72, y), line, fontsize=size)
                y += size + 6
            y += 6
    out.set_metadata({"title": doc.title, "author": ", ".join(doc.authors)})
    out.save(str(pdf_path))
    out.close()
    print(f"Wrote {pdf_path} ({pdf_path.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--id", type=int, required=True, help="Project Gutenberg book id (e.g. 1661)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("fixtures"), help="Output directory"
    )
    parser.add_argument(
        "--slug", default=None, help="Output file stem (default: gutenberg-<id>)"
    )
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    slug = args.slug or f"gutenberg-{args.id}"
    epub_path = fetch_epub(args.id, args.out / f"{slug}.epub")
    if not args.no_pdf:
        build_pdf(epub_path, args.out / f"{slug}.pdf")


if __name__ == "__main__":
    main()
