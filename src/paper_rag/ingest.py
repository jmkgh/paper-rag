"""Turn PDFs into page-aware text chunks: PDF -> pages -> (references removed) -> chunks.

Usage (prints per-paper statistics, does not index anything):
    python -m paper_rag.ingest --config config.yaml

Design decisions:
- `page` is the 1-based PDF page, not the journal's printed page number.
- Chunks never cross a page boundary, so every chunk can be cited with exactly one page.
- Chunk size is measured in characters (no tokenizer dependency).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from paper_rag.config import Config, load_config
from paper_rag.download import read_papers

log = logging.getLogger(__name__)

# A line consisting only of one of these words starts the bibliography.
REFERENCES_HEADING = re.compile(
    r"^\s*(references|bibliography|literature cited)\s*$", re.IGNORECASE
)
# Default plain-text flags, plus re-joining words hyphenated at line ends ("vary-\ning"),
# minus ligature preservation so that "\ufb01" becomes "fi" and "refinement" stays searchable.
TEXT_FLAGS = (pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_DEHYPHENATE) & ~pymupdf.TEXT_PRESERVE_LIGATURES


@dataclass(frozen=True)
class Chunk:
    paper_id: str
    title: str
    page: int  # 1-based PDF page
    chunk_id: str  # stable across runs, e.g. "krause2025-p3-c2"
    text: str


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return (page_number, raw_text) for every page; line breaks are kept."""
    with pymupdf.open(pdf_path) as doc:
        return [(i, page.get_text("text", flags=TEXT_FLAGS)) for i, page in enumerate(doc, start=1)]


def strip_references(pages: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], int | None]:
    """Drop everything from the last references heading to the end of the document.

    The *last* matching heading is used so that an earlier mention (e.g. in a table of
    contents) does not cut off the body. Returns the remaining pages and the page on
    which the cut happened (None if no heading was found).
    """
    for idx in range(len(pages) - 1, -1, -1):
        page_no, text = pages[idx]
        lines = text.splitlines()
        heading_lines = [i for i, line in enumerate(lines) if REFERENCES_HEADING.match(line)]
        if heading_lines:
            kept = "\n".join(lines[: heading_lines[-1]])
            return pages[:idx] + [(page_no, kept)], page_no
    return pages, None


def normalize_whitespace(text: str) -> str:
    """Collapse line breaks and repeated whitespace into single spaces."""
    return " ".join(text.split())


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into windows of at most `size` characters overlapping by ~`overlap`.

    Cuts are moved back to the last space so words stay intact (a single word longer
    than `size` is hard-cut). No trailing chunk is produced that would consist only of
    overlap with the previous one.
    """
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + 1, end + 1)
            if cut > start:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(end - overlap, 0)
        if next_start > 0 and text[next_start - 1] != " ":  # landed mid-word: skip to next word
            space = text.find(" ", next_start, end)
            next_start = space + 1 if space != -1 else end
        start = next_start if next_start > start else end  # always make progress
    return chunks


def chunk_pages(
    paper_id: str, title: str, pages: list[tuple[int, str]], size: int, overlap: int
) -> list[Chunk]:
    """Chunk each page separately and attach metadata."""
    chunks: list[Chunk] = []
    for page_no, text in pages:
        for n, piece in enumerate(chunk_text(normalize_whitespace(text), size, overlap)):
            chunk_id = f"{paper_id}-p{page_no}-c{n}"
            chunks.append(Chunk(paper_id, title, page_no, chunk_id, piece))
    return chunks


def ingest_pdf(paper_id: str, title: str, pdf_path: Path, size: int, overlap: int) -> list[Chunk]:
    """Full pipeline for one PDF."""
    pages, cut_page = strip_references(extract_pages(pdf_path))
    if cut_page is None:
        log.warning("%s: no references heading found, indexing all pages", paper_id)
    else:
        log.info("%s: references removed from page %d on", paper_id, cut_page)
    return chunk_pages(paper_id, title, pages, size, overlap)


def ingest_all(cfg: Config) -> list[Chunk]:
    """Chunk every paper from papers.csv whose PDF is present."""
    chunks: list[Chunk] = []
    for paper in read_papers(cfg.paths.papers_csv):
        pdf_path = cfg.paths.pdf_dir / f"{paper.id}.pdf"
        if not pdf_path.exists():
            log.warning("%s: PDF missing (%s), run paper_rag.download first", paper.id, pdf_path)
            continue
        chunks += ingest_pdf(
            paper.id,
            paper.title,
            pdf_path,
            cfg.chunking.chunk_size,
            cfg.chunking.chunk_overlap,
        )
    return chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk all PDFs and print statistics.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    chunks = ingest_all(load_config(args.config))
    per_paper = Counter(c.paper_id for c in chunks)
    pages = {pid: len({c.page for c in chunks if c.paper_id == pid}) for pid in per_paper}
    for pid, n in per_paper.items():
        lengths = [len(c.text) for c in chunks if c.paper_id == pid]
        print(
            f"{pid}: {pages[pid]} pages, {n} chunks, "
            f"chars/chunk min {min(lengths)} / mean {sum(lengths) // n} / max {max(lengths)}"
        )
    print(f"total: {len(chunks)} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
