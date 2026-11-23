"""Download the PDFs listed in `papers.csv` into the configured PDF directory.

Usage:
    python -m paper_rag.download --config config.yaml [--force]

The CSV holds only metadata (id, title, doi, url, license); PDFs are not checked in.
`url` must point directly to the PDF, not to the article landing page.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

from paper_rag.config import load_config

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("id", "title", "doi", "url", "license")
# The id becomes the filename and the citation key `[paper_id, p. x]`, so keep it simple.
ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")
USER_AGENT = "paper-rag/0.1"
TIMEOUT_S = 60


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    doi: str
    url: str
    license: str


def read_papers(csv_path: Path) -> list[Paper]:
    """Parse and validate the paper list; raises ValueError on structural problems."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path}: missing column(s) {missing}")
        rows = [{k: (row[k] or "").strip() for k in REQUIRED_COLUMNS} for row in reader]

    papers: list[Paper] = []
    seen: set[str] = set()
    for line_no, row in enumerate(rows, start=2):  # line 1 is the header
        paper = Paper(**row)
        if not ID_PATTERN.match(paper.id):
            raise ValueError(f"{csv_path}:{line_no}: invalid id {paper.id!r} (use [a-z0-9_-])")
        if paper.id in seen:
            raise ValueError(f"{csv_path}:{line_no}: duplicate id {paper.id!r}")
        if not paper.url:
            raise ValueError(f"{csv_path}:{line_no}: empty url for {paper.id!r}")
        if not paper.license.upper().startswith("CC BY"):
            log.warning("%s: license is %r, expected CC BY", paper.id, paper.license)
        seen.add(paper.id)
        papers.append(paper)
    return papers


def download_pdf(url: str, target: Path) -> None:
    """Download `url` to `target` atomically; raises if the response is not a PDF.

    Checking the `%PDF` magic bytes is more reliable than the Content-Type header:
    publishers sometimes answer with an HTML page (cookie wall, bot check) and status 200.
    """
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError(
            f"response is not a PDF (Content-Type: {response.headers.get('Content-Type')})"
        )

    tmp = target.with_suffix(".pdf.part")
    tmp.write_bytes(response.content)
    tmp.replace(target)  # atomic rename: no half-written PDFs after an interruption


def download_all(papers: list[Paper], pdf_dir: Path, force: bool = False) -> dict[str, list[str]]:
    """Download every paper, continuing after failures. Returns ids grouped by outcome."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    outcome: dict[str, list[str]] = {"downloaded": [], "skipped": [], "failed": []}
    for paper in papers:
        target = pdf_dir / f"{paper.id}.pdf"
        if target.exists() and not force:
            outcome["skipped"].append(paper.id)
            continue
        try:
            download_pdf(paper.url, target)
        except (requests.RequestException, ValueError, OSError) as exc:
            log.error("%s: download failed: %s", paper.id, exc)
            outcome["failed"].append(paper.id)
        else:
            log.info("%s: downloaded", paper.id)
            outcome["downloaded"].append(paper.id)
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--force", action="store_true", help="re-download existing PDFs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    papers = read_papers(cfg.paths.papers_csv)
    outcome = download_all(papers, cfg.paths.pdf_dir, force=args.force)
    log.info(
        "downloaded: %d, skipped (already present): %d, failed: %d %s",
        len(outcome["downloaded"]),
        len(outcome["skipped"]),
        len(outcome["failed"]),
        outcome["failed"] or "",
    )
    return 1 if outcome["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
