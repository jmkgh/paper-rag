from dataclasses import replace
from pathlib import Path

import pytest
import requests

from paper_rag import download
from paper_rag.config import load_config
from paper_rag.download import Paper, download_all, read_papers

HEADER = "id,title,doi,url,license\n"


def _csv(tmp_path: Path, body: str, header: str = HEADER) -> Path:
    path = tmp_path / "papers.csv"
    path.write_text(header + body, encoding="utf-8")
    return path


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200, content_type: str = "application/pdf"):
        self.content = content
        self.status_code = status
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def fake_get(monkeypatch):
    """Replace requests.get with a lookup table url -> FakeResponse; records calls."""
    responses: dict[str, FakeResponse] = {}
    calls: list[str] = []

    def _get(url, **kwargs):
        calls.append(url)
        return responses[url]

    monkeypatch.setattr(download.requests, "get", _get)
    return responses, calls


def _paper(pid: str) -> Paper:
    return Paper(
        id=pid, title="T", doi="10.1/x", url=f"https://example.org/{pid}.pdf", license="CC BY 4.0"
    )


# --- read_papers ---------------------------------------------------------------------


def test_read_papers_parses_rows(tmp_path):
    path = _csv(tmp_path, "smith2020,A title,10.1/a,https://example.org/a.pdf,CC BY 4.0\n")
    assert read_papers(path) == [
        Paper("smith2020", "A title", "10.1/a", "https://example.org/a.pdf", "CC BY 4.0")
    ]


def test_read_papers_rejects_missing_column(tmp_path):
    path = _csv(tmp_path, "a,T,10.1/a,https://x/a.pdf\n", header="id,title,doi,url\n")
    with pytest.raises(ValueError, match="license"):
        read_papers(path)


def test_read_papers_rejects_duplicate_id(tmp_path):
    row = "a,T,10.1/a,https://x/a.pdf,CC BY 4.0\n"
    with pytest.raises(ValueError, match="duplicate"):
        read_papers(_csv(tmp_path, row + row))


@pytest.mark.parametrize("bad_id", ["Smith2020", "a b", "../evil", ""])
def test_read_papers_rejects_invalid_id(tmp_path, bad_id):
    with pytest.raises(ValueError, match="invalid id"):
        read_papers(_csv(tmp_path, f"{bad_id},T,10.1/a,https://x/a.pdf,CC BY 4.0\n"))


def test_read_papers_warns_on_non_cc_by(tmp_path, caplog):
    read_papers(_csv(tmp_path, "a,T,10.1/a,https://x/a.pdf,CC BY-NC 4.0\n"))
    assert "expected CC BY" not in caplog.text  # CC BY-NC still starts with "CC BY"
    read_papers(_csv(tmp_path, "a,T,10.1/a,https://x/a.pdf,all rights reserved\n"))
    assert "expected CC BY" in caplog.text


# --- download_all --------------------------------------------------------------------


def test_downloads_pdf(tmp_path, fake_get):
    responses, _ = fake_get
    paper = _paper("a")
    responses[paper.url] = FakeResponse(b"%PDF-1.7 ...")

    outcome = download_all([paper], tmp_path)

    assert outcome["downloaded"] == ["a"]
    assert (tmp_path / "a.pdf").read_bytes() == b"%PDF-1.7 ..."


def test_skips_existing_file(tmp_path, fake_get):
    _, calls = fake_get
    (tmp_path / "a.pdf").write_bytes(b"%PDF existing")

    outcome = download_all([_paper("a")], tmp_path)

    assert outcome["skipped"] == ["a"]
    assert calls == []


def test_force_redownloads(tmp_path, fake_get):
    responses, _ = fake_get
    paper = _paper("a")
    responses[paper.url] = FakeResponse(b"%PDF new")
    (tmp_path / "a.pdf").write_bytes(b"%PDF old")

    download_all([paper], tmp_path, force=True)

    assert (tmp_path / "a.pdf").read_bytes() == b"%PDF new"


def test_html_response_is_rejected_and_leaves_no_file(tmp_path, fake_get):
    responses, _ = fake_get
    paper = _paper("a")
    responses[paper.url] = FakeResponse(b"<html>cookie wall</html>", content_type="text/html")

    outcome = download_all([paper], tmp_path)

    assert outcome["failed"] == ["a"]
    assert list(tmp_path.iterdir()) == []


def test_failure_does_not_stop_other_downloads(tmp_path, fake_get):
    responses, _ = fake_get
    bad, good = _paper("bad"), _paper("good")
    responses[bad.url] = FakeResponse(b"", status=404)
    responses[good.url] = FakeResponse(b"%PDF ok")

    outcome = download_all([bad, good], tmp_path)

    assert outcome == {"downloaded": ["good"], "skipped": [], "failed": ["bad"]}


def test_main_returns_1_on_failure(tmp_path, fake_get, monkeypatch):
    responses, _ = fake_get
    csv_path = _csv(tmp_path, "a,T,10.1/a,https://example.org/a.pdf,CC BY 4.0\n")
    responses["https://example.org/a.pdf"] = FakeResponse(b"", status=500)

    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    paths = replace(cfg.paths, papers_csv=csv_path, pdf_dir=tmp_path / "pdfs")
    monkeypatch.setattr(download, "load_config", lambda _: replace(cfg, paths=paths))

    assert download.main([]) == 1
