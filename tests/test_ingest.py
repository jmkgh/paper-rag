from pathlib import Path

import pymupdf
import pytest

from paper_rag.ingest import (
    chunk_pages,
    chunk_text,
    extract_pages,
    ingest_pdf,
    normalize_whitespace,
    strip_references,
)

WORDS = " ".join(f"word{i}" for i in range(400))  # ~3000 characters, no long words


def _make_pdf(path: Path, pages: list[str]) -> Path:
    """Create a small PDF with one text block per page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


# --- chunk_text ----------------------------------------------------------------------


def test_short_text_gives_single_chunk():
    assert chunk_text("a short text", size=100, overlap=10) == ["a short text"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ", size=100, overlap=10) == []


def test_chunks_respect_size_and_keep_words_intact():
    chunks = chunk_text(WORDS, size=200, overlap=30)
    words = set(WORDS.split())
    assert len(chunks) > 1
    for chunk in chunks:
        assert 0 < len(chunk) <= 200
        assert set(chunk.split()) <= words  # every token is a complete word


def test_consecutive_chunks_overlap():
    chunks = chunk_text(WORDS, size=200, overlap=30)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.split()[0] in prev.split()  # next chunk starts inside the previous one


def test_chunks_cover_the_whole_text():
    chunks = chunk_text(WORDS, size=200, overlap=30)
    covered = set()
    for chunk in chunks:
        covered |= set(chunk.split())
    assert covered == set(WORDS.split())
    assert chunks[-1].endswith("word399")


def test_no_overlap_only_tail_chunk():
    chunks = chunk_text(WORDS, size=200, overlap=30)
    # the last chunk must contain something the previous chunk does not
    assert not set(chunks[-1].split()) <= set(chunks[-2].split())


def test_overlong_word_is_hard_cut():
    chunks = chunk_text("x" * 250, size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks).count("x") >= 250


def test_normalize_whitespace():
    assert normalize_whitespace(" a\n b\t\tc  \n") == "a b c"


# --- strip_references ----------------------------------------------------------------


def test_strip_references_mid_page_cuts_rest_and_following_pages():
    pages = [(1, "Intro\nbody"), (2, "Results\nReferences\nSmith 2020"), (3, "Jones 2019")]
    kept, cut = strip_references(pages)
    assert cut == 2
    assert kept == [(1, "Intro\nbody"), (2, "Results")]


def test_strip_references_uses_last_heading():
    pages = [(1, "Contents\nReferences\nIntro"), (2, "Body"), (3, "References\nSmith 2020")]
    kept, cut = strip_references(pages)
    assert cut == 3
    assert [p for p, _ in kept] == [1, 2, 3]
    assert "Intro" in kept[0][1]


def test_strip_references_without_heading_keeps_everything():
    pages = [(1, "Body"), (2, "More body")]
    assert strip_references(pages) == (pages, None)


def test_word_references_inside_a_sentence_is_not_a_heading():
    pages = [(1, "See the references in Table 2.\nReferences are listed below")]
    assert strip_references(pages) == (pages, None)


@pytest.mark.parametrize(
    "heading", ["REFERENCES", "  References  ", "Bibliography", "Literature Cited"]
)
def test_heading_variants(heading):
    _, cut = strip_references([(1, f"Body\n{heading}\nSmith 2020")])
    assert cut == 1


# --- PDF extraction and metadata ------------------------------------------------------


def test_extract_pages_is_one_based(tmp_path):
    pdf = _make_pdf(tmp_path / "t.pdf", ["first page", "second page"])
    pages = extract_pages(pdf)
    assert [p for p, _ in pages] == [1, 2]
    assert "second page" in pages[1][1]


def test_extract_pages_dehyphenates_line_breaks(tmp_path):
    pdf = _make_pdf(tmp_path / "t.pdf", ["dynamic vegeta-\ntion model"])
    text = normalize_whitespace(extract_pages(pdf)[0][1])
    assert "vegetation" in text


def test_chunk_pages_metadata_and_stable_ids():
    pages = [(1, WORDS), (2, "short page")]
    chunks = chunk_pages("smith2020", "A title", pages, size=500, overlap=50)

    assert {c.paper_id for c in chunks} == {"smith2020"}
    assert {c.title for c in chunks} == {"A title"}
    assert chunks[-1].page == 2 and chunks[-1].text == "short page"
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "smith2020-p1-c0"
    assert ids == [c.chunk_id for c in chunk_pages("smith2020", "A title", pages, 500, 50)]


def test_chunks_do_not_cross_pages():
    pages = [(1, "alpha " * 50), (2, "beta " * 50)]
    for chunk in chunk_pages("p", "T", pages, size=120, overlap=20):
        expected = "alpha" if chunk.page == 1 else "beta"
        assert set(chunk.text.split()) == {expected}


def test_ingest_pdf_end_to_end(tmp_path):
    pdf = _make_pdf(
        tmp_path / "t.pdf",
        ["Introduction text", "Results text\nReferences\nSmith, J. (2020)", "Jones (2019)"],
    )
    chunks = ingest_pdf("p", "T", pdf, size=1000, overlap=150)
    assert [(c.page, c.text) for c in chunks] == [(1, "Introduction text"), (2, "Results text")]
