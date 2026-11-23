"""Streamlit app tests via AppTest (headless, same process); retriever and LLM are mocked."""

from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from paper_rag.answer import Answer
from paper_rag.retrieve import RetrievedChunk

APP = str(Path(__file__).resolve().parents[1] / "app.py")

CHUNKS = [
    RetrievedChunk(
        "krause2025-p1-c0", "krause2025", "Herbivory", 1, "Boreal forests *shift*.", 0.83
    ),
    RetrievedChunk("krause2025-p14-c2", "krause2025", "Herbivory", 14, "Summary text.", 0.81),
]


class FakeRetriever:
    def __init__(self, *args, **kwargs):
        pass

    def retrieve(self, question, top_k=None):
        return CHUNKS[:top_k]


class MissingIndexRetriever:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Index not found; run `python -m paper_rag.index` first.")


def fake_answer(question, chunks, cfg):
    refused = "unknown" in question
    text = "not found in the sources" if refused else "Mixed forests [krause2025, p. 1]."
    citations = [] if refused else [("krause2025", 1)]
    return Answer(question, text, "fake-model", chunks, citations, [], refused, 2.0, 1000, 1000, 20)


@pytest.fixture(autouse=True)
def clear_streamlit_caches():
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def _run(retriever_cls=FakeRetriever, question: str | None = None) -> AppTest:
    with (
        patch("paper_rag.retrieve.Retriever", retriever_cls),
        patch("paper_rag.answer.generate_answer", fake_answer),
    ):
        at = AppTest.from_file(APP, default_timeout=60).run()
        if question is not None:
            at.text_area[0].input(question)
            at.button[0].click().run()
    return at


def test_start_page_shows_form():
    at = _run()
    assert not at.exception
    assert at.title[0].value == "paper-rag"
    assert len(at.text_area) == 1 and at.button[0].label == "Ask"


def test_question_shows_answer_and_marked_sources():
    at = _run(question="How do boreal forests change?")
    assert not at.exception
    assert any("Mixed forests [krause2025, p. 1]." in m.value for m in at.markdown)
    labels = [e.label for e in at.expander]
    assert len(labels) == 2
    assert labels[0].startswith("✓ [krause2025, p. 1]")
    assert labels[1].startswith("[krause2025, p. 14]")
    # raw chunk text is escaped, PDF link points to the page
    assert any(r"Boreal forests \*shift\*." in m.value for m in at.markdown)
    assert any("#page=14" in m.value for m in at.markdown)


def test_refusal_shows_info():
    at = _run(question="Something unknown?")
    assert not at.exception
    assert at.info[0].value.startswith("The model found no answer")


def test_missing_index_shows_error_instead_of_traceback():
    at = _run(retriever_cls=MissingIndexRetriever)
    assert not at.exception
    assert "paper_rag.index" in at.error[0].value
    assert len(at.text_area) == 0  # app stopped before the form
