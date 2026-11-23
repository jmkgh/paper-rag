"""Streamlit UI: ask a question, get a cited answer and inspect the retrieved sources.

Run:
    uv run streamlit run app.py
    PAPER_RAG_CONFIG=other.yaml uv run streamlit run app.py   # different config

The app is a thin layer: all logic lives in the paper_rag package.
"""

from __future__ import annotations

import os
import re

import streamlit as st

from paper_rag.answer import generate_answer
from paper_rag.config import Config, load_config
from paper_rag.download import read_papers
from paper_rag.index import Embedder
from paper_rag.retrieve import Retriever

CONFIG_PATH = os.environ.get("PAPER_RAG_CONFIG", "config.yaml")


@st.cache_resource
def get_config(path: str) -> Config:
    return load_config(path)


@st.cache_resource
def get_retriever(path: str) -> Retriever:
    """Open the Chroma collection once per process instead of on every rerun."""
    cfg = get_config(path)
    return Retriever(cfg, Embedder(cfg.embedding))


@st.cache_data
def get_paper_urls(path: str) -> dict[str, str]:
    return {p.id: p.url for p in read_papers(get_config(path).paths.papers_csv)}


def escape_markdown(text: str) -> str:
    """Show raw PDF text literally (no accidental emphasis, links or LaTeX from `$`)."""
    return re.sub(r"([\\`*_\[\]#$<>|~])", r"\\\1", text)


st.set_page_config(page_title="paper-rag", layout="centered")
cfg = get_config(CONFIG_PATH)
urls = get_paper_urls(CONFIG_PATH)

with st.sidebar:
    st.header("Configuration")
    st.markdown(
        f"- **LLM:** `{cfg.llm.model}`\n"
        f"- **num_ctx:** {cfg.llm.num_ctx}\n"
        f"- **Embeddings:** `{cfg.embedding.model}` (prefixes: {cfg.embedding.use_prefixes})\n"
        f"- **Chunks:** {cfg.chunking.chunk_size} chars, overlap {cfg.chunking.chunk_overlap}"
    )
    top_k = st.slider("Retrieved chunks (top_k)", 1, 10, cfg.retrieval.top_k)
    st.caption("Change models and chunking in config.yaml.")

st.title("paper-rag")
st.caption(
    f"Ask questions about {len(urls)} open-access papers. Answers are generated only from "
    "the retrieved passages and cite them as [paper_id, p. X]."
)

try:
    retriever = get_retriever(CONFIG_PATH)
except RuntimeError as exc:  # index for this config not built yet
    st.error(str(exc))
    st.stop()

with st.form("ask"):
    question = st.text_area("Question", height=80)
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Retrieving passages and generating the answer ..."):
        try:
            chunks = retriever.retrieve(question.strip(), top_k)
            st.session_state["result"] = generate_answer(question.strip(), chunks, cfg.llm)
        except Exception as exc:  # connection errors from Ollama or the LLM provider
            st.error(
                f"Could not answer the question: {exc}\n\n"
                "Is Ollama running (`ollama serve`) and are the models pulled?"
            )
            st.stop()

# Kept in session state so the answer survives reruns (e.g. moving the slider).
result = st.session_state.get("result")
if result is not None:
    st.subheader("Answer")
    if result.refused:
        st.info("The model found no answer in the retrieved passages.")
    st.markdown(result.answer)
    st.caption(
        f"{result.latency_s:.1f} s · {result.model} · "
        f"{result.prompt_tokens or '?'} prompt / {result.completion_tokens or '?'} answer tokens"
    )
    if result.invalid_citations:
        cited = ", ".join(f"[{p}, p. {page}]" for p, page in result.invalid_citations)
        st.warning(f"Cited but not among the retrieved passages: {cited}")

    st.subheader(f"Sources (top {len(result.retrieved)})")
    st.caption("✓ = cited in the answer")
    cited_pages = set(result.citations)
    for chunk in result.retrieved:
        mark = "✓ " if (chunk.paper_id, chunk.page) in cited_pages else ""
        label = f"{mark}[{chunk.paper_id}, p. {chunk.page}] {chunk.title} · score {chunk.score:.2f}"
        with st.expander(label):
            st.markdown(escape_markdown(chunk.text))
            if chunk.paper_id in urls:
                st.markdown(
                    f"[Open PDF at p. {chunk.page}]({urls[chunk.paper_id]}#page={chunk.page})"
                )
