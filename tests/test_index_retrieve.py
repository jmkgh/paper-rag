"""Index + retrieval tests with a fake, deterministic embedding function (no Ollama)."""

import hashlib
import math
from dataclasses import replace
from pathlib import Path

import pytest

from paper_rag import index as index_mod
from paper_rag.config import load_config
from paper_rag.index import Embedder, build_index, collection_name, get_client
from paper_rag.ingest import Chunk
from paper_rag.retrieve import Retriever

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
DIM = 64


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Hashed bag of words: texts sharing words get similar vectors."""
    vectors = []
    for text in texts:
        vec = [0.0] * DIM
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


class RecordingEmbed:
    """Wraps fake_embed and records every call (texts as sent to the model)."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return fake_embed(texts)


CHUNKS = [
    Chunk(
        "grazer2020", "Grazing", 3, "grazer2020-p3-c0", "herbivores graze grass and reduce biomass"
    ),
    Chunk("fire2019", "Fire", 5, "fire2019-p5-c0", "wildfire burns dry fuel in savanna ecosystems"),
    Chunk("ocean2018", "Ocean", 2, "ocean2018-p2-c0", "marine plankton drives ocean productivity"),
    Chunk("ocean2018", "Ocean", 4, "ocean2018-p4-c1", "fish predation on zooplankton in the sea"),
]


@pytest.fixture
def cfg(tmp_path):
    base = load_config(REPO_CONFIG)
    return replace(base, paths=replace(base.paths, index_dir=tmp_path / "index"))


@pytest.fixture
def client(cfg):
    return get_client(cfg)


# --- Embedder ------------------------------------------------------------------------


def test_prefixes_applied_when_enabled(cfg):
    rec = RecordingEmbed()
    emb = Embedder(cfg.embedding, rec)
    emb.embed_documents(["doc"])
    emb.embed_query("question")
    assert rec.calls == [["search_document: doc"], ["search_query: question"]]


def test_prefixes_omitted_when_disabled(cfg):
    rec = RecordingEmbed()
    emb = Embedder(replace(cfg.embedding, use_prefixes=False), rec)
    emb.embed_documents(["doc"])
    emb.embed_query("question")
    assert rec.calls == [["doc"], ["question"]]


# --- collection naming ---------------------------------------------------------------


def test_collection_name_is_deterministic_and_valid(cfg):
    name = collection_name(cfg)
    assert name == collection_name(cfg)
    assert name.startswith("chunks-") and len(name) == len("chunks-") + 8


@pytest.mark.parametrize(
    "change",
    [
        lambda c: replace(c, embedding=replace(c.embedding, use_prefixes=False)),
        lambda c: replace(c, embedding=replace(c.embedding, model="other-embed")),
        lambda c: replace(c, chunking=replace(c.chunking, chunk_size=500)),
        lambda c: replace(c, chunking=replace(c.chunking, chunk_overlap=0)),
    ],
)
def test_index_relevant_settings_change_collection_name(cfg, change):
    assert collection_name(change(cfg)) != collection_name(cfg)


def test_llm_and_top_k_do_not_change_collection_name(cfg):
    other = replace(cfg, llm=replace(cfg.llm, model="ollama/gemma3:12b"))
    other = replace(other, retrieval=replace(other.retrieval, top_k=10))
    assert collection_name(other) == collection_name(cfg)


# --- build_index ---------------------------------------------------------------------


def test_build_index_stores_chunks_with_metadata(cfg, client):
    collection = build_index(cfg, Embedder(cfg.embedding, fake_embed), CHUNKS, client)
    assert collection.count() == len(CHUNKS)
    got = collection.get(ids=["fire2019-p5-c0"], include=["documents", "metadatas"])
    assert got["documents"] == ["wildfire burns dry fuel in savanna ecosystems"]
    assert got["metadatas"] == [{"paper_id": "fire2019", "title": "Fire", "page": 5}]
    assert collection.metadata["use_prefixes"] is True


def test_build_index_embeds_in_batches(cfg, client, monkeypatch):
    monkeypatch.setattr(index_mod, "BATCH_SIZE", 3)
    rec = RecordingEmbed()
    build_index(cfg, Embedder(cfg.embedding, rec), CHUNKS, client)
    assert [len(c) for c in rec.calls] == [3, 1]
    assert all(t.startswith("search_document: ") for call in rec.calls for t in call)


def test_rebuild_replaces_instead_of_duplicating(cfg, client):
    emb = Embedder(cfg.embedding, fake_embed)
    build_index(cfg, emb, CHUNKS, client)
    collection = build_index(cfg, emb, CHUNKS[:2], client)
    assert collection.count() == 2


# --- Retriever -----------------------------------------------------------------------


def test_retrieve_ranks_matching_chunk_first(cfg, client):
    emb = Embedder(cfg.embedding, fake_embed)
    build_index(cfg, emb, CHUNKS, client)

    hits = Retriever(cfg, emb, client).retrieve("how do herbivores graze grass", top_k=2)

    assert len(hits) == 2
    best = hits[0]
    assert (best.paper_id, best.page, best.chunk_id) == ("grazer2020", 3, "grazer2020-p3-c0")
    assert best.title == "Grazing"
    assert isinstance(best.page, int)
    assert hits[0].score >= hits[1].score
    assert -1.0 <= best.score <= 1.0


def test_retrieve_uses_config_top_k_by_default(cfg, client):
    cfg = replace(cfg, retrieval=replace(cfg.retrieval, top_k=3))
    emb = Embedder(cfg.embedding, fake_embed)
    build_index(cfg, emb, CHUNKS, client)
    assert len(Retriever(cfg, emb, client).retrieve("ocean")) == 3


def test_missing_index_gives_clear_error(cfg, client):
    with pytest.raises(RuntimeError, match="paper_rag.index"):
        Retriever(cfg, Embedder(cfg.embedding, fake_embed), client)
