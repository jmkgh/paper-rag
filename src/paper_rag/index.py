"""Embed chunks and store them in a persistent Chroma collection.

Usage:
    python -m paper_rag.index --config config.yaml

Every combination of embedding settings and chunking settings gets its own collection
(name = short hash of those settings), so index variants can be compared side by side
in the evaluation without overwriting each other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence

import chromadb
from chromadb.errors import NotFoundError

from paper_rag.config import Config, EmbeddingConfig, load_config
from paper_rag.ingest import Chunk, ingest_all

log = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]
BATCH_SIZE = 64


def ollama_embed_fn(model: str) -> EmbedFn:
    """Embedding function backed by the local Ollama server (batch endpoint)."""
    import ollama  # imported lazily so tests never need a running server

    client = ollama.Client()

    def embed(texts: list[str]) -> list[list[float]]:
        return [list(v) for v in client.embed(model=model, input=texts).embeddings]

    return embed


class Embedder:
    """Applies the task prefixes from the config and delegates to an embedding function.

    nomic-embed-text is trained with "search_document: " / "search_query: " prefixes;
    `use_prefixes` switches them off for comparison in the evaluation.
    """

    def __init__(self, cfg: EmbeddingConfig, embed_fn: EmbedFn | None = None):
        self.cfg = cfg
        self.embed_fn = embed_fn or ollama_embed_fn(cfg.model)

    def _prefix(self, prefix: str, texts: Sequence[str]) -> list[str]:
        return [prefix + t for t in texts] if self.cfg.use_prefixes else list(texts)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_fn(self._prefix(self.cfg.document_prefix, texts))

    def embed_query(self, text: str) -> list[float]:
        return self.embed_fn(self._prefix(self.cfg.query_prefix, [text]))[0]


def index_params(cfg: Config) -> dict:
    """All settings that change the content of the index."""
    return {
        "embedding_model": cfg.embedding.model,
        "use_prefixes": cfg.embedding.use_prefixes,
        "document_prefix": cfg.embedding.document_prefix if cfg.embedding.use_prefixes else "",
        "query_prefix": cfg.embedding.query_prefix if cfg.embedding.use_prefixes else "",
        "chunk_size": cfg.chunking.chunk_size,
        "chunk_overlap": cfg.chunking.chunk_overlap,
    }


def collection_name(cfg: Config) -> str:
    """Deterministic collection name derived from the index parameters."""
    payload = json.dumps(index_params(cfg), sort_keys=True).encode()
    return f"chunks-{hashlib.sha256(payload).hexdigest()[:8]}"


def get_client(cfg: Config) -> chromadb.ClientAPI:
    cfg.paths.index_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(cfg.paths.index_dir),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )


def build_index(
    cfg: Config,
    embedder: Embedder,
    chunks: list[Chunk] | None = None,
    client: chromadb.ClientAPI | None = None,
) -> chromadb.Collection:
    """(Re)build the collection for `cfg` from scratch.

    Rebuilding instead of updating incrementally keeps the index exactly in sync with
    papers.csv: removed papers or changed chunking leave no stale entries behind.
    """
    chunks = ingest_all(cfg) if chunks is None else chunks
    client = client or get_client(cfg)
    name = collection_name(cfg)
    try:
        client.delete_collection(name)
    except NotFoundError:
        pass
    collection = client.create_collection(
        name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata=index_params(cfg),
        embedding_function=None,  # we always pass embeddings ourselves
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embedder.embed_documents([c.text for c in batch]),
            documents=[c.text for c in batch],
            metadatas=[{"paper_id": c.paper_id, "title": c.title, "page": c.page} for c in batch],
        )
        log.info("embedded %d/%d chunks", start + len(batch), len(chunks))
    return collection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Chroma index from all PDFs.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Ollama request otherwise

    cfg = load_config(args.config)
    t0 = time.perf_counter()
    collection = build_index(cfg, Embedder(cfg.embedding))
    print(
        f"collection {collection.name}: {collection.count()} chunks "
        f"in {time.perf_counter() - t0:.1f} s ({json.dumps(index_params(cfg))})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
