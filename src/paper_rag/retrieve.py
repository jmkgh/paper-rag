"""Top-k vector search over the index built by `paper_rag.index`.

Usage:
    python -m paper_rag.retrieve "How is grazing represented?" --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import chromadb
from chromadb.errors import NotFoundError

from paper_rag.config import Config, load_config
from paper_rag.index import Embedder, collection_name, get_client


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    paper_id: str
    title: str
    page: int
    text: str
    score: float  # cosine similarity, 1 = identical direction


class Retriever:
    def __init__(self, cfg: Config, embedder: Embedder, client: chromadb.ClientAPI | None = None):
        self.cfg = cfg
        self.embedder = embedder
        name = collection_name(cfg)
        try:
            self.collection = (client or get_client(cfg)).get_collection(name)
        except NotFoundError as exc:
            raise RuntimeError(
                f"Index '{name}' for the current config not found in {cfg.paths.index_dir}; "
                "run `python -m paper_rag.index` first."
            ) from exc

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the `top_k` most similar chunks, best first."""
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(question)],
            n_results=top_k or self.cfg.retrieval.top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                paper_id=meta["paper_id"],
                title=meta["title"],
                page=int(meta["page"]),
                text=doc,
                score=1.0 - dist,  # Chroma returns cosine *distance*
            )
            for chunk_id, doc, meta, dist in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show the top-k chunks for a question.")
    parser.add_argument("question")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    retriever = Retriever(cfg, Embedder(cfg.embedding))
    for rank, hit in enumerate(retriever.retrieve(args.question, args.top_k), start=1):
        print(f"{rank}. [{hit.paper_id}, p. {hit.page}] score={hit.score:.3f} ({hit.chunk_id})")
        print(f"   {hit.text[:200]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
