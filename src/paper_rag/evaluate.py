"""Evaluate retrieval and answers on the questions in `eval/questions.yaml`.

Usage:
    python -m paper_rag.evaluate --config config.yaml [--label baseline] [--retrieval-only]
    python -m paper_rag.evaluate --summarize eval/results/<run>   # after manual rating

Each run writes to `eval/results/<timestamp>_<label>/`:
    config.yaml     the full configuration plus index name and git commit
    retrieval.csv   per question: rank of the expected paper / page, top-5 hits
    answers.csv     per question: answer, citations, refusal, latency, empty `rating` column
    summary.md      metrics table (and rating counts after --summarize)

Retrieval metrics (hit@k, MRR) use the top MAX_K ranks and only answerable questions;
questions without an expected paper only test whether the model refuses.
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from paper_rag.answer import Answer, generate_answer
from paper_rag.config import Config, load_config
from paper_rag.download import read_papers
from paper_rag.index import Embedder, collection_name, index_params
from paper_rag.retrieve import RetrievedChunk, Retriever

log = logging.getLogger(__name__)

KS = (1, 3, 5)
MAX_K = max(KS)
RATINGS = ("correct", "partial", "wrong", "correctly_refused", "false_refusal")


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expected_paper: str | None
    expected_pages: list[int]
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return self.expected_paper is not None


# --- questions -----------------------------------------------------------------------


def load_questions(path: Path, known_papers: set[str]) -> list[Question]:
    """Read and validate the question file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions: list[Question] = []
    seen: set[str] = set()
    for item in data.get("questions") or []:
        missing = {"id", "question", "expected_paper"} - set(item)
        if missing:
            raise ValueError(f"question {item.get('id', '?')}: missing field(s) {sorted(missing)}")
        q = Question(
            id=str(item["id"]),
            question=item["question"],
            expected_paper=item["expected_paper"],
            expected_pages=[int(p) for p in item.get("expected_pages") or []],
            notes=item.get("notes") or "",
        )
        if q.id in seen:
            raise ValueError(f"duplicate question id {q.id!r}")
        if q.answerable and q.expected_paper not in known_papers:
            raise ValueError(f"question {q.id}: unknown paper {q.expected_paper!r}")
        if q.answerable and not q.expected_pages:
            raise ValueError(f"question {q.id}: expected_pages required for answerable questions")
        seen.add(q.id)
        questions.append(q)
    if not questions:
        raise ValueError(f"{path}: no questions found")
    return questions


# --- metrics -------------------------------------------------------------------------


def first_rank(hits: list[RetrievedChunk], match: Callable[[RetrievedChunk], bool]) -> int | None:
    """1-based rank of the first matching hit, or None."""
    return next((rank for rank, hit in enumerate(hits, start=1) if match(hit)), None)


def paper_rank(q: Question, hits: list[RetrievedChunk]) -> int | None:
    return first_rank(hits, lambda h: h.paper_id == q.expected_paper)


def page_rank(q: Question, hits: list[RetrievedChunk]) -> int | None:
    return first_rank(hits, lambda h: h.paper_id == q.expected_paper and h.page in q.expected_pages)


def retrieval_metrics(ranks: list[int | None]) -> dict[str, float]:
    """hit@k for k in KS and MRR@MAX_K over a list of ranks (None = not found)."""
    n = len(ranks)
    if n == 0:
        return {}
    metrics = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in KS}
    metrics["MRR"] = sum(1 / r for r in ranks if r is not None and r <= MAX_K) / n
    return metrics


# --- run -----------------------------------------------------------------------------


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fmt_hits(hits: list[RetrievedChunk]) -> str:
    return " ".join(f"{h.paper_id}:{h.page}" for h in hits)


def _fmt_citations(citations: list[tuple[str, int]]) -> str:
    return " ".join(f"{p}:{page}" for p, page in citations)


def run_evaluation(
    cfg: Config,
    questions: list[Question],
    retriever: Retriever,
    out_dir: Path,
    answer_fn: Callable[[str, list[RetrievedChunk]], Answer] | None,
) -> dict[str, dict[str, float]]:
    """Evaluate all questions and write the run directory. `answer_fn=None` skips the LLM."""
    out_dir.mkdir(parents=True, exist_ok=True)
    top_k = cfg.retrieval.top_k
    retrieval_rows, answer_rows = [], []
    paper_ranks, page_ranks = [], []

    for q in questions:
        hits = retriever.retrieve(q.question, max(MAX_K, top_k))
        row = {"id": q.id, "question": q.question, "expected_paper": q.expected_paper or ""}
        row["expected_pages"] = " ".join(map(str, q.expected_pages))
        if q.answerable:
            pr, pg = paper_rank(q, hits), page_rank(q, hits)
            paper_ranks.append(pr)
            page_ranks.append(pg)
            row.update(paper_rank=pr or "", page_rank=pg or "")
        else:
            row.update(paper_rank="", page_rank="")
        row["top_hits"] = _fmt_hits(hits[:MAX_K])
        row["top_scores"] = " ".join(f"{h.score:.3f}" for h in hits[:MAX_K])
        retrieval_rows.append(row)

        if answer_fn is not None:
            log.info("answering %s", q.id)
            a = answer_fn(q.question, hits[:top_k])
            expected = {(q.expected_paper, p) for p in q.expected_pages}
            answer_rows.append(
                {
                    "id": q.id,
                    "question": q.question,
                    "expected_paper": q.expected_paper or "",
                    "expected_pages": " ".join(map(str, q.expected_pages)),
                    "expected_refusal": not q.answerable,
                    "notes": q.notes,
                    "answer": a.answer,
                    "citations": _fmt_citations(a.citations),
                    "invalid_citations": _fmt_citations(a.invalid_citations),
                    "cites_expected_page": bool(expected & set(a.citations)),
                    "refused": a.refused,
                    "latency_s": f"{a.latency_s:.1f}",
                    "prompt_tokens": a.prompt_tokens or "",
                    "completion_tokens": a.completion_tokens or "",
                    "context": _fmt_hits(a.retrieved),
                    "rating": "",
                }
            )

    metrics = {"paper": retrieval_metrics(paper_ranks), "page": retrieval_metrics(page_ranks)}
    _write_csv(out_dir / "retrieval.csv", retrieval_rows)
    if answer_rows:
        _write_csv(out_dir / "answers.csv", answer_rows)
    snapshot = {
        **cfg.raw,
        "_run": {
            "index_collection": collection_name(cfg),
            "index_params": index_params(cfg),
            "git_commit": git_commit(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n_questions": len(questions),
            "retrieval_only": answer_fn is None,
        },
    }
    (out_dir / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), "utf-8")
    (out_dir / "summary.md").write_text(
        render_summary(cfg, questions, metrics, answer_rows), encoding="utf-8"
    )
    return metrics


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_summary(
    cfg: Config,
    questions: list[Question],
    metrics: dict[str, dict[str, float]],
    answer_rows: list[dict],
) -> str:
    n_ans = sum(q.answerable for q in questions)
    cols = [f"hit@{k}" for k in KS] + ["MRR"]
    lines = [
        "# Evaluation run",
        "",
        f"- LLM: `{cfg.llm.model}` (num_ctx {cfg.llm.num_ctx}, temperature {cfg.llm.temperature})",
        f"- Embeddings: `{cfg.embedding.model}`, prefixes: {cfg.embedding.use_prefixes}",
        f"- Chunks: {cfg.chunking.chunk_size} chars, overlap {cfg.chunking.chunk_overlap}; "
        f"top_k for answering: {cfg.retrieval.top_k}",
        f"- Questions: {len(questions)} ({n_ans} answerable, "
        f"{len(questions) - n_ans} not in collection)",
        "",
        f"## Retrieval (answerable questions, top {MAX_K})",
        "",
        "| level | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for level in ("paper", "page"):
        values = " | ".join(f"{metrics[level].get(c, 0):.2f}" for c in cols)
        lines.append(f"| {level} | {values} |")

    if answer_rows:
        latencies = [float(r["latency_s"]) for r in answer_rows]
        refused_ok = sum(r["refused"] for r in answer_rows if r["expected_refusal"])
        n_unans = sum(r["expected_refusal"] for r in answer_rows)
        refused_wrong = sum(r["refused"] for r in answer_rows if not r["expected_refusal"])
        invalid = sum(bool(r["invalid_citations"]) for r in answer_rows)
        cites_exp = sum(r["cites_expected_page"] for r in answer_rows if not r["expected_refusal"])
        lines += [
            "",
            "## Answers (automatic checks)",
            "",
            f"- Refused when answer not in collection: {refused_ok}/{n_unans}",
            f"- Refused although answerable: {refused_wrong}/{n_ans}",
            f"- Cited an expected page: {cites_exp}/{n_ans}",
            f"- Answers with citations outside the context: {invalid}/{len(answer_rows)}",
            f"- Latency: median {sorted(latencies)[len(latencies) // 2]:.1f} s, "
            f"max {max(latencies):.1f} s",
            "",
            "Manual ratings: fill the `rating` column in `answers.csv`, then run "
            "`python -m paper_rag.evaluate --summarize <run dir>`.",
        ]
    return "\n".join(lines) + "\n"


# --- manual ratings ------------------------------------------------------------------

RATINGS_HEADER = "## Answers (manual rating)"


def summarize_ratings(run_dir: Path) -> Counter:
    """Count manual ratings in answers.csv and append them to summary.md."""
    with (run_dir / "answers.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ratings = [r["rating"].strip() for r in rows]
    invalid = {r["id"]: v for r, v in zip(rows, ratings, strict=True) if v not in RATINGS}
    if invalid:
        raise ValueError(f"missing or invalid ratings (allowed: {', '.join(RATINGS)}): {invalid}")

    counts = Counter(ratings)
    summary_path = run_dir / "summary.md"
    summary = summary_path.read_text(encoding="utf-8").split(RATINGS_HEADER)[0].rstrip()
    table = ["", "", RATINGS_HEADER, "", "| rating | count |", "|---|---|"]
    table += [f"| {r} | {counts.get(r, 0)} |" for r in RATINGS]
    summary_path.write_text(summary + "\n".join(table) + "\n", encoding="utf-8")
    return counts


# --- CLI -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--label", default="run", help="suffix for the run directory")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM")
    parser.add_argument("--summarize", type=Path, help="count manual ratings of a finished run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.summarize:
        print(dict(summarize_ratings(args.summarize)))
        return 0

    cfg = load_config(args.config)
    known = {p.id for p in read_papers(cfg.paths.papers_csv)}
    questions = load_questions(cfg.paths.questions, known)
    retriever = Retriever(cfg, Embedder(cfg.embedding))
    answer_fn = None if args.retrieval_only else (lambda q, c: generate_answer(q, c, cfg.llm))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = cfg.paths.results_dir / f"{stamp}_{args.label}"
    run_evaluation(cfg, questions, retriever, out_dir, answer_fn)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"results written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
