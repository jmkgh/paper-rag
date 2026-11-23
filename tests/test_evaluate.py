"""Evaluation tests: metrics by hand, question validation, a full run with fakes."""

import csv
from pathlib import Path

import pytest
import yaml

from paper_rag.answer import Answer
from paper_rag.config import load_config
from paper_rag.evaluate import (
    Question,
    load_questions,
    page_rank,
    paper_rank,
    retrieval_metrics,
    run_evaluation,
    summarize_ratings,
)
from paper_rag.retrieve import RetrievedChunk

REPO_ROOT = Path(__file__).resolve().parents[1]


def hit(paper: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(f"{paper}-p{page}-c0", paper, "T", page, "text", 0.5)


HITS = [hit("a", 1), hit("b", 4), hit("b", 2), hit("c", 9), hit("a", 3)]


# --- ranks and metrics ---------------------------------------------------------------


def test_paper_and_page_rank():
    q = Question("q", "?", "b", [2])
    assert paper_rank(q, HITS) == 2  # first "b" hit
    assert page_rank(q, HITS) == 3  # first "b" hit on page 2


def test_rank_none_when_missing():
    q = Question("q", "?", "z", [1])
    assert paper_rank(q, HITS) is None
    assert page_rank(q, HITS) is None


def test_retrieval_metrics_by_hand():
    # ranks 1, 2, 4, missing -> hit@1 = 1/4, hit@3 = 2/4, hit@5 = 3/4
    # MRR = (1 + 1/2 + 1/4 + 0) / 4 = 0.4375
    m = retrieval_metrics([1, 2, 4, None])
    assert m == {"hit@1": 0.25, "hit@3": 0.5, "hit@5": 0.75, "MRR": 0.4375}


def test_retrieval_metrics_ignores_ranks_beyond_max_k():
    assert retrieval_metrics([7]) == {"hit@1": 0, "hit@3": 0, "hit@5": 0, "MRR": 0}


# --- question validation -------------------------------------------------------------


def _questions_file(tmp_path, items) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(yaml.safe_dump({"questions": items}), encoding="utf-8")
    return path


def test_repo_questions_are_valid():
    cfg = load_config(REPO_ROOT / "config.yaml")
    known = {row["id"] for row in csv.DictReader(cfg.paths.papers_csv.open(encoding="utf-8"))}
    questions = load_questions(cfg.paths.questions, known)
    assert len(questions) >= 10
    assert any(not q.answerable for q in questions)


def test_unknown_paper_is_rejected(tmp_path):
    path = _questions_file(
        tmp_path, [{"id": "q1", "question": "?", "expected_paper": "x", "expected_pages": [1]}]
    )
    with pytest.raises(ValueError, match="unknown paper"):
        load_questions(path, {"a"})


def test_duplicate_id_is_rejected(tmp_path):
    item = {"id": "q1", "question": "?", "expected_paper": "a", "expected_pages": [1]}
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(_questions_file(tmp_path, [item, item]), {"a"})


def test_answerable_question_needs_pages(tmp_path):
    path = _questions_file(tmp_path, [{"id": "q1", "question": "?", "expected_paper": "a"}])
    with pytest.raises(ValueError, match="expected_pages"):
        load_questions(path, {"a"})


def test_unanswerable_question_needs_no_pages(tmp_path):
    path = _questions_file(tmp_path, [{"id": "u1", "question": "?", "expected_paper": None}])
    (q,) = load_questions(path, {"a"})
    assert not q.answerable and q.expected_pages == []


# --- full run ------------------------------------------------------------------------


class FakeRetriever:
    def retrieve(self, question, top_k=None):
        return HITS[:top_k]


def fake_answer(question, chunks):
    if "unknown" in question:
        text = "not found in the sources"
        citations, refused = [], True
    else:
        text = "Answer [b, p. 2] [zz, p. 1]"
        citations, refused = [("b", 2), ("zz", 1)], False
    return Answer(
        question=question,
        answer=text,
        model="fake",
        retrieved=chunks,
        citations=citations,
        invalid_citations=[c for c in citations if c[0] == "zz"],
        refused=refused,
        latency_s=1.5,
        prompt_tokens=100,
        completion_tokens=10,
    )


QUESTIONS = [
    Question("q1", "about b", "b", [2]),
    Question("q2", "about c", "c", [1]),  # paper at rank 4, page never
    Question("u1", "unknown topic", None, []),
]


@pytest.fixture
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


def test_full_run_writes_all_files(tmp_path, cfg):
    metrics = run_evaluation(cfg, QUESTIONS, FakeRetriever(), tmp_path, fake_answer)

    assert {p.name for p in tmp_path.iterdir()} == {
        "config.yaml",
        "retrieval.csv",
        "answers.csv",
        "summary.md",
    }
    # q1: paper rank 2, page rank 3; q2: paper rank 4, page missing
    assert metrics["paper"] == {
        "hit@1": 0.0,
        "hit@3": 0.5,
        "hit@5": 1.0,
        "MRR": (1 / 2 + 1 / 4) / 2,
    }
    assert metrics["page"]["hit@3"] == 0.5 and metrics["page"]["hit@5"] == 0.5

    answers = list(csv.DictReader((tmp_path / "answers.csv").open(encoding="utf-8")))
    assert [a["id"] for a in answers] == ["q1", "q2", "u1"]
    assert answers[0]["cites_expected_page"] == "True"
    assert answers[0]["invalid_citations"] == "zz:1"
    assert answers[2]["refused"] == "True" and answers[2]["expected_refusal"] == "True"
    assert all(a["rating"] == "" for a in answers)

    snapshot = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert snapshot["llm"]["model"] == cfg.llm.model
    assert snapshot["_run"]["index_collection"].startswith("chunks-")

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| paper | 0.00 | 0.50 | 1.00 | 0.38 |" in summary
    assert "Refused when answer not in collection: 1/1" in summary


def test_retrieval_only_run_skips_answers(tmp_path, cfg):
    run_evaluation(cfg, QUESTIONS, FakeRetriever(), tmp_path, answer_fn=None)
    assert not (tmp_path / "answers.csv").exists()
    assert (tmp_path / "retrieval.csv").exists()


def _rate(run_dir: Path, ratings: list[str]) -> None:
    path = run_dir / "answers.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row, rating in zip(rows, ratings, strict=True):
        row["rating"] = rating
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_counts_ratings_and_is_idempotent(tmp_path, cfg):
    run_evaluation(cfg, QUESTIONS, FakeRetriever(), tmp_path, fake_answer)
    _rate(tmp_path, ["correct", "partial", "correctly_refused"])

    counts = summarize_ratings(tmp_path)
    summarize_ratings(tmp_path)  # running twice must not duplicate the table

    assert counts == {"correct": 1, "partial": 1, "correctly_refused": 1}
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert summary.count("## Answers (manual rating)") == 1
    assert "| partial | 1 |" in summary and "| wrong | 0 |" in summary


def test_summarize_rejects_missing_or_invalid_ratings(tmp_path, cfg):
    run_evaluation(cfg, QUESTIONS, FakeRetriever(), tmp_path, fake_answer)
    _rate(tmp_path, ["correct", "", "great"])
    with pytest.raises(ValueError, match="q2.*u1|u1.*q2"):
        summarize_ratings(tmp_path)
