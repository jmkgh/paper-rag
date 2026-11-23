"""Answer generation tests with a fake retriever and a fake LLM (no Ollama, no network)."""

import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_rag.answer import REFUSAL, answer_question, build_messages, parse_citations
from paper_rag.config import load_config
from paper_rag.retrieve import RetrievedChunk

LLM_CFG = load_config(Path(__file__).resolve().parents[1] / "config.yaml").llm

CHUNKS = [
    RetrievedChunk(
        "krause2025-p4-c0", "krause2025", "Herbivory", 4, "Grazers reduce biomass.", 0.8
    ),
    RetrievedChunk("harfoot2014-p2-c1", "harfoot2014", "Madingley", 2, "Cohorts of animals.", 0.7),
]


class FakeRetriever:
    def __init__(self, chunks=CHUNKS):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, question, top_k=None):
        self.calls.append((question, top_k))
        return self.chunks


class FakeLLM:
    """Records the call and returns a LiteLLM-shaped response."""

    def __init__(self, content: str, prompt_tokens: int = 300, completion_tokens: int = 20):
        self.content = content
        self.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=self.usage)


def _ask(llm, cfg=LLM_CFG, retriever=None):
    return answer_question(
        "How do grazers act?", retriever or FakeRetriever(), cfg, completion_fn=llm
    )


# --- prompt --------------------------------------------------------------------------


def test_messages_contain_rules_context_headers_and_question():
    system, user = build_messages("How do grazers act?", CHUNKS)
    assert system["role"] == "system" and REFUSAL in system["content"]
    assert "[paper_id, p. X]" in system["content"]
    assert "[krause2025, p. 4]\nGrazers reduce biomass." in user["content"]
    assert "[harfoot2014, p. 2]\nCohorts of animals." in user["content"]
    assert user["content"].endswith("Question: How do grazers act?")


# --- citation parsing ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Grazers reduce biomass [krause2025, p. 4].", [("krause2025", 4)]),
        (
            "A [krause2025, p. 4] and B [harfoot2014, p. 2].",
            [("krause2025", 4), ("harfoot2014", 2)],
        ),
        ("Both [krause2025, p. 4; harfoot2014, p. 2].", [("krause2025", 4), ("harfoot2014", 2)]),
        ("Range [krause2025, pp. 4-5].", [("krause2025", 4), ("krause2025", 5)]),
        ("Twice [krause2025, p. 4] [krause2025, p. 4].", [("krause2025", 4)]),
        ("No citation, just [1] and [see Fig. 2].", []),
    ],
)
def test_parse_citations(text, expected):
    assert parse_citations(text) == expected


# --- answer_question -----------------------------------------------------------------


def test_answer_with_valid_citation():
    llm = FakeLLM("Grazers reduce biomass [krause2025, p. 4].")
    result = _ask(llm)
    assert result.answer == "Grazers reduce biomass [krause2025, p. 4]."
    assert result.citations == [("krause2025", 4)]
    assert result.invalid_citations == []
    assert result.refused is False
    assert result.retrieved == CHUNKS
    assert (result.prompt_tokens, result.completion_tokens) == (300, 20)
    assert result.prompt_tokens_estimate and result.prompt_tokens_estimate > 0
    assert result.latency_s >= 0


def test_citation_outside_context_is_flagged():
    result = _ask(FakeLLM("Something [krause2025, p. 9] and [other2020, p. 1]."))
    assert result.invalid_citations == [("krause2025", 9), ("other2020", 1)]


def test_refusal_is_detected():
    result = _ask(FakeLLM("Not found in the sources."))
    assert result.refused is True
    assert result.citations == []


def test_ollama_call_gets_num_ctx_and_generation_settings():
    llm = FakeLLM("x")
    _ask(llm)
    assert llm.kwargs["model"] == LLM_CFG.model
    assert llm.kwargs["num_ctx"] == LLM_CFG.num_ctx
    assert llm.kwargs["temperature"] == LLM_CFG.temperature
    assert llm.kwargs["max_tokens"] == LLM_CFG.max_tokens
    assert llm.kwargs["messages"][0]["role"] == "system"


def test_api_model_gets_no_num_ctx():
    llm = FakeLLM("x")
    _ask(llm, cfg=replace(LLM_CFG, model="openai/gpt-4o-mini"))
    assert "num_ctx" not in llm.kwargs


def test_warns_when_context_exceeds_num_ctx(caplog):
    big = [replace(CHUNKS[0], text="word " * 3000)]
    with caplog.at_level(logging.WARNING):
        _ask(FakeLLM("x"), cfg=replace(LLM_CFG, num_ctx=1024), retriever=FakeRetriever(big))
    assert "exceeds num_ctx" in caplog.text


def test_warns_when_reported_prompt_tokens_suggest_truncation(caplog):
    llm = FakeLLM("x", prompt_tokens=LLM_CFG.num_ctx - LLM_CFG.max_tokens)
    with caplog.at_level(logging.WARNING):
        _ask(llm)
    assert "probably truncated" in caplog.text


def test_no_warning_for_normal_prompt(caplog):
    with caplog.at_level(logging.WARNING):
        _ask(FakeLLM("x"))
    assert caplog.text == ""
