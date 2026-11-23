"""Answer a question from retrieved chunks via LiteLLM, with citations and context checks.

Usage:
    python -m paper_rag.answer "How do herbivores affect vegetation?" --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import litellm

from paper_rag.config import LLMConfig, load_config
from paper_rag.index import Embedder
from paper_rag.retrieve import RetrievedChunk, Retriever

log = logging.getLogger(__name__)

REFUSAL = "not found in the sources"

SYSTEM_PROMPT = f"""You answer questions about scientific papers.
Rules:
1. Use ONLY the information in the provided context. Do not use prior knowledge.
2. Support every statement with a citation in the form [paper_id, p. X], using the
   paper_id and page shown in the header of the context passage you used.
3. If the context does not contain the answer, reply exactly: "{REFUSAL}".
4. Be concise."""

# "[krause2025, p. 4]", "[a, pp. 3-4; b, p. 7]"
BRACKET = re.compile(r"\[([^\[\]]+)\]")
CITATION = re.compile(
    r"^\s*([a-z0-9_-]+)\s*,\s*pp?\.\s*(\d+)(?:\s*[-–]\s*(\d+))?\s*$", re.IGNORECASE
)

CompletionFn = Callable[..., object]
Citation = tuple[str, int]  # (paper_id, page)


@dataclass
class Answer:
    question: str
    answer: str
    model: str
    retrieved: list[RetrievedChunk]
    citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[Citation] = field(default_factory=list)  # not in the context
    refused: bool = False
    latency_s: float = 0.0
    prompt_tokens_estimate: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    context = "\n\n".join(f"[{c.paper_id}, p. {c.page}]\n{c.text}" for c in chunks)
    user = f"Context:\n\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_citations(text: str) -> list[Citation]:
    """Extract (paper_id, page) pairs in order of appearance, without duplicates."""
    found: list[Citation] = []
    for bracket in BRACKET.findall(text):
        for part in bracket.split(";"):
            m = CITATION.match(part)
            if not m:
                continue
            first = int(m.group(2))
            last = int(m.group(3) or first)
            for page in range(first, max(first, last) + 1):
                if (m.group(1), page) not in found:
                    found.append((m.group(1), page))
    return found


def is_ollama(model: str) -> bool:
    return model.startswith(("ollama/", "ollama_chat/"))


def check_context_budget(messages: list[dict[str, str]], cfg: LLMConfig) -> int | None:
    """Estimate prompt tokens and warn if prompt + answer budget may exceed num_ctx.

    The estimate uses LiteLLM's token counter (a tiktoken approximation for local
    models), so it is a sanity check, not an exact count.
    """
    try:
        estimate = litellm.token_counter(model=cfg.model, messages=messages)
    except Exception as exc:  # the estimate must never break answering
        log.warning("could not estimate prompt tokens: %s", exc)
        return None
    if is_ollama(cfg.model) and estimate + cfg.max_tokens > cfg.num_ctx:
        log.warning(
            "prompt (~%d tokens) + max_tokens (%d) exceeds num_ctx (%d): "
            "Ollama will silently truncate the prompt",
            estimate,
            cfg.max_tokens,
            cfg.num_ctx,
        )
    return estimate


def answer_question(
    question: str,
    retriever: Retriever,
    cfg: LLMConfig,
    top_k: int | None = None,
    completion_fn: CompletionFn = litellm.completion,
) -> Answer:
    """Retrieve context for `question` and generate a cited answer."""
    chunks = retriever.retrieve(question, top_k)
    return generate_answer(question, chunks, cfg, completion_fn)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    cfg: LLMConfig,
    completion_fn: CompletionFn = litellm.completion,
) -> Answer:
    """Generate an answer from already retrieved chunks (used directly by the evaluation)."""
    messages = build_messages(question, chunks)
    estimate = check_context_budget(messages, cfg)

    kwargs: dict = {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
    if is_ollama(cfg.model):
        kwargs["num_ctx"] = cfg.num_ctx  # Ollama's default (4096 here) truncates silently

    t0 = time.perf_counter()
    response = completion_fn(model=cfg.model, messages=messages, **kwargs)
    latency = time.perf_counter() - t0

    text = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if is_ollama(cfg.model) and prompt_tokens and prompt_tokens >= cfg.num_ctx - cfg.max_tokens:
        log.warning(
            "prompt used %d of num_ctx=%d tokens: context was probably truncated",
            prompt_tokens,
            cfg.num_ctx,
        )
    log.info("prompt tokens: estimate %s, reported %s", estimate, prompt_tokens)

    citations = parse_citations(text)
    in_context = {(c.paper_id, c.page) for c in chunks}
    return Answer(
        question=question,
        answer=text,
        model=cfg.model,
        retrieved=chunks,
        citations=citations,
        invalid_citations=[c for c in citations if c not in in_context],
        refused=REFUSAL in text.lower(),
        latency_s=latency,
        prompt_tokens_estimate=estimate,
        prompt_tokens=prompt_tokens,
        completion_tokens=getattr(usage, "completion_tokens", None),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer a question from the indexed papers.")
    parser.add_argument("question")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "LiteLLM"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(args.config)
    retriever = Retriever(cfg, Embedder(cfg.embedding))
    result = answer_question(args.question, retriever, cfg.llm, args.top_k)

    print(f"\n{result.answer}\n")
    print(f"model: {result.model} | {result.latency_s:.1f} s | refused: {result.refused}")
    print(
        f"tokens: prompt {result.prompt_tokens} (est. {result.prompt_tokens_estimate}), "
        f"completion {result.completion_tokens}"
    )
    print(f"citations: {result.citations} | not in context: {result.invalid_citations}")
    print(
        "retrieved:",
        ", ".join(f"[{c.paper_id}, p. {c.page}] {c.score:.2f}" for c in result.retrieved),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
