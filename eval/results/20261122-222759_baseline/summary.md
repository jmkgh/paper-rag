# Evaluation run

- LLM: `ollama_chat/qwen2.5:7b` (num_ctx 8192, temperature 0.0)
- Embeddings: `nomic-embed-text`, prefixes: True
- Chunks: 1000 chars, overlap 150; top_k for answering: 5
- Questions: 18 (15 answerable, 3 not in collection)

## Retrieval (answerable questions, top 5)

| level | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| paper | 0.93 | 1.00 | 1.00 | 0.97 |
| page | 0.67 | 0.87 | 0.87 | 0.76 |

## Answers (automatic checks)

- Refused when answer not in collection: 3/3
- Refused although answerable: 3/15
- Cited an expected page: 11/15
- Answers with citations outside the context: 0/18
- Latency: median 13.6 s, max 24.4 s

Manual ratings: fill the `rating` column in `answers.csv`, then run `python -m paper_rag.evaluate --summarize <run dir>`.

## Answers (manual rating)

| rating | count |
|---|---|
| correct | 11 |
| partial | 1 |
| wrong | 0 |
| correctly_refused | 3 |
| false_refusal | 3 |
