# Evaluation run

- LLM: `ollama_chat/qwen2.5:7b` (num_ctx 8192, temperature 0.0)
- Embeddings: `nomic-embed-text`, prefixes: False
- Chunks: 1000 chars, overlap 150; top_k for answering: 5
- Questions: 18 (15 answerable, 3 not in collection)

## Retrieval (answerable questions, top 5)

| level | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| paper | 0.93 | 1.00 | 1.00 | 0.97 |
| page | 0.47 | 0.80 | 0.87 | 0.62 |
