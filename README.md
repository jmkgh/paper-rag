# paper-rag
[![CI](https://github.com/jmkgh/paper-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/jmkgh/paper-rag/actions/workflows/ci.yml)

A small, fully local RAG (retrieval-augmented generation) assistant that answers
questions about a collection of open-access papers on ecological modelling and
theory. Every answer cites paper and page as `[paper_id, p. X]`, or the assistant
replies "not found in the sources".

The focus is on **transparent evaluation**: how often does retrieval find the right
passage, and how good are the answers? Retrieval and answer quality are measured
separately, every run stores its configuration, and the results are part of the
repository.

Built without RAG frameworks (no LangChain, no LlamaIndex): each step is a short,
readable module that can be tested and explained on its own.

## Architecture

```
 A) once: build the index
 papers.csv ──download──► PDFs ──ingest──► chunks ──index──► Chroma collection
 (DOIs, URLs,             data/pdfs/     (≤1000 chars,      (embedding + text +
  licenses)                               one page each)     paper/page metadata)

 B) per question
 question ──retrieve──► top-k chunks ──answer──► local LLM ──► answer with [paper_id, p. X]
          (embed question,             (prompt: rules +          + citation checks
           cosine similarity)           chunks + question)

 evaluate: runs B for every question in eval/questions.yaml and logs metrics
 app.py:   Streamlit UI for B with expandable sources and PDF page links
```

| Module | Responsibility |
|---|---|
| `config.py` | Load `config.yaml` into typed, immutable dataclasses; unknown or missing keys fail early |
| `download.py` | Download PDFs listed in `data/papers.csv`, check the `%PDF` header, write atomically |
| `ingest.py` | PDF → page texts (pymupdf) → strip bibliography → chunks that never cross a page boundary |
| `index.py` | Embed chunks with `nomic-embed-text` via Ollama and store them in a persistent Chroma collection |
| `retrieve.py` | Embed the question and return the top-k most similar chunks |
| `answer.py` | Build the prompt, call the LLM via LiteLLM, parse and check citations, guard the context length |
| `evaluate.py` | Retrieval metrics, answer log for manual rating, run snapshots |
| `app.py` | Streamlit UI on top of `retrieve` and `answer` |

Stack: Python 3.11, [uv](https://docs.astral.sh/uv/), pymupdf, Ollama
(`nomic-embed-text`, `qwen2.5:7b`), Chroma, LiteLLM, Streamlit, pytest, ruff.
Developed on a MacBook Pro M1 with 16 GB RAM.

## Installation and usage

Requirements: [Ollama](https://ollama.com) and [uv](https://docs.astral.sh/uv/).

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

uv sync                                   # creates .venv from uv.lock
uv run python -m paper_rag.download       # PDFs -> data/pdfs/ (not in the repo)
uv run python -m paper_rag.index          # chunks + embeddings -> data/index/
uv run streamlit run app.py               # UI at http://localhost:8501
```

Without uv (e.g. in a conda environment): `pip install -e .` and drop the `uv run`.

Further commands:

```bash
uv run python -m paper_rag.ingest                    # chunk statistics per paper
uv run python -m paper_rag.retrieve "question"       # top-k chunks only, no LLM
uv run python -m paper_rag.answer "question"         # answer + citations + token counts
uv run python -m paper_rag.evaluate --label myrun    # full evaluation run
uv run python -m paper_rag.evaluate --summarize eval/results/<run>   # after manual rating
uv run pytest && uv run ruff check .                 # tests and lint (no models needed)
```

All settings live in `config.yaml`; another file can be passed with `--config`
(or `PAPER_RAG_CONFIG=other.yaml` for the app). Switching to an API model is a
configuration change only, e.g. `model: anthropic/<model>` plus the provider's API key
as environment variable (LiteLLM handles the call).

## Papers

Seven papers, all CC BY 4.0. Only metadata and download URLs are in the repository,
not the PDFs.

| id | Title | DOI |
|---|---|---|
| harfoot2014 | Emergent Global Patterns of Ecosystem Structure and Function from a Mechanistic General Ecosystem Model | [10.1371/journal.pbio.1001841](https://doi.org/10.1371/journal.pbio.1001841) |
| krause2025 | Modelling herbivory impacts on vegetation structure and productivity | [10.5194/gmd-18-9633-2025](https://doi.org/10.5194/gmd-18-9633-2025) |
| meunier2017 | From Elements to Function: Toward Unifying Ecological Stoichiometry and Trait-Based Ecology | [10.3389/fenvs.2017.00018](https://doi.org/10.3389/fenvs.2017.00018) |
| traulsen2023 | The future of theoretical evolutionary game theory | [10.1098/rstb.2021.0508](https://doi.org/10.1098/rstb.2021.0508) |
| leimar2023 | Game theory in biology: 50 years and onwards | [10.1098/rstb.2021.0509](https://doi.org/10.1098/rstb.2021.0509) |
| yang2023 | Time delays modulate the stability of complex ecosystems | [10.1038/s41559-023-02158-x](https://doi.org/10.1038/s41559-023-02158-x) |
| glazier2022 | Variable metabolic scaling breaks the law: from 'Newtonian' to 'Darwinian' approaches | [10.1098/rspb.2022.1605](https://doi.org/10.1098/rspb.2022.1605) |

## Design decisions

- **Chunks of ≤ 1000 characters with 150 characters (15 %) overlap, one page each.**
  About 250 tokens: small enough for a focused embedding, large enough to keep a
  statement together with its context. The overlap keeps sentences at a cut intact in
  at least one chunk. Chunks never cross a page boundary, so every chunk can be cited
  with exactly one page (at the cost of splitting sentences across page breaks).
  Characters instead of tokens avoid a tokenizer dependency.
- **Bibliography removed** from the last line that reads only "References",
  "Bibliography" or "Literature cited". Reference lists are dense in technical terms
  and would produce misleading retrieval hits.
- **nomic-embed-text task prefixes** (`search_document: ` / `search_query: `) as a
  config switch, so their effect can be measured (see results).
- **One Chroma collection per index configuration.** The collection name is a hash of
  embedding model, prefixes and chunking parameters. Changing them makes the app and
  the evaluation fail with "index not found" instead of silently searching an index
  built with other settings; variants can coexist.
- **`num_ctx` set explicitly (8192) and checked twice.** Ollama's default context here
  is 4096 tokens and longer prompts are truncated without an error. The prompt size is
  estimated before the call and compared with the token count Ollama reports after it;
  both checks log a warning.
- **`ollama_chat/` instead of `ollama/` in LiteLLM.** `ollama_chat/` uses Ollama's chat
  endpoint with the model's own chat template; `ollama/` flattens the messages into a
  generic text prompt.
- **`temperature: 0`**, so that differences between evaluation runs come from the
  configuration rather than sampling.
- **Citation checks after generation.** Citations are parsed from the answer; those
  pointing to a page that was not in the context are flagged (`invalid_citations`) and
  shown as a warning in the app.
- **Dependency injection for models.** The embedding and completion functions are
  passed in, so all 86 tests run with fakes, in seconds and in CI without Ollama.

## Evaluation

### Design

- `eval/questions.yaml`: 18 questions, 15 answerable (2–3 per paper) and 3 whose answer
  is **not** in the collection, with topically close distractors (tests refusal).
  Each question has `expected_paper`, `expected_pages` (1-based PDF pages) and `notes`
  with the expected content; the expected pages were checked against the extracted text.
- **Retrieval:** hit@k (k = 1, 3, 5) and MRR@5, at paper level (any chunk of the right
  paper) and page level (a chunk of the right paper on an expected page).
- **Answers:** automatic checks (refusals, cited expected page, citations outside the
  context, latency) plus a manual rating per answer: `correct` / `partial` / `wrong` /
  `correctly_refused` / `false_refusal`. Rating rule: the rating judges the content;
  the citation format is covered by the automatic checks. No LLM-as-a-judge yet.
- Every run writes `retrieval.csv`, `answers.csv`, `summary.md` and a `config.yaml`
  snapshot (with index name and git commit) to `eval/results/<timestamp>_<label>/`.

### Results of the first run

Run [`20261122-222759_baseline`](eval/results/20261122-222759_baseline/summary.md):
`qwen2.5:7b`, `nomic-embed-text` with prefixes, 1000/150 chunks, top_k 5.

| Retrieval (15 answerable questions) | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| paper level | 0.93 | 1.00 | 1.00 | 0.97 |
| page level | 0.67 | 0.87 | 0.87 | 0.76 |

| Answers (manual rating, 18 questions) | count |
|---|---|
| correct | 11 |
| partial | 1 |
| wrong | 0 |
| correctly_refused | 3 / 3 |
| false_refusal | 3 |

Automatic checks: 11/15 answers cite an expected page, 0/18 cite a page outside the
context. Median latency 13.6 s (max 24.4 s) on the M1.

**Findings**

- **All three false refusals are retrieval misses, not LLM errors.** Ranked over the
  whole index (516 chunks), the chunk containing the answer was at position 59 (q01),
  9 (q02) and 16 (q05). Refusing was the correct behaviour given that context; with
  `top_k: 10`, q02 is answered correctly (prompt ≈ 2,300 tokens, latency ≈ 30 s).
- **Embeddings favour topic over detail.** The answer sentence for q01 ("organisms with
  body masses between 10 mg and 150,000 kg …") does not mention the Madingley model and
  has a cosine similarity of 0.73 to the question; a generic sentence about the
  Madingley model scores 0.82. Prepending "The Madingley model simulates" to the
  answer sentence raises it to 0.93, which motivates adding document context to
  chunks (see next steps).
- **Paper-level hits overstate retrieval quality.** hit@5 is 1.00 at paper level but
  answers depend on the exact passage; even page level is lenient (q02 counted as a hit
  with a different chunk of the same page).
- The one `partial` (q13) paraphrases the result too narrowly; q15 is correct in
  content but copies the paper's numeric references instead of `[paper_id, p. X]`.

### Variant: without nomic task prefixes

Run [`20261123-152107_noprefix`](eval/results/20261123-152107_noprefix/summary.md)
(retrieval only, `config-noprefix.yaml`):

| Page level | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| with prefixes | 0.67 | 0.87 | 0.87 | 0.76 |
| without prefixes | 0.47 | 0.80 | 0.87 | 0.62 |

Paper level is identical. The prefixes mainly move the right page to the top. With 15
questions one question equals 0.067, so this is an indication, not a significant
result.

## Known limitations

- **PDF text extraction alters content.** Superscripts and minus signs are lost
  (`10⁻⁵ kg` becomes `1025 kg`, `km⁻²` becomes `km22`), and in harfoot2014 the text layer
  reads "10 mg" where the paper prints "10 µg" (µ set in a symbol font). An answer can
  therefore reproduce its context faithfully and still be wrong. Some author names keep
  separated diacritics (`Jo¨ rn`).
- **Running headers and footers** (e.g. "J. Krause et al.: Modelling herbivory …") are
  part of the chunks and inflate similarity between unrelated passages.
- **The bibliography heuristic** cuts everything after the last references heading, so
  appendices placed after the references are lost; supporting-information and
  author-contribution sections are indexed.
- **Small evaluation set** (15 + 3 questions, one rater, questions drafted with an LLM).
  Differences of one or two questions between variants are not meaningful.
- **Page-level evaluation is coarser than the actual retrieval unit** (chunk).
- **Phrasing sensitivity:** a shorter wording of q05 retrieves the right page, the
  wording in the question set does not.
- Answers take 10–30 s on an M1 with a 7B model.

## Next steps

- Add document context to each chunk before embedding (title, section heading), as the
  q01 experiment suggests.
- Hybrid search (BM25 + vector) for exact terms and numbers; re-ranking of the top
  candidates.
- Remove running headers and footers during ingestion.
- Chunk-level (passage-level) ground truth in addition to pages; more questions.
- Compare chunk sizes and embedding models.
- Compare the local model with an API model (quality, latency, cost) and with
  `gemma3:12b`.
- LLM-as-a-judge, validated against the manual ratings.
- Docker image and a FastAPI endpoint.

## License

No license is granted for the code in this repository. The papers are published under
CC BY 4.0 by their authors and are only linked here, not redistributed.
