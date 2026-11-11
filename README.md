# rag-radiology-reports

A **local RAG (Retrieval-Augmented Generation)** system over radiology reports / clinical
text, with **source citations** and an **abstention check against hallucination**. It is
designed to run **fully offline**: retrieval uses TF-IDF (scikit-learn) by default, and
generation uses Ollama when it is available, otherwise a **deterministic extractive
fallback**.

> Part of the `clinical-ml-portfolio`. Author: Michele De Giulio, 2026 (MIT license).

---

## Data notice (read this first)

The corpus in `corpus/sample_reports.jsonl` (and in `src/rag_reports/sample_corpus.py`)
is **SYNTHETIC**: the radiology reports were written for demonstration purposes. They are
**NOT** real patient data, do not come from any medical record and must not be used for
clinical purposes. Every record is tagged with `"synthetic": true` in its metadata.

The synthetic corpus and the evaluation queries are **in Italian**. They are kept in
Italian because the retrieval setup (including the Italian stop-word list in
`retriever.py`) and the tests are built around them.

To use a **real**, public corpus, see below (Open-i / Indiana University Chest X-ray
reports).

---

## Pipeline

```
document ingestion → chunking → indexing (TF-IDF) → retrieval (top-k)
        → GROUNDED generation with [doc:ID] citations → hallucination check
```

- **Retrieval (default): TF-IDF** (`scikit-learn`). No downloads, no network.
  Optional: dense embeddings with `sentence-transformers` behind the `dense` extra
  (NOT required, NOT used in the tests).
- **Generation: local Ollama** at `http://localhost:11434` if the service responds;
  **otherwise an extractive fallback** that builds the answer from the retrieved chunks
  and cites the sources as `[doc:ID]`. **The tests always use the fallback** (Ollama is
  not required).
- **Hallucination check**: if the highest retrieval similarity is below a threshold, the
  system abstains and returns a fixed message stating that the information is not present
  in the documents (`ABSTAIN_MESSAGE` in `generator.py`).

## Layout

```
rag-radiology-reports/
├── app/main.py                 # Streamlit: question → answer + cited source passages
├── src/rag_reports/
│   ├── corpus.py               # loads the versioned synthetic corpus; external loaders
│   ├── chunking.py             # splits text into chunks
│   ├── retriever.py            # TF-IDF + interface for optional dense embeddings
│   ├── generator.py            # Ollama backend with extractive fallback; Ollama detection
│   ├── rag.py                  # orchestration: retrieve → generate with citations
│   ├── evaluation.py           # retrieval hit-rate + abstention test
│   └── sample_corpus.py        # in-code synthetic corpus (versioned source of truth)
├── scripts/download_data.py    # prepares/validates the corpus (idempotent)
├── scripts/evaluate.py         # metrics → results/metrics.json + figure
├── tests/                      # OFFLINE pytest (TF-IDF + fallback, no Ollama/network)
├── notebooks/01_validation.ipynb
├── results/                    # metrics.json + figure (committed)
├── corpus/sample_reports.jsonl # versioned synthetic corpus
├── data/.gitkeep               # for optional external corpora (gitignored)
├── pyproject.toml
├── README.md / LICENSE / .gitignore
```

## Installation

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --extra dev
```

## Quick start

Validate/regenerate the synthetic corpus (idempotent):

```bash
uv run python scripts/download_data.py
```

Compute the retrieval and abstention metrics:

```bash
uv run python scripts/evaluate.py
# → results/metrics.json + results/hit_rate.png
```

Chat interface (Streamlit):

```bash
uv run streamlit run app/main.py
```

Python example (the question is in Italian, like the corpus: "Are there signs of
pneumonia?"):

```python
from rag_reports.corpus import load_sample_corpus
from rag_reports.rag import RagPipeline

rag = RagPipeline.from_documents(load_sample_corpus())
ans = rag.answer("Ci sono segni di polmonite?")
print(ans.text)          # answer with [doc:ID] citations
print(ans.sources)       # source passages used
print(ans.abstained)     # True if it declined for lack of evidence
```

## Using Ollama as the generation backend (optional)

The system detects Ollama at runtime; if it is not running, it uses the extractive
fallback without errors. To use a local LLM:

```bash
# 1) install Ollama: https://ollama.com
# 2) pull a model, e.g.:
ollama pull llama3.2
# 3) start the service (it usually starts on its own): ollama serve
# 4) in the app select the "Ollama (auto)" backend, or
#    (the question means "Describe the pleural effusion findings."):
uv run python -c "from rag_reports.corpus import load_sample_corpus; \
from rag_reports.rag import RagPipeline; \
r=RagPipeline.from_documents(load_sample_corpus(), backend='ollama', ollama_model='llama3.2'); \
print(r.answer('Descrivi i reperti di versamento pleurico.').text)"
```

The Ollama service is expected at `http://localhost:11434`. Set the `OLLAMA_HOST`
environment variable to change the endpoint.

## Extending to a real corpus (Open-i)

The pipeline accepts a JSONL file with fields `{id, text, ...meta}`. A real, public
dataset is **Open-i / Indiana University Chest X-ray reports**
(<https://openi.nlm.nih.gov/>). Download the reports, convert them to JSONL and use
`load_jsonl_corpus("data/openi_reports.jsonl")`. The `data/` folder is gitignored so it
can hold external corpora that should not be versioned. Open-i reports are in English,
so the Italian stop-word list in `retriever.py` would need to be replaced.

## Functional check (not a benchmark)

`scripts/evaluate.py` runs two checks on the synthetic corpus and writes them to
`results/metrics.json`:

- retrieval hit-rate@k on 12 in-corpus queries, each paired with the report it should
  retrieve;
- abstention on 5 out-of-corpus queries (general-knowledge questions unrelated to
  radiology), where the correct behaviour is to decline.

<!-- METRICS:START -->
- **Hit-rate@1**: 1.000
- **Hit-rate@3**: 1.000
- **Hit-rate@5**: 1.000
- **In-corpus queries**: 12
- **Abstention** (out-of-corpus queries correctly declined): 5 / 5 = 1.000
<!-- METRICS:END -->

These numbers come from a small set of queries written by the author over a small
synthetic corpus of 15 reports. They show that the pipeline works end to end (chunking,
retrieval, citation, abstention). They do **not** measure retrieval quality: the queries
reuse the wording of the reports they target, and there is no held-out or independently
annotated query set.

`results/metrics.json` also contains an ablation with the abstention threshold set to 0
(`min_score=0`): the system then answers every out-of-corpus query (0 / 5 abstentions),
which shows that the threshold is what triggers the abstention.

Figure: `results/hit_rate.png` (hit-rate@k).

## Tests

```bash
uv run ruff check .
uv run pytest -q
```

The tests run **offline**: they check chunking, TF-IDF retrieval (a seeded query
retrieves the expected report), the extractive fallback (answer with citations) and
abstention on out-of-corpus queries. The Ollama backend is mocked or skipped.

## Data and licenses

- The corpus in `corpus/sample_reports.jsonl` and `src/rag_reports/sample_corpus.py` is
  synthetic. It was written for this project, is versioned in this repository and is
  covered by the repository's MIT license.
- It contains no patient data.
- External corpora such as Open-i are not included; if you download one, its own terms
  of use apply.

## References

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*,
  NeurIPS 2020. arXiv:2005.11401 — <https://arxiv.org/abs/2005.11401>
- Open-i / Indiana University Chest X-ray reports — <https://openi.nlm.nih.gov/>
- Ollama — <https://ollama.com>
- scikit-learn TF-IDF — <https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting>

## License

MIT — see `LICENSE`.
