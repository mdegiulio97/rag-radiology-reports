# rag-radiology-reports

Sistema **RAG (Retrieval-Augmented Generation) locale** su referti radiologici / testo
clinico, con **citazioni alle fonti** e **controllo anti-allucinazione**. Pensato per
girare **100% offline**: il retrieval di default è TF-IDF (scikit-learn) e la
generazione usa Ollama se disponibile, altrimenti un fallback **estrattivo deterministico**.

> Parte del portfolio `clinical-ml-portfolio`. Autore: Michele De Giulio, 2026 (licenza MIT).

---

## ⚠️ Avvertenza sui dati (LEGGERE)

Il corpus incluso in `corpus/sample_reports.jsonl` (e in
`src/rag_reports/sample_corpus.py`) è **SINTETICO**: si tratta di referti radiologici
**generati artificialmente** a scopo dimostrativo. **NON** sono dati reali di pazienti,
non provengono da alcuna cartella clinica e non vanno usati per scopi clinici. Ogni
record è etichettato con `"synthetic": true` nei metadati.

Per usare un corpus **reale** e pubblico vedi più sotto (Open-i / Indiana University
Chest X-ray reports).

---

## Pipeline

```
ingest documenti → chunking → indicizzazione (TF-IDF) → retrieval (top-k)
        → generazione GROUNDED con citazioni [doc:ID] → anti-allucinazione
```

- **Retrieval (default): TF-IDF** (`scikit-learn`). Nessun download, nessuna rete.
  Opzionale: embedding densi con `sentence-transformers` dietro extra `dense`
  (NON richiesto, NON usato nei test).
- **Generazione: Ollama locale** su `http://localhost:11434` se il servizio risponde;
  **altrimenti fallback estrattivo** che compone la risposta dai chunk recuperati e
  cita le fonti `[doc:ID]`. I **test usano sempre il fallback** (Ollama non è richiesto).
- **Anti-allucinazione**: se la similarità massima del retrieval è sotto soglia, il
  sistema si astiene e risponde:
  *"L'informazione non è presente nei documenti."*

## Struttura

```
rag-radiology-reports/
├── app/main.py                 # Streamlit: domanda → risposta + passaggi-fonte citati
├── src/rag_reports/
│   ├── corpus.py               # carica il corpus sintetico versionato; loader esterni
│   ├── chunking.py             # split del testo in chunk
│   ├── retriever.py            # TF-IDF + interfaccia per embedding densi opzionali
│   ├── generator.py            # backend Ollama con fallback estrattivo; detect Ollama
│   ├── rag.py                  # orchestrazione retrieve → genera con citazioni
│   ├── evaluation.py           # hit-rate retrieval + test anti-allucinazione
│   └── sample_corpus.py        # corpus sintetico in-code (fonte di verità versionata)
├── scripts/download_data.py    # prepara/valida il corpus (idempotente)
├── scripts/evaluate.py         # metriche → results/metrics.json + figura
├── tests/                      # pytest OFFLINE (TF-IDF + fallback, niente Ollama/rete)
├── notebooks/01_validation.ipynb
├── results/                    # metrics.json + figura (committati)
├── corpus/sample_reports.jsonl # corpus sintetico versionato
├── data/.gitkeep               # per corpus esterni opzionali (gitignorato)
├── pyproject.toml
├── README.md / LICENSE / .gitignore
```

## Installazione

Requisiti: [uv](https://docs.astral.sh/uv/) e Python 3.12.

```bash
uv sync --extra dev
```

## Uso rapido

Validare/rigenerare il corpus sintetico (idempotente):

```bash
uv run python scripts/download_data.py
```

Calcolare le metriche di retrieval e anti-allucinazione:

```bash
uv run python scripts/evaluate.py
# → results/metrics.json + results/hit_rate.png
```

Interfaccia chat (Streamlit):

```bash
uv run streamlit run app/main.py
```

Esempio da Python:

```python
from rag_reports.corpus import load_sample_corpus
from rag_reports.rag import RagPipeline

rag = RagPipeline.from_documents(load_sample_corpus())
ans = rag.answer("Ci sono segni di polmonite?")
print(ans.text)          # risposta con citazioni [doc:ID]
print(ans.sources)       # passaggi-fonte usati
print(ans.abstained)     # True se ha rifiutato per mancanza di evidenza
```

## Usare Ollama come backend reale (opzionale)

Il sistema rileva Ollama a runtime; se assente, usa il fallback estrattivo senza errori.
Per usare un LLM locale:

```bash
# 1) installa Ollama: https://ollama.com
# 2) scarica un modello, es.:
ollama pull llama3.2
# 3) avvia il servizio (di norma parte da solo): ollama serve
# 4) nell'app seleziona il backend "Ollama (auto)" oppure:
uv run python -c "from rag_reports.corpus import load_sample_corpus; \
from rag_reports.rag import RagPipeline; \
r=RagPipeline.from_documents(load_sample_corpus(), backend='ollama', ollama_model='llama3.2'); \
print(r.answer('Descrivi i reperti di versamento pleurico.').text)"
```

Il servizio Ollama atteso su `http://localhost:11434`. Variabile d'ambiente
`OLLAMA_HOST` per cambiare endpoint.

## Estendere a un corpus reale (Open-i)

La pipeline accetta un JSONL con campi `{id, text, ...meta}`. Un dataset reale e
pubblico è **Open-i / Indiana University Chest X-ray reports**
(<https://openi.nlm.nih.gov/>). Scarica i referti, convertili in JSONL e usa
`load_jsonl_corpus("data/openi_reports.jsonl")`. La cartella `data/` è gitignorata
proprio per ospitare corpora esterni non versionabili.

## Risultati (numeri reali)

Metriche calcolate da `scripts/evaluate.py` sul corpus sintetico con un set di
query curate (domanda → documento atteso). Vedi `results/metrics.json`.

<!-- METRICS:START -->
- **Hit-rate@1**: 1.000
- **Hit-rate@3**: 1.000
- **Hit-rate@5**: 1.000
- **Query curate (in-corpus)**: 12
- **Anti-allucinazione** (query fuori-corpus correttamente astenute): 5 / 5 = 1.000
<!-- METRICS:END -->

Figura: `results/hit_rate.png` (hit-rate@k).

## Test

```bash
uv run ruff check .
uv run pytest -q
```

I test sono **offline**: verificano chunking, retrieval TF-IDF (la query seminata
recupera il documento giusto), il fallback estrattivo (risposta con citazioni) e
l'astensione su query fuori-corpus. Il backend Ollama è mockato/saltato.

## Riferimenti

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*,
  NeurIPS 2020. arXiv:2005.11401 — <https://arxiv.org/abs/2005.11401>
- Open-i / Indiana University Chest X-ray reports — <https://openi.nlm.nih.gov/>
- Ollama — <https://ollama.com>
- scikit-learn TF-IDF — <https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting>

## Licenza

MIT — vedi `LICENSE`.
