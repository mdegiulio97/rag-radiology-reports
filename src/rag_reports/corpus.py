"""Caricamento del corpus.

Espone:
- :func:`load_sample_corpus`  → corpus SINTETICO versionato (in-code).
- :func:`load_jsonl_corpus`   → loader generico per corpora esterni in JSONL
  (es. Open-i / Indiana University Chest X-ray reports, https://openi.nlm.nih.gov/).
- :func:`write_sample_jsonl`  → serializza il corpus sintetico in JSONL (usato dallo
  script idempotente ``scripts/download_data.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

from .sample_corpus import get_sample_reports

# Percorso del corpus sintetico versionato (relativo alla root del repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_JSONL_PATH = _REPO_ROOT / "corpus" / "sample_reports.jsonl"

# Campi obbligatori per ogni documento del corpus.
REQUIRED_FIELDS = ("id", "text")


def _validate_documents(docs: list[dict]) -> list[dict]:
    """Valida che ogni documento abbia i campi richiesti e id univoci."""
    seen: set[str] = set()
    for i, doc in enumerate(docs):
        for field in REQUIRED_FIELDS:
            if field not in doc or doc[field] in (None, ""):
                raise ValueError(f"Documento #{i}: campo obbligatorio mancante o vuoto: '{field}'")
        doc_id = str(doc["id"])
        if doc_id in seen:
            raise ValueError(f"Identificatore duplicato nel corpus: '{doc_id}'")
        seen.add(doc_id)
    return docs


def load_sample_corpus() -> list[dict]:
    """Corpus SINTETICO versionato (referti generati artificialmente, non reali)."""
    return _validate_documents(get_sample_reports())


def load_jsonl_corpus(path: str | Path) -> list[dict]:
    """Carica un corpus esterno da file JSONL (un documento JSON per riga).

    Ogni riga deve contenere almeno ``id`` e ``text``. Adatto a corpora pubblici
    convertiti in JSONL (es. Open-i). Le righe vuote vengono ignorate.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus non trovato: {path}")
    docs: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON non valido alla riga {lineno} di {path}: {exc}") from exc
    return _validate_documents(docs)


def write_sample_jsonl(path: str | Path | None = None) -> Path:
    """Serializza il corpus sintetico in JSONL (idempotente: riscrive il file)."""
    path = Path(path) if path is not None else SAMPLE_JSONL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    docs = load_sample_corpus()
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return path
