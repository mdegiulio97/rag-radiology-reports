"""Corpus loading.

Provides:
- :func:`load_sample_corpus`  → versioned SYNTHETIC corpus (in-code).
- :func:`load_jsonl_corpus`   → generic loader for external corpora in JSONL
  (e.g. Open-i / Indiana University Chest X-ray reports, https://openi.nlm.nih.gov/).
- :func:`write_sample_jsonl`  → serializes the synthetic corpus to JSONL (used by the
  idempotent script ``scripts/download_data.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

from .sample_corpus import get_sample_reports

# Path of the versioned synthetic corpus (relative to the repository root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_JSONL_PATH = _REPO_ROOT / "corpus" / "sample_reports.jsonl"

# Required fields for every document in the corpus.
REQUIRED_FIELDS = ("id", "text")


def _validate_documents(docs: list[dict]) -> list[dict]:
    """Check that every document has the required fields and a unique id."""
    seen: set[str] = set()
    for i, doc in enumerate(docs):
        for field in REQUIRED_FIELDS:
            if field not in doc or doc[field] in (None, ""):
                raise ValueError(f"Document #{i}: required field missing or empty: '{field}'")
        doc_id = str(doc["id"])
        if doc_id in seen:
            raise ValueError(f"Duplicate identifier in corpus: '{doc_id}'")
        seen.add(doc_id)
    return docs


def load_sample_corpus() -> list[dict]:
    """Versioned SYNTHETIC corpus (invented reports, not real patient data)."""
    return _validate_documents(get_sample_reports())


def load_jsonl_corpus(path: str | Path) -> list[dict]:
    """Load an external corpus from a JSONL file (one JSON document per line).

    Each line must contain at least ``id`` and ``text``. Suitable for public corpora
    converted to JSONL (e.g. Open-i). Empty lines are skipped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    docs: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {lineno} of {path}: {exc}") from exc
    return _validate_documents(docs)


def write_sample_jsonl(path: str | Path | None = None) -> Path:
    """Serialize the synthetic corpus to JSONL (idempotent: rewrites the file)."""
    path = Path(path) if path is not None else SAMPLE_JSONL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    docs = load_sample_corpus()
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return path
