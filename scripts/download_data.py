"""Prepara e valida il corpus SINTETICO versionato (idempotente).

Questo script NON scarica nulla dalla rete: il corpus sintetico è la fonte di verità
in-code (``src/rag_reports/sample_corpus.py``). Lo script lo serializza in
``corpus/sample_reports.jsonl`` e ne verifica l'integrità. Eseguirlo più volte
produce lo stesso risultato.

Per usare un corpus REALE e pubblico (es. Open-i / Indiana University Chest X-ray,
https://openi.nlm.nih.gov/) convertilo in JSONL e mettilo in ``data/`` (gitignorata),
poi caricalo con ``rag_reports.corpus.load_jsonl_corpus``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permette l'esecuzione diretta (python scripts/download_data.py) senza install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_reports.corpus import (  # noqa: E402
    SAMPLE_JSONL_PATH,
    load_jsonl_corpus,
    load_sample_corpus,
    write_sample_jsonl,
)


def main() -> int:
    docs = load_sample_corpus()
    n_synth = sum(1 for d in docs if d.get("synthetic") is True)

    if n_synth != len(docs):
        print("ERRORE: non tutti i referti sono marcati come sintetici.")
        return 1

    path = write_sample_jsonl()
    # Re-load dal file per validare la serializzazione (round-trip).
    reloaded = load_jsonl_corpus(path)

    if len(reloaded) != len(docs):
        print("ERRORE: il numero di documenti dopo il round-trip non coincide.")
        return 1

    print("Corpus SINTETICO validato (referti generati artificialmente, NON reali).")
    print(f"  Documenti: {len(docs)} (tutti synthetic=True)")
    print(f"  File JSONL: {SAMPLE_JSONL_PATH}")
    print("  Round-trip JSONL: OK")
    print("\nPer un corpus reale e pubblico: Open-i (https://openi.nlm.nih.gov/) -> JSONL in data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
