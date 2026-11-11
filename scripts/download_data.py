"""Prepare and validate the versioned SYNTHETIC corpus (idempotent).

This script does NOT download anything: the synthetic corpus is the in-code source of
truth (``src/rag_reports/sample_corpus.py``). The script serializes it to
``corpus/sample_reports.jsonl`` and checks its integrity. Running it several times
gives the same result.

To use a REAL, public corpus (e.g. Open-i / Indiana University Chest X-ray,
https://openi.nlm.nih.gov/), convert it to JSONL, put it in ``data/`` (gitignored),
then load it with ``rag_reports.corpus.load_jsonl_corpus``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allows direct execution (python scripts/download_data.py) without installing.
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
        print("ERROR: not all reports are marked as synthetic.")
        return 1

    path = write_sample_jsonl()
    # Reload from the file to validate the serialization (round trip).
    reloaded = load_jsonl_corpus(path)

    if len(reloaded) != len(docs):
        print("ERROR: the number of documents after the round trip does not match.")
        return 1

    print("SYNTHETIC corpus validated (invented reports, NOT real patient data).")
    print(f"  Documents: {len(docs)} (all synthetic=True)")
    print(f"  JSONL file: {SAMPLE_JSONL_PATH}")
    print("  JSONL round trip: OK")
    print("\nFor a real, public corpus: Open-i (https://openi.nlm.nih.gov/) -> JSONL in data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
