"""Test del corpus sintetico (offline): integrità e marcatura 'synthetic'."""

from __future__ import annotations

import json

from rag_reports.corpus import load_jsonl_corpus, load_sample_corpus, write_sample_jsonl


def test_sample_corpus_loads_and_is_synthetic():
    docs = load_sample_corpus()
    assert 10 <= len(docs) <= 20
    assert all(d.get("synthetic") is True for d in docs)
    assert all(d.get("id") and d.get("text") for d in docs)


def test_sample_corpus_ids_unique():
    docs = load_sample_corpus()
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids))


def test_write_and_reload_jsonl(tmp_path):
    out = tmp_path / "corpus.jsonl"
    write_sample_jsonl(out)
    reloaded = load_jsonl_corpus(out)
    assert len(reloaded) == len(load_sample_corpus())
    # ogni riga è JSON valido
    for line in out.read_text(encoding="utf-8").splitlines():
        json.loads(line)
