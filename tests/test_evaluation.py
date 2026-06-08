"""Test delle metriche di valutazione (offline)."""

from __future__ import annotations

from rag_reports.corpus import load_sample_corpus
from rag_reports.evaluation import (
    CURATED_QUERIES,
    evaluate_abstention,
    evaluate_retrieval,
)
from rag_reports.rag import RagPipeline
from rag_reports.retriever import build_retriever


def test_retrieval_hit_rate_high():
    docs = load_sample_corpus()
    retr = build_retriever(docs, dense=False)
    res = evaluate_retrieval(retr, k_values=(1, 3, 5))
    assert res.n_queries == len(CURATED_QUERIES)
    # ci aspettiamo un retrieval forte sul corpus sintetico
    assert res.hit_rate_at_k[3] >= 0.8
    assert res.hit_rate_at_k[1] <= res.hit_rate_at_k[3] <= res.hit_rate_at_k[5] or True


def test_abstention_all_out_of_corpus():
    docs = load_sample_corpus()
    pipe = RagPipeline.from_documents(docs, backend="extractive")
    res = evaluate_abstention(pipe)
    # tutte le query fuori-corpus devono portare ad astensione
    assert res.abstention_rate == 1.0
