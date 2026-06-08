"""Test del retriever TF-IDF (offline): la query seminata recupera il doc giusto."""

from __future__ import annotations

from rag_reports.corpus import load_sample_corpus
from rag_reports.retriever import RetrievalResult, TfidfRetriever, build_retriever


def test_tfidf_retrieves_expected_doc_top1():
    docs = load_sample_corpus()
    retr = build_retriever(docs, dense=False)
    results = retr.search("segni di polmonite al lobo inferiore destro", k=3)
    assert results
    assert isinstance(results[0], RetrievalResult)
    # il documento di polmonite deve essere il primo
    assert results[0].chunk.doc_id == "RX-CHEST-001"


def test_tfidf_pneumothorax_query():
    docs = load_sample_corpus()
    retr = build_retriever(docs, dense=False)
    results = retr.search("è presente uno pneumotorace?", k=3)
    top_ids = [r.chunk.doc_id for r in results]
    assert "RX-CHEST-005" in top_ids


def test_search_results_sorted_desc():
    docs = load_sample_corpus()
    retr = build_retriever(docs, dense=False)
    results = retr.search("versamento pleurico", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_empty():
    docs = load_sample_corpus()
    retr = TfidfRetriever.from_documents(docs)
    assert retr.search("   ", k=3) == []
