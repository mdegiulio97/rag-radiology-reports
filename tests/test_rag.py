"""Test della pipeline RAG end-to-end (offline, backend estrattivo)."""

from __future__ import annotations

from rag_reports.corpus import load_sample_corpus
from rag_reports.generator import ABSTAIN_MESSAGE
from rag_reports.rag import RagPipeline


def _pipeline():
    return RagPipeline.from_documents(load_sample_corpus(), backend="extractive")


def test_in_corpus_query_answers_with_citation():
    rag = _pipeline()
    ans = rag.answer("Ci sono segni di polmonite al lobo inferiore destro?")
    assert not ans.abstained
    assert "RX-CHEST-001" in ans.source_ids
    assert "[doc:RX-CHEST-001]" in ans.text


def test_out_of_corpus_query_abstains():
    rag = _pipeline()
    ans = rag.answer("Qual è la capitale della Francia?")
    assert ans.abstained
    assert ans.text == ABSTAIN_MESSAGE
    assert ans.sources == []


def test_backend_name_is_extractive():
    rag = _pipeline()
    ans = rag.answer("È presente uno pneumotorace?")
    assert ans.backend == "extractive"


def test_pneumothorax_end_to_end():
    rag = _pipeline()
    ans = rag.answer("È presente uno pneumotorace?")
    assert not ans.abstained
    assert "RX-CHEST-005" in ans.source_ids
