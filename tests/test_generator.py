"""Test del generatore (offline): fallback estrattivo con citazioni; Ollama mockato."""

from __future__ import annotations

from rag_reports.chunking import Chunk
from rag_reports.generator import (
    ABSTAIN_MESSAGE,
    ExtractiveGenerator,
    OllamaGenerator,
    detect_ollama,
    make_generator,
)
from rag_reports.retriever import RetrievalResult


def _fake_results():
    chunk = Chunk(
        chunk_id="RX-CHEST-001#0",
        doc_id="RX-CHEST-001",
        text="Addensamento al lobo inferiore destro, compatibile con polmonite.",
        meta={"exam": "Torace"},
    )
    return [RetrievalResult(chunk=chunk, score=0.42)]


def test_extractive_includes_citation():
    gen = ExtractiveGenerator()
    out = gen.generate("Polmonite?", _fake_results())
    assert "[doc:RX-CHEST-001]" in out
    assert "Fonti:" in out


def test_extractive_empty_results_abstains():
    gen = ExtractiveGenerator()
    assert gen.generate("qualsiasi", []) == ABSTAIN_MESSAGE


def test_detect_ollama_no_service_no_exception():
    # endpoint inesistente: deve restituire False senza sollevare eccezioni
    assert detect_ollama(host="http://localhost:1", timeout=0.2) is False


def test_make_generator_extractive_default():
    gen = make_generator(backend="extractive")
    assert isinstance(gen, ExtractiveGenerator)


def test_ollama_falls_back_when_unreachable():
    # OllamaGenerator verso host morto → deve ricadere sull'estrattivo (con citazioni)
    gen = OllamaGenerator(model="llama3.2", host="http://localhost:1", timeout=0.2)
    out = gen.generate("Polmonite?", _fake_results())
    assert "[doc:RX-CHEST-001]" in out


def test_auto_backend_without_ollama_is_extractive():
    # con host irraggiungibile, 'auto' deve scegliere l'estrattivo
    gen = make_generator(backend="auto", host="http://localhost:1")
    assert isinstance(gen, ExtractiveGenerator)
