"""Test del chunking (offline)."""

from __future__ import annotations

from rag_reports.chunking import Chunk, chunk_documents, chunk_text, split_sentences


def test_split_sentences_basic():
    text = "Reperti: niente. Conclusioni: nella norma."
    sents = split_sentences(text)
    assert len(sents) >= 2
    assert all(s for s in sents)


def test_chunk_text_respects_max_chars():
    text = " ".join([f"Frase numero {i}." for i in range(40)])
    chunks = chunk_text(text, max_chars=80, overlap_chars=10)
    assert len(chunks) > 1
    # ogni chunk non eccede troppo il limite (una frase può sforare da sola)
    assert all(len(c) <= 80 + 20 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_documents_inherits_doc_id():
    docs = [{"id": "DOC-1", "text": "Prima frase. Seconda frase.", "exam": "Torace"}]
    chunks = chunk_documents(docs, max_chars=20, overlap_chars=5)
    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_id == "DOC-1" for c in chunks)
    assert all(c.chunk_id.startswith("DOC-1#") for c in chunks)
    assert all(c.meta.get("exam") == "Torace" for c in chunks)
