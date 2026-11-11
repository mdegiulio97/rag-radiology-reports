"""Splitting documents into indexable chunks.

The reports in the synthetic corpus are short: the default strategy splits them into
sentences and then groups sentences into windows of at most ``max_chars`` characters,
with an overlap of ``overlap_chars`` so that context is not cut. Each chunk inherits the
``id`` of its source document (used for the ``[doc:ID]`` citations).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Split at sentence ends (period, semicolon or colon followed by whitespace). Simple, offline.
_SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+")


@dataclass(frozen=True)
class Chunk:
    """An indexable piece of text, with a reference to its source document."""

    chunk_id: str        # e.g. "RX-CHEST-001#0"
    doc_id: str          # id of the source document
    text: str            # chunk text
    meta: dict           # metadata inherited from the document (modality, exam, ...)


def split_sentences(text: str) -> list[str]:
    """Split a text into sentences (offline heuristic, no external dependencies)."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip())]
    return [p for p in parts if p]


def chunk_text(text: str, max_chars: int = 320, overlap_chars: int = 60) -> list[str]:
    """Group sentences into windows of at most ``max_chars`` characters, with overlap.

    A single sentence longer than ``max_chars`` is still emitted as its own chunk.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be in [0, max_chars)")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if not current:
            current = sent
        elif len(current) + 1 + len(sent) <= max_chars:
            current = f"{current} {sent}"
        else:
            chunks.append(current)
            # overlap: start again from the tail of the previous chunk
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = f"{tail} {sent}".strip() if tail else sent
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(
    docs: list[dict], max_chars: int = 320, overlap_chars: int = 60
) -> list[Chunk]:
    """Turn a list of documents into a flat list of :class:`Chunk`."""
    out: list[Chunk] = []
    for doc in docs:
        doc_id = str(doc["id"])
        meta = {k: v for k, v in doc.items() if k != "text"}
        pieces = chunk_text(doc["text"], max_chars=max_chars, overlap_chars=overlap_chars)
        if not pieces:
            continue
        for i, piece in enumerate(pieces):
            out.append(
                Chunk(chunk_id=f"{doc_id}#{i}", doc_id=doc_id, text=piece, meta=meta)
            )
    return out
