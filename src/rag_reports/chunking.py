"""Suddivisione dei documenti in chunk indicizzabili.

I referti del corpus sintetico sono brevi: la strategia di default suddivide per
frasi e poi accorpa le frasi in finestre di al più ``max_chars`` caratteri, con un
overlap di ``overlap_chars`` per non spezzare il contesto. Ogni chunk eredita l'``id``
del documento sorgente (usato per le citazioni ``[doc:ID]``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Split su fine-frase (punto, punto e virgola seguiti da spazio). Semplice e offline.
_SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+")


@dataclass(frozen=True)
class Chunk:
    """Un frammento di testo indicizzabile, con riferimento al documento sorgente."""

    chunk_id: str        # es. "RX-CHEST-001#0"
    doc_id: str          # id del documento sorgente
    text: str            # testo del chunk
    meta: dict           # metadati ereditati dal documento (modality, exam, ...)


def split_sentences(text: str) -> list[str]:
    """Divide un testo in frasi (euristica offline, nessuna dipendenza esterna)."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip())]
    return [p for p in parts if p]


def chunk_text(text: str, max_chars: int = 320, overlap_chars: int = 60) -> list[str]:
    """Accorpa le frasi in finestre di al più ``max_chars`` caratteri con overlap.

    Se una singola frase supera ``max_chars`` viene comunque emessa come chunk a sé.
    """
    if max_chars <= 0:
        raise ValueError("max_chars deve essere positivo")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars deve essere in [0, max_chars)")

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
            # overlap: riparti dalla coda del chunk precedente
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = f"{tail} {sent}".strip() if tail else sent
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(
    docs: list[dict], max_chars: int = 320, overlap_chars: int = 60
) -> list[Chunk]:
    """Trasforma una lista di documenti in una lista piatta di :class:`Chunk`."""
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
