"""Retrieval dei chunk.

Default: **TF-IDF** (scikit-learn) con similarità coseno — 100% offline.
Opzionale (extra ``dense``): embedding densi con ``sentence-transformers``, dietro
flag, NON usato nei test e NON richiesto dalle dipendenze di base.

Tutti i retriever espongono la stessa interfaccia :class:`BaseRetriever` con
``search(query, k) -> list[RetrievalResult]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk, chunk_documents

# Stop-word italiane comuni: scikit-learn non ne fornisce per l'italiano. Rimuoverle
# evita che parole funzione (articoli, preposizioni, interrogativi) gonfino la
# similarità di query fuori-corpus, ampliando il margine per l'anti-allucinazione.
ITALIAN_STOPWORDS: list[str] = [
    "a", "ad", "al", "alla", "alle", "agli", "ai", "allo", "anche", "c", "che", "chi",
    "ci", "co", "col", "come", "con", "cosa", "da", "dal", "dalla", "dei", "del",
    "della", "delle", "dello", "degli", "di", "dove", "e", "ed", "è", "gli", "ha",
    "hai", "hanno", "ho", "i", "il", "in", "io", "la", "le", "li", "lo", "ma", "mi",
    "ne", "nei", "nel", "nella", "nelle", "nello", "negli", "no", "noi", "non", "o",
    "per", "più", "qual", "quale", "quali", "quando", "quanto", "quante", "quanti",
    "quanta", "che", "se", "si", "sono", "su", "sui", "sul", "sulla", "sulle", "sullo",
    "sugli", "te", "ti", "tra", "tu", "tua", "tuo", "un", "una", "uno", "vi", "voi",
    "fa", "fra", "con", "del", "ci", "ne", "presente", "segni", "reperti",
]


@dataclass(frozen=True)
class RetrievalResult:
    """Risultato di una ricerca: un chunk con il suo punteggio di similarità."""

    chunk: Chunk
    score: float


class BaseRetriever(ABC):
    """Interfaccia comune ai retriever (TF-IDF e, opzionale, embedding densi)."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Il retriever richiede almeno un chunk.")
        self.chunks = chunks

    @classmethod
    def from_documents(
        cls, docs: list[dict], max_chars: int = 320, overlap_chars: int = 60, **kwargs
    ) -> "BaseRetriever":
        chunks = chunk_documents(docs, max_chars=max_chars, overlap_chars=overlap_chars)
        return cls(chunks, **kwargs)

    @abstractmethod
    def search(self, query: str, k: int = 3) -> list[RetrievalResult]:
        """Restituisce i ``k`` chunk più rilevanti, ordinati per score decrescente."""


class TfidfRetriever(BaseRetriever):
    """Retriever TF-IDF + similarità coseno. Backend di default, offline."""

    def __init__(self, chunks: list[Chunk]):
        super().__init__(chunks)
        # Unigrammi + bigrammi; lowercase; niente stop-list italiana built-in in
        # scikit-learn, ma TF-IDF già pesa al ribasso i termini molto frequenti.
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            stop_words=ITALIAN_STOPWORDS,
        )
        self._matrix = self._vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 3) -> list[RetrievalResult]:
        if not query or not query.strip():
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).ravel()
        k = max(1, min(k, len(self.chunks)))
        # ordinamento parziale per efficienza, poi ordinamento fine
        top_idx = np.argsort(-sims)[:k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(sims[i]))
            for i in top_idx
        ]


class DenseRetriever(BaseRetriever):
    """Retriever con embedding densi (sentence-transformers). OPZIONALE.

    Richiede l'extra ``dense``: ``uv sync --extra dense``. Non usato nei test e non
    nel percorso di default. Import pigro per non imporre la dipendenza.
    """

    def __init__(self, chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(chunks)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dipende da extra opzionale
            raise ImportError(
                "DenseRetriever richiede 'sentence-transformers'. "
                "Installa con: uv sync --extra dense"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._emb = self._model.encode(
            [c.text for c in chunks], normalize_embeddings=True
        )

    def search(self, query: str, k: int = 3) -> list[RetrievalResult]:  # pragma: no cover
        if not query or not query.strip():
            return []
        q = self._model.encode([query], normalize_embeddings=True)
        sims = (self._emb @ q.T).ravel()
        k = max(1, min(k, len(self.chunks)))
        top_idx = np.argsort(-sims)[:k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(sims[i]))
            for i in top_idx
        ]


def build_retriever(
    docs: list[dict], dense: bool = False, **kwargs
) -> BaseRetriever:
    """Factory: TF-IDF (default) oppure densi se ``dense=True`` ed extra installato."""
    if dense:
        return DenseRetriever.from_documents(docs, **kwargs)
    return TfidfRetriever.from_documents(docs, **kwargs)
