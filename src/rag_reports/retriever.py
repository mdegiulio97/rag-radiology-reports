"""Chunk retrieval.

Default: **TF-IDF** (scikit-learn) with cosine similarity — fully offline.
Optional (``dense`` extra): dense embeddings with ``sentence-transformers``, behind a
flag, NOT used in the tests and NOT required by the base dependencies.

All retrievers expose the same :class:`BaseRetriever` interface with
``search(query, k) -> list[RetrievalResult]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk, chunk_documents

# Common Italian stop words (the corpus is in Italian): scikit-learn ships no Italian
# list. Removing them keeps function words (articles, prepositions, interrogatives) from
# inflating the similarity of out-of-corpus queries, which widens the margin for the
# abstention threshold.
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
    """Search result: a chunk with its similarity score."""

    chunk: Chunk
    score: float


class BaseRetriever(ABC):
    """Common interface for retrievers (TF-IDF and, optionally, dense embeddings)."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("The retriever needs at least one chunk.")
        self.chunks = chunks

    @classmethod
    def from_documents(
        cls, docs: list[dict], max_chars: int = 320, overlap_chars: int = 60, **kwargs
    ) -> "BaseRetriever":
        chunks = chunk_documents(docs, max_chars=max_chars, overlap_chars=overlap_chars)
        return cls(chunks, **kwargs)

    @abstractmethod
    def search(self, query: str, k: int = 3) -> list[RetrievalResult]:
        """Return the ``k`` most relevant chunks, sorted by decreasing score."""


class TfidfRetriever(BaseRetriever):
    """TF-IDF retriever + cosine similarity. Default backend, offline."""

    def __init__(self, chunks: list[Chunk]):
        super().__init__(chunks)
        # Unigrams + bigrams; lowercase; scikit-learn has no built-in Italian stop
        # list (hence ITALIAN_STOPWORDS), and TF-IDF already down-weights very
        # frequent terms.
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
        # sort by decreasing similarity and keep the top k
        top_idx = np.argsort(-sims)[:k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(sims[i]))
            for i in top_idx
        ]


class DenseRetriever(BaseRetriever):
    """Retriever with dense embeddings (sentence-transformers). OPTIONAL.

    Requires the ``dense`` extra: ``uv sync --extra dense``. Not used in the tests and
    not on the default path. Lazy import so the dependency is not required.
    """

    def __init__(self, chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(chunks)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on an optional extra
            raise ImportError(
                "DenseRetriever requires 'sentence-transformers'. "
                "Install it with: uv sync --extra dense"
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
    """Factory: TF-IDF (default), or dense if ``dense=True`` and the extra is installed."""
    if dense:
        return DenseRetriever.from_documents(docs, **kwargs)
    return TfidfRetriever.from_documents(docs, **kwargs)
