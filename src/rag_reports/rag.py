"""RAG orchestration: retrieve → (hallucination check) → generate with citations.

The abstention threshold is applied to the **best** retrieval score: if the top score
is below ``min_score`` the system abstains and returns the standard message
(``ABSTAIN_MESSAGE``) instead of producing content.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .generator import ABSTAIN_MESSAGE, make_generator
from .retriever import BaseRetriever, RetrievalResult, build_retriever

# Default threshold on the top TF-IDF score (cosine similarity in [0, 1]).
DEFAULT_MIN_SCORE = 0.05


@dataclass
class RagAnswer:
    """Result of a RAG query."""

    question: str
    text: str
    sources: list[RetrievalResult] = field(default_factory=list)
    abstained: bool = False
    backend: str = "extractive"

    @property
    def source_ids(self) -> list[str]:
        """Ids of the cited documents, de-duplicated, in order of relevance."""
        seen: list[str] = []
        for r in self.sources:
            if r.chunk.doc_id not in seen:
                seen.append(r.chunk.doc_id)
        return seen


class RagPipeline:
    """Full RAG pipeline: retriever + generator + abstention logic."""

    def __init__(
        self,
        retriever: BaseRetriever,
        backend: str = "auto",
        min_score: float = DEFAULT_MIN_SCORE,
        ollama_model: str = "llama3.2",
        top_k: int = 3,
    ):
        self.retriever = retriever
        self.min_score = min_score
        self.top_k = top_k
        self.generator = make_generator(backend=backend, model=ollama_model)

    @classmethod
    def from_documents(
        cls,
        docs: list[dict],
        backend: str = "auto",
        min_score: float = DEFAULT_MIN_SCORE,
        ollama_model: str = "llama3.2",
        top_k: int = 3,
        dense: bool = False,
        **chunk_kwargs,
    ) -> "RagPipeline":
        retriever = build_retriever(docs, dense=dense, **chunk_kwargs)
        return cls(
            retriever,
            backend=backend,
            min_score=min_score,
            ollama_model=ollama_model,
            top_k=top_k,
        )

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievalResult]:
        return self.retriever.search(question, k=k or self.top_k)

    def answer(self, question: str, k: int | None = None) -> RagAnswer:
        """Answer the question with citations, or abstain if the evidence is weak."""
        backend_name = getattr(self.generator, "name", "extractive")
        results = self.retrieve(question, k=k)
        top_score = results[0].score if results else 0.0

        if not results or top_score < self.min_score:
            return RagAnswer(
                question=question,
                text=ABSTAIN_MESSAGE,
                sources=[],
                abstained=True,
                backend=backend_name,
            )

        text = self.generator.generate(question, results)
        return RagAnswer(
            question=question,
            text=text,
            sources=results,
            abstained=False,
            backend=backend_name,
        )
