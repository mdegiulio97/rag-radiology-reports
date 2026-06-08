"""Valutazione del sistema RAG.

Due famiglie di metriche, entrambe offline:

1. **Hit-rate del retrieval @k**: su un set di query curate (domanda → documento
   atteso), frazione di query per cui il documento atteso compare tra i primi ``k``
   risultati.
2. **Anti-allucinazione**: su query fuori-corpus, frazione di query per cui la
   pipeline si astiene correttamente ("non presente nei documenti").

Le query curate di default si riferiscono al corpus SINTETICO versionato.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rag import RagPipeline
from .retriever import BaseRetriever

# Query curate: (domanda, id del documento atteso). Riferite al corpus sintetico.
CURATED_QUERIES: list[tuple[str, str]] = [
    ("Ci sono segni di polmonite al lobo inferiore destro?", "RX-CHEST-001"),
    ("Versamento pleurico sinistro di che entità?", "RX-CHEST-002"),
    ("L'esame del torace è nei limiti della norma?", "RX-CHEST-003"),
    ("Ci sono segni di scompenso cardiaco ed edema polmonare?", "RX-CHEST-004"),
    ("È presente uno pneumotorace?", "RX-CHEST-005"),
    ("È stato descritto un nodulo polmonare solitario?", "RX-CHEST-006"),
    ("Ci sono reperti di embolia polmonare alla TC?", "CT-CHEST-007"),
    ("Si vede enfisema polmonare e BPCO?", "CT-CHEST-009"),
    ("La RM encefalo mostra un ictus ischemico acuto?", "MR-BRAIN-010"),
    ("Ci sono calcoli nella colecisti?", "US-ABD-012"),
    ("È descritta una frattura del radio distale?", "RX-BONE-014"),
    ("È presente un'ernia discale lombare L5-S1?", "MR-SPINE-015"),
]

# Query fuori-corpus: la pipeline deve astenersi (anti-allucinazione).
OUT_OF_CORPUS_QUERIES: list[str] = [
    "Qual è la capitale della Francia?",
    "Qual è la ricetta della carbonara?",
    "Chi ha vinto il campionato di calcio nel 1998?",
    "Quanto fa 2 più 2 in aritmetica modulare?",
    "Come si configura un router domestico?",
]


@dataclass
class RetrievalEval:
    """Esito della valutazione del retrieval."""

    k_values: list[int]
    hit_rate_at_k: dict[int, float]
    n_queries: int
    per_query: list[dict]


@dataclass
class AbstentionEval:
    """Esito del test anti-allucinazione."""

    n_queries: int
    n_correct_abstentions: int
    abstention_rate: float
    per_query: list[dict]


def _doc_ids_in_results(results) -> list[str]:
    seen: list[str] = []
    for r in results:
        if r.chunk.doc_id not in seen:
            seen.append(r.chunk.doc_id)
    return seen


def evaluate_retrieval(
    retriever: BaseRetriever,
    queries: list[tuple[str, str]] | None = None,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> RetrievalEval:
    """Calcola hit-rate@k del retrieval sulle query curate."""
    queries = queries if queries is not None else CURATED_QUERIES
    max_k = max(k_values)
    hits = {k: 0 for k in k_values}
    per_query: list[dict] = []

    for question, expected in queries:
        results = retriever.search(question, k=max_k)
        ranked_ids = _doc_ids_in_results(results)
        rank = ranked_ids.index(expected) + 1 if expected in ranked_ids else None
        for k in k_values:
            if rank is not None and rank <= k:
                hits[k] += 1
        per_query.append(
            {
                "question": question,
                "expected": expected,
                "retrieved": ranked_ids,
                "rank": rank,
            }
        )

    n = len(queries)
    hit_rate = {k: (hits[k] / n if n else 0.0) for k in k_values}
    return RetrievalEval(
        k_values=list(k_values),
        hit_rate_at_k=hit_rate,
        n_queries=n,
        per_query=per_query,
    )


def evaluate_abstention(
    pipeline: RagPipeline, queries: list[str] | None = None
) -> AbstentionEval:
    """Verifica che la pipeline si astenga sulle query fuori-corpus."""
    queries = queries if queries is not None else OUT_OF_CORPUS_QUERIES
    per_query: list[dict] = []
    correct = 0
    for q in queries:
        ans = pipeline.answer(q)
        if ans.abstained:
            correct += 1
        per_query.append({"question": q, "abstained": ans.abstained})
    n = len(queries)
    return AbstentionEval(
        n_queries=n,
        n_correct_abstentions=correct,
        abstention_rate=(correct / n if n else 0.0),
        per_query=per_query,
    )
