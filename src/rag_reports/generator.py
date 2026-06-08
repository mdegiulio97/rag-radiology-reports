"""Generazione della risposta a partire dai chunk recuperati.

Due backend:

- **Ollama** (``OllamaGenerator``): usa un LLM locale via ``http://localhost:11434``
  se il servizio risponde. Il rilevamento è *robusto*: l'assenza del servizio NON
  solleva eccezioni (try/except su requests, timeout breve).
- **Estrattivo** (``ExtractiveGenerator``): fallback **deterministico** che NON usa
  rete né LLM. Compone la risposta concatenando i passaggi-fonte recuperati e
  aggiunge le citazioni ``[doc:ID]``. È il backend usato dai test.

Entrambi rispettano il contratto: dato un prompt e i chunk-fonte, restituiscono una
stringa di risposta che cita le fonti.
"""

from __future__ import annotations

import os

import requests

from .retriever import RetrievalResult

# Frase standard di astensione (anti-allucinazione).
ABSTAIN_MESSAGE = "L'informazione non è presente nei documenti."

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def detect_ollama(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 0.5) -> bool:
    """True se un servizio Ollama risponde su ``host``. Mai solleva eccezioni."""
    url = host.rstrip("/") + "/api/tags"
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except (requests.exceptions.RequestException, ValueError, OSError):
        return False


def _format_citations(results: list[RetrievalResult]) -> str:
    """Elenco ordinato e deduplicato delle fonti citate, come ``[doc:ID]``."""
    seen: list[str] = []
    for r in results:
        if r.chunk.doc_id not in seen:
            seen.append(r.chunk.doc_id)
    return " ".join(f"[doc:{doc_id}]" for doc_id in seen)


class ExtractiveGenerator:
    """Fallback estrattivo deterministico (nessuna rete, nessun LLM).

    Compone la risposta dai passaggi recuperati e cita ``[doc:ID]``. Deterministico:
    stesso input → stesso output. È il backend di default quando Ollama è assente.
    """

    name = "extractive"

    def generate(self, question: str, results: list[RetrievalResult]) -> str:
        if not results:
            return ABSTAIN_MESSAGE
        lines = [
            "In base ai documenti recuperati, i passaggi pertinenti sono:",
            "",
        ]
        for r in results:
            lines.append(f"- {r.chunk.text} [doc:{r.chunk.doc_id}]")
        lines.append("")
        lines.append(f"Fonti: {_format_citations(results)}")
        return "\n".join(lines)


class OllamaGenerator:
    """Backend Ollama. Costruisce un prompt grounded e chiama l'API locale.

    In caso di errore di rete o servizio assente, ricade sul generatore estrattivo,
    così la pipeline non fallisce mai.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = DEFAULT_OLLAMA_HOST,
        timeout: float = 60.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._fallback = ExtractiveGenerator()

    def _build_prompt(self, question: str, results: list[RetrievalResult]) -> str:
        context_blocks = []
        for r in results:
            context_blocks.append(f"[doc:{r.chunk.doc_id}] {r.chunk.text}")
        context = "\n".join(context_blocks)
        return (
            "Sei un assistente che risponde SOLO in base ai documenti forniti.\n"
            "Cita le fonti tra parentesi quadre nel formato [doc:ID].\n"
            "Se l'informazione non è presente nei documenti, rispondi esattamente: "
            f"\"{ABSTAIN_MESSAGE}\".\n\n"
            f"DOCUMENTI:\n{context}\n\n"
            f"DOMANDA: {question}\n\nRISPOSTA:"
        )

    def generate(self, question: str, results: list[RetrievalResult]) -> str:
        if not results:
            return ABSTAIN_MESSAGE
        prompt = self._build_prompt(question, results)
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or "").strip()
            if not text:
                return self._fallback.generate(question, results)
            return text
        except (requests.exceptions.RequestException, ValueError, OSError):
            # servizio assente / errore di rete / JSON invalido → fallback robusto
            return self._fallback.generate(question, results)


def make_generator(backend: str = "auto", model: str = "llama3.2", host: str = DEFAULT_OLLAMA_HOST):
    """Crea il generatore.

    - ``"extractive"`` → sempre fallback estrattivo (default sicuro/offline).
    - ``"ollama"``     → backend Ollama (con auto-fallback interno se irraggiungibile).
    - ``"auto"``       → Ollama se rilevato, altrimenti estrattivo.
    """
    backend = backend.lower()
    if backend == "extractive":
        return ExtractiveGenerator()
    if backend == "ollama":
        return OllamaGenerator(model=model, host=host)
    if backend == "auto":
        if detect_ollama(host=host):
            return OllamaGenerator(model=model, host=host)
        return ExtractiveGenerator()
    raise ValueError(f"Backend sconosciuto: {backend!r}")
