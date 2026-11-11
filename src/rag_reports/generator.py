"""Answer generation from the retrieved chunks.

Two backends:

- **Ollama** (``OllamaGenerator``): uses a local LLM via ``http://localhost:11434``
  if the service responds. Detection is *robust*: a missing service does NOT raise
  exceptions (try/except around requests, short timeout).
- **Extractive** (``ExtractiveGenerator``): **deterministic** fallback that uses
  neither the network nor an LLM. It builds the answer by concatenating the retrieved
  source passages and adds the ``[doc:ID]`` citations. This is the backend used by the
  tests.

Both follow the same contract: given a question and the source chunks, they return an
answer string that cites its sources.
"""

from __future__ import annotations

import os

import requests

from .retriever import RetrievalResult

# Standard abstention message (hallucination check).
ABSTAIN_MESSAGE = "The information is not present in the documents."

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def detect_ollama(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 0.5) -> bool:
    """True if an Ollama service responds at ``host``. Never raises."""
    url = host.rstrip("/") + "/api/tags"
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except (requests.exceptions.RequestException, ValueError, OSError):
        return False


def _format_citations(results: list[RetrievalResult]) -> str:
    """Ordered, de-duplicated list of the cited sources, as ``[doc:ID]``."""
    seen: list[str] = []
    for r in results:
        if r.chunk.doc_id not in seen:
            seen.append(r.chunk.doc_id)
    return " ".join(f"[doc:{doc_id}]" for doc_id in seen)


class ExtractiveGenerator:
    """Deterministic extractive fallback (no network, no LLM).

    Builds the answer from the retrieved passages and cites ``[doc:ID]``. Deterministic:
    same input → same output. This is the default backend when Ollama is not available.
    """

    name = "extractive"

    def generate(self, question: str, results: list[RetrievalResult]) -> str:
        if not results:
            return ABSTAIN_MESSAGE
        lines = [
            "Based on the retrieved documents, the relevant passages are:",
            "",
        ]
        for r in results:
            lines.append(f"- {r.chunk.text} [doc:{r.chunk.doc_id}]")
        lines.append("")
        lines.append(f"Sources: {_format_citations(results)}")
        return "\n".join(lines)


class OllamaGenerator:
    """Ollama backend. Builds a grounded prompt and calls the local API.

    On a network error or when the service is missing, it falls back to the extractive
    generator, so the pipeline never fails.
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
            "You answer ONLY on the basis of the documents provided.\n"
            "Cite the sources in square brackets using the format [doc:ID].\n"
            "Answer in the same language as the question.\n"
            "If the information is not present in the documents, answer exactly: "
            f"\"{ABSTAIN_MESSAGE}\".\n\n"
            f"DOCUMENTS:\n{context}\n\n"
            f"QUESTION: {question}\n\nANSWER:"
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
            # service missing / network error / invalid JSON → robust fallback
            return self._fallback.generate(question, results)


def make_generator(backend: str = "auto", model: str = "llama3.2", host: str = DEFAULT_OLLAMA_HOST):
    """Create the generator.

    - ``"extractive"`` → always the extractive fallback (safe/offline default).
    - ``"ollama"``     → Ollama backend (with internal fallback if unreachable).
    - ``"auto"``       → Ollama if detected, otherwise extractive.
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
    raise ValueError(f"Unknown backend: {backend!r}")
