"""Interfaccia Streamlit: chat RAG sui referti radiologici sintetici.

Domanda → risposta GROUNDED con passaggi-fonte citati [doc:ID]. Selettore del backend
di generazione (Ollama auto / estrattivo). Funziona SENZA Ollama: il default ricade
sul fallback estrattivo se il servizio non è rilevato.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Permette `streamlit run app/main.py` senza install del pacchetto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_reports.corpus import load_sample_corpus  # noqa: E402
from rag_reports.generator import detect_ollama  # noqa: E402
from rag_reports.rag import RagPipeline  # noqa: E402

st.set_page_config(page_title="RAG referti radiologici", page_icon="🩻", layout="centered")

st.title("RAG su referti radiologici")
st.caption(
    "Retrieval TF-IDF (offline) + generazione grounded con citazioni [doc:ID]. "
    "Corpus SINTETICO: referti generati artificialmente, NON dati reali di pazienti."
)


@st.cache_data(show_spinner=False)
def _load_docs():
    return load_sample_corpus()


@st.cache_resource(show_spinner=False)
def _build_pipeline(backend: str, min_score: float, top_k: int):
    return RagPipeline.from_documents(
        _load_docs(), backend=backend, min_score=min_score, top_k=top_k
    )


with st.sidebar:
    st.header("Impostazioni")
    ollama_up = detect_ollama()
    st.write(f"Servizio Ollama: {'rilevato ✅' if ollama_up else 'non rilevato ❌'}")
    backend_label = st.selectbox(
        "Backend di generazione",
        options=["Estrattivo (offline)", "Ollama (auto)"],
        index=0 if not ollama_up else 1,
        help="L'estrattivo non richiede LLM né rete. Ollama richiede un modello locale.",
    )
    backend = "extractive" if backend_label.startswith("Estrattivo") else "auto"
    top_k = st.slider("Numero di passaggi (top-k)", 1, 5, 3)
    min_score = st.slider(
        "Soglia anti-allucinazione",
        0.0,
        0.5,
        0.05,
        0.01,
        help="Sotto questa similarità il sistema si astiene.",
    )
    st.markdown("---")
    st.caption(f"Documenti nel corpus: {len(_load_docs())}")

pipeline = _build_pipeline(backend, min_score, top_k)

st.subheader("Fai una domanda ai referti")
question = st.text_input(
    "Domanda",
    placeholder="Es.: Ci sono segni di polmonite? È presente uno pneumotorace?",
)

if st.button("Chiedi", type="primary") and question.strip():
    ans = pipeline.answer(question)
    st.markdown("### Risposta")
    if ans.abstained:
        st.warning(ans.text)
    else:
        st.write(ans.text)
        st.caption(f"Backend: {ans.backend} · Fonti: {', '.join(ans.source_ids)}")
        with st.expander("Passaggi-fonte recuperati", expanded=True):
            for r in ans.sources:
                st.markdown(
                    f"**[doc:{r.chunk.doc_id}]** "
                    f"(score {r.score:.3f}) — *{r.chunk.meta.get('exam', '')}*"
                )
                st.write(r.chunk.text)

st.markdown("---")
st.caption(
    "Riferimento: Lewis et al., Retrieval-Augmented Generation, NeurIPS 2020 "
    "(arXiv:2005.11401). Estensione corpus reale: Open-i (openi.nlm.nih.gov)."
)
