"""Streamlit interface: RAG chat over the synthetic radiology reports.

Question → GROUNDED answer with cited source passages [doc:ID]. Selector for the
generation backend (Ollama auto / extractive). Works WITHOUT Ollama: the default falls
back to the extractive backend if the service is not detected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allows `streamlit run app/main.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_reports.corpus import load_sample_corpus  # noqa: E402
from rag_reports.generator import detect_ollama  # noqa: E402
from rag_reports.rag import RagPipeline  # noqa: E402

st.set_page_config(page_title="RAG radiology reports", page_icon="🩻", layout="centered")

st.title("RAG over radiology reports")
st.caption(
    "TF-IDF retrieval (offline) + grounded generation with [doc:ID] citations. "
    "SYNTHETIC corpus in Italian: invented reports, NOT real patient data. "
    "Ask questions in Italian."
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
    st.header("Settings")
    ollama_up = detect_ollama()
    st.write(f"Ollama service: {'detected ✅' if ollama_up else 'not detected ❌'}")
    backend_label = st.selectbox(
        "Generation backend",
        options=["Extractive (offline)", "Ollama (auto)"],
        index=0 if not ollama_up else 1,
        help="Extractive needs no LLM and no network. Ollama needs a local model.",
    )
    backend = "extractive" if backend_label.startswith("Extractive") else "auto"
    top_k = st.slider("Number of passages (top-k)", 1, 5, 3)
    min_score = st.slider(
        "Abstention threshold",
        0.0,
        0.5,
        0.05,
        0.01,
        help="Below this similarity the system abstains.",
    )
    st.markdown("---")
    st.caption(f"Documents in the corpus: {len(_load_docs())}")

pipeline = _build_pipeline(backend, min_score, top_k)

st.subheader("Ask the reports a question")
question = st.text_input(
    "Question",
    placeholder="E.g.: Ci sono segni di polmonite? È presente uno pneumotorace?",
)

if st.button("Ask", type="primary") and question.strip():
    ans = pipeline.answer(question)
    st.markdown("### Answer")
    if ans.abstained:
        st.warning(ans.text)
    else:
        st.write(ans.text)
        st.caption(f"Backend: {ans.backend} · Sources: {', '.join(ans.source_ids)}")
        with st.expander("Retrieved source passages", expanded=True):
            for r in ans.sources:
                st.markdown(
                    f"**[doc:{r.chunk.doc_id}]** "
                    f"(score {r.score:.3f}) — *{r.chunk.meta.get('exam', '')}*"
                )
                st.write(r.chunk.text)

st.markdown("---")
st.caption(
    "Reference: Lewis et al., Retrieval-Augmented Generation, NeurIPS 2020 "
    "(arXiv:2005.11401). Real-corpus extension: Open-i (openi.nlm.nih.gov)."
)
