"""Calcola le metriche di retrieval e anti-allucinazione → results/.

Produce:
- ``results/metrics.json``: hit-rate@k del retrieval sulle query curate, esito del
  test anti-allucinazione, e un confronto "retrieval on vs off" (con vs senza
  retrieval) sull'astensione.
- ``results/hit_rate.png``: figura a barre dell'hit-rate@k.

Tutto offline e deterministico (backend di generazione: estrattivo).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Esecuzione diretta senza install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_reports.corpus import load_sample_corpus  # noqa: E402
from rag_reports.evaluation import (  # noqa: E402
    CURATED_QUERIES,
    evaluate_abstention,
    evaluate_retrieval,
)
from rag_reports.rag import RagPipeline  # noqa: E402
from rag_reports.retriever import build_retriever  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _plot_hit_rate(hit_rate: dict[int, float], out_path: Path) -> None:
    ks = sorted(hit_rate)
    vals = [hit_rate[k] for k in ks]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar([f"@{k}" for k in ks], vals, color="#2b7bba")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Hit-rate")
    ax.set_xlabel("Top-k")
    ax.set_title("Retrieval TF-IDF — Hit-rate@k (corpus sintetico)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_sample_corpus()

    # --- Retrieval hit-rate@k -------------------------------------------------
    retriever = build_retriever(docs, dense=False)
    retr = evaluate_retrieval(retriever, k_values=(1, 3, 5))

    # --- Anti-allucinazione: retrieval ON vs OFF ------------------------------
    # ON: pipeline normale con retrieval → deve astenersi sulle query fuori-corpus.
    pipe_on = RagPipeline.from_documents(docs, backend="extractive")
    abst_on = evaluate_abstention(pipe_on)

    # OFF (ablazione): soglia anti-allucinazione disattivata (min_score=0). Senza il
    # gate, il sistema risponderebbe SEMPRE → astensione attesa ~0 → mostra l'utilità
    # del controllo anti-allucinazione.
    pipe_off = RagPipeline.from_documents(docs, backend="extractive", min_score=0.0)
    abst_off = evaluate_abstention(pipe_off)

    metrics = {
        "corpus": {
            "n_documents": len(docs),
            "synthetic": True,
            "note": "Referti generati artificialmente (NON dati reali di pazienti).",
        },
        "retrieval": {
            "backend": "tfidf",
            "n_curated_queries": retr.n_queries,
            "hit_rate_at_k": {str(k): round(v, 4) for k, v in retr.hit_rate_at_k.items()},
            "per_query": retr.per_query,
        },
        "anti_hallucination": {
            "n_out_of_corpus_queries": abst_on.n_queries,
            "retrieval_on": {
                "min_score": pipe_on.min_score,
                "correct_abstentions": abst_on.n_correct_abstentions,
                "abstention_rate": round(abst_on.abstention_rate, 4),
            },
            "retrieval_off_ablation": {
                "min_score": pipe_off.min_score,
                "correct_abstentions": abst_off.n_correct_abstentions,
                "abstention_rate": round(abst_off.abstention_rate, 4),
                "note": "Gate anti-allucinazione disattivato (min_score=0): astensione attesa ~0.",
            },
        },
    }

    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    fig_path = RESULTS_DIR / "hit_rate.png"
    _plot_hit_rate(retr.hit_rate_at_k, fig_path)

    # --- Riepilogo a console --------------------------------------------------
    print("=== Valutazione RAG (corpus sintetico) ===")
    print(f"Documenti: {len(docs)} | Query curate: {retr.n_queries} (= {len(CURATED_QUERIES)})")
    for k in sorted(retr.hit_rate_at_k):
        print(f"  Hit-rate@{k}: {retr.hit_rate_at_k[k]:.3f}")
    print(
        f"Anti-allucinazione (retrieval ON): "
        f"{abst_on.n_correct_abstentions}/{abst_on.n_queries} astensioni corrette "
        f"({abst_on.abstention_rate:.3f})"
    )
    print(
        f"Ablazione (gate OFF): "
        f"{abst_off.n_correct_abstentions}/{abst_off.n_queries} astensioni "
        f"({abst_off.abstention_rate:.3f})"
    )
    print(f"\nScritti: {metrics_path}")
    print(f"         {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
