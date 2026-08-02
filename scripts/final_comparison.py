"""
Pulls the CF, content, and best-tuned hybrid results into one table and
chart, so the whole Week 4/5 comparison lives in one place instead of
two separate CSVs from different scripts.

USAGE
    python scripts/final_comparison.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_CSV = ROOT / "reports" / "model_comparison.csv"
HYBRID_CSV = ROOT / "reports" / "hybrid_weight_tuning.csv"
OUT_CSV = ROOT / "reports" / "final_comparison.csv"
OUT_PNG = ROOT / "reports" / "figures" / "final_comparison.png"

METRICS = ["precision", "recall", "ndcg", "coverage"]


def main():
    baseline = pd.read_csv(MODEL_CSV)
    hybrid_runs = pd.read_csv(HYBRID_CSV).sort_values("ndcg", ascending=False)
    best_hybrid = hybrid_runs.iloc[0]

    combined = pd.concat([
        baseline,
        pd.DataFrame([{"model": "hybrid", **best_hybrid[METRICS].to_dict()}]),
    ], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)

    print(combined.to_string(index=False))

    cf = combined[combined.model == "cf"].iloc[0]
    content = combined[combined.model == "content"].iloc[0]
    hyb = combined[combined.model == "hybrid"].iloc[0]

    beats_both = all(hyb[m] > max(cf[m], content[m]) for m in ["precision", "recall", "ndcg"])
    print(f"\nHybrid beats both standalone models on precision/recall/ndcg: {beats_both}")
    print(f"Hybrid coverage ({hyb['coverage']:.4f}) vs cf ({cf['coverage']:.4f}) "
          f"vs content ({content['coverage']:.4f})")

    x = np.arange(len(METRICS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, cf[METRICS], width, label="CF", color="#667eea")
    ax.bar(x, content[METRICS], width, label="Content", color="#764ba2")
    ax.bar(x + width, hyb[METRICS], width, label="Hybrid", color="#f093fb")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in METRICS])
    ax.set_ylabel("Score")
    ax.set_title("CF vs Content vs Hybrid")
    ax.legend()
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved {OUT_CSV}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
