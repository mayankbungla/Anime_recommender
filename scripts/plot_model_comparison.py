"""
Bar chart comparing the CF and content models across every metric in
reports/model_comparison.csv, side by side. Makes the tradeoffs from the
evaluation actually visible instead of buried in a table.

USAGE
    python scripts/plot_model_comparison.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "reports" / "model_comparison.csv"
OUT_PATH = ROOT / "reports" / "figures" / "model_comparison.png"

METRICS = ["precision", "recall", "ndcg", "coverage"]


def main():
    df = pd.read_csv(CSV_PATH).set_index("model")

    x = np.arange(len(METRICS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, df.loc["cf", METRICS], width, label="CF", color="#667eea")
    ax.bar(x + width / 2, df.loc["content", METRICS], width, label="Content", color="#764ba2")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in METRICS])
    ax.set_ylabel("Score")
    ax.set_title("CF vs Content: precision, recall, ndcg, coverage")
    ax.legend()
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
