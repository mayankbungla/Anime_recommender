"""
Precomputes per-anime popularity scores and saves them to a small file,
so the FastAPI backend can read that instead of loading the full
ratings_train.parquet (millions of rows) just to compute this once at
startup. Rerun this whenever the model is retrained on new data.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anime_recommender.models.hybrid import build_popularity_scores  # noqa: E402

TRAIN_PATH = ROOT / "data" / "processed" / "ratings_train.parquet"
OUT_PATH = ROOT / "models" / "popularity_scores.parquet"


def main():
    train = pd.read_parquet(TRAIN_PATH)
    popularity = build_popularity_scores(train)

    out = pd.DataFrame({
        "anime_id": list(popularity.keys()),
        "popularity": list(popularity.values()),
    })
    out.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(out):,} popularity scores -> {OUT_PATH}")


if __name__ == "__main__":
    main()
