"""
Day 18 + 19: build the content model.

  1. join the cleaned catalogue with the fetched synopses
  2. embed each anime's text with sentence-transformers (Day 18)
  3. fit the NearestNeighbors retriever and save everything (Day 19)

Output lands in models/content/:
  synopsis_embeddings.npy   the float32 embedding matrix
  content_catalog.parquet   the catalogue rows aligned to those vectors

Run scripts/fetch_synopses.py first. If the synopsis file is missing the
build still runs, falling back to title + genres for every anime, so you
can smoke-test the pipeline before the (slow) fetch finishes.

USAGE
    python scripts/build_content_model.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_recommender.features.content_model import (  # noqa: E402
    CONTENT_DIR,
    ContentRecommender,
    build_catalog_text,
    embed_texts,
)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
CATALOG_PATH = PROCESSED_DIR / "anime_clean.parquet"
SYNOPSES_PATH = PROCESSED_DIR / "anime_synopses.parquet"


def main():
    catalog = pd.read_parquet(CATALOG_PATH)

    if SYNOPSES_PATH.exists():
        synopses = pd.read_parquet(SYNOPSES_PATH)
        catalog = catalog.merge(synopses, on="anime_id", how="left")
        n_syn = catalog["synopsis"].fillna("").str.len().ge(40).sum()
        print(f"{n_syn:,}/{len(catalog):,} anime have a usable synopsis.")
    else:
        print("No synopsis file found. Falling back to title + genres for all.")
        catalog["synopsis"] = ""

    texts = [build_catalog_text(row) for _, row in catalog.iterrows()]

    print(f"Embedding {len(texts):,} anime (first run downloads the model)...")
    embeddings = embed_texts(texts)
    print(f"Embeddings: {embeddings.shape}")

    # keep only the columns the retriever and downstream code need
    keep = catalog[["anime_id", "name", "genre"]].reset_index(drop=True)
    rec = ContentRecommender(embeddings, keep)
    rec.save()

    print(f"Saved content model to {CONTENT_DIR}")


if __name__ == "__main__":
    main()
