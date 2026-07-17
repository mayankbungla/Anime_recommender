"""
Days 8-11 pipeline, end to end:
  load raw dataset -> clean -> split by user -> save train/val/test

Day 11 update: saved as Parquet now (was CSV through Day 10). Parquet keeps
dtypes exact (no more int64 ids silently round-tripping as strings/floats),
compresses much smaller than CSV, and loads faster for the training script.

USAGE
-----
    python scripts/prepare_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from anime_recommender.data.dataset import load_raw_dataset
from anime_recommender.data.cleaning import clean_anime_metadata, clean_ratings
from anime_recommender.data.split import train_val_test_split_by_user

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main():
    print("Loading raw dataset...")
    anime_df, ratings_df = load_raw_dataset()

    print("Cleaning...")
    anime_clean = clean_anime_metadata(anime_df)
    ratings_clean = clean_ratings(ratings_df, anime_clean)
    print(f"  ratings: {len(ratings_df):,} -> {len(ratings_clean):,} rows")
    print(f"  users:   {ratings_clean['user_id'].nunique():,}")
    print(f"  anime:   {ratings_clean['anime_id'].nunique():,}")

    print("Splitting by user (train/val/test)...")
    train_df, val_df, test_df = train_val_test_split_by_user(ratings_clean)
    print(f"  train: {len(train_df):,} rows")
    print(f"  val:   {len(val_df):,} rows")
    print(f"  test:  {len(test_df):,} rows")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "anime_clean.parquet": anime_clean,
        "ratings_train.parquet": train_df,
        "ratings_val.parquet": val_df,
        "ratings_test.parquet": test_df,
    }
    for fname, df in outputs.items():
        df.to_parquet(PROCESSED_DIR / fname, index=False)

    print(f"Saved to {PROCESSED_DIR}/")

    # Verify each file round-trips correctly via pandas.read_parquet before
    # trusting it downstream (Day 11 explicitly calls for this check).
    print("Verifying Parquet round-trip...")
    for fname, original_df in outputs.items():
        reloaded = pd.read_parquet(PROCESSED_DIR / fname)
        assert reloaded.shape == original_df.shape, f"{fname}: shape mismatch after reload"
        assert list(reloaded.columns) == list(original_df.columns), f"{fname}: column mismatch after reload"
        print(f"  {fname}: OK  ({reloaded.shape[0]:,} rows, {reloaded.shape[1]} cols)")

    # Clean up the old CSVs from Days 8-10 now that Parquet is the source
    # of truth, so nothing downstream can accidentally read stale CSVs.
    old_csvs = ["anime_clean.csv", "ratings_train.csv", "ratings_val.csv", "ratings_test.csv"]
    for fname in old_csvs:
        old_path = PROCESSED_DIR / fname
        if old_path.exists():
            old_path.unlink()
            print(f"  removed stale {fname}")


if __name__ == "__main__":
    main()
