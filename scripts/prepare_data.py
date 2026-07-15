"""
Days 8-10 pipeline, end to end:
  load raw dataset -> clean -> split by user -> save train/val/test

Saved as CSV for now — Day 11 swaps this over to Parquet, so don't be
surprised that this script gets a one-line diff later.

USAGE
-----
    python scripts/prepare_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    anime_clean.to_csv(PROCESSED_DIR / "anime_clean.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "ratings_train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "ratings_val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "ratings_test.csv", index=False)

    print(f"Saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
