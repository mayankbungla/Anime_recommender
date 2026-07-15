"""
Day 8 — Download the offline ratings dataset.

Source: Kaggle "Anime Recommendations Database"
https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database

This dataset ships two files:
  - anime.csv   (anime_id, name, genre, type, episodes, rating, members)
  - rating.csv  (user_id, anime_id, rating)  -- rating == -1 means "watched,
                 no score given", which we'll deal with explicitly in the
                 cleaning step (Day 9), not silently drop here.

USAGE
-----
Requires a Kaggle account + API token (kaggle.json). One-time setup:
    1. Go to https://www.kaggle.com/settings/account -> "Create New Token"
    2. This downloads kaggle.json. Place it at ~/.kaggle/kaggle.json
       (chmod 600 ~/.kaggle/kaggle.json)
    3. pip install kagglehub

Then run:
    python scripts/download_dataset.py

If you'd rather not deal with API credentials, download the two CSVs
manually from the Kaggle page above and drop them directly into
data/raw/anime.csv and data/raw/rating.csv — the loader in
src/anime_recommender/data/dataset.py doesn't care how they got there.
"""

import shutil
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
KAGGLE_DATASET = "CooperUnion/anime-recommendations-database"


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit(
            "kagglehub is not installed. Run `pip install kagglehub`, or "
            "skip this script and manually place anime.csv / rating.csv "
            f"into {RAW_DIR}"
        ) from e

    print(f"Downloading {KAGGLE_DATASET} via kagglehub...")
    dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"Downloaded to cache at: {dataset_path}")

    for fname in ("anime.csv", "rating.csv"):
        src = dataset_path / fname
        if not src.exists():
            print(f"  ! WARNING: expected {fname} not found at {src}")
            continue
        dst = RAW_DIR / fname
        shutil.copy(src, dst)
        print(f"  copied -> {dst}")

    print("Done. Files are in data/raw/.")


if __name__ == "__main__":
    download()
