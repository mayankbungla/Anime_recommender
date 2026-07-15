"""
Day 8 — Load the raw offline dataset (anime.csv, rating.csv) into dataframes.

This is deliberately separate from jikan_client.py: jikan_client.py fetches
live per-request metadata for the UI, while this module loads the static
bulk dataset that Weeks 3-6 will train and evaluate models against.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def load_raw_anime(path: Path | str = RAW_DIR / "anime.csv") -> pd.DataFrame:
    """Load anime.csv: one row per anime (anime_id, name, genre, type,
    episodes, rating, members)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_dataset.py first, or "
            "place anime.csv there manually — see that script's docstring."
        )
    return pd.read_csv(path)


def load_raw_ratings(path: Path | str = RAW_DIR / "rating.csv") -> pd.DataFrame:
    """Load rating.csv: one row per (user_id, anime_id, rating).
    rating == -1 means the user watched it but didn't score it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_dataset.py first, or "
            "place rating.csv there manually — see that script's docstring."
        )
    return pd.read_csv(path)


def load_raw_dataset(raw_dir: Path | str = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: returns (anime_df, ratings_df)."""
    raw_dir = Path(raw_dir)
    return load_raw_anime(raw_dir / "anime.csv"), load_raw_ratings(raw_dir / "rating.csv")


if __name__ == "__main__":
    anime_df, ratings_df = load_raw_dataset()
    print(f"anime.csv:  {len(anime_df):,} rows, columns: {list(anime_df.columns)}")
    print(f"rating.csv: {len(ratings_df):,} rows, columns: {list(ratings_df.columns)}")
