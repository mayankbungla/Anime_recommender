"""
Day 12 — EDA pass on the cleaned data.

Loads the Parquet splits from Day 11, recombines train/val/test back into
one dataframe (splitting only matters for modeling, not for describing the
overall cleaned dataset), and produces:
  1. Rating distribution
  2. Genre distribution (top genres)
  3. Most-rated anime (top titles by number of ratings)
  4. User-item matrix sparsity

Charts are saved to reports/figures/. Run this after scripts/prepare_data.py.

USAGE
-----
    python scripts/eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def load_full_ratings() -> pd.DataFrame:
    """Recombine the three splits into one dataframe for describing the
    overall cleaned dataset (splitting is only relevant for modeling)."""
    parts = [
        pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet"),
        pd.read_parquet(PROCESSED_DIR / "ratings_val.parquet"),
        pd.read_parquet(PROCESSED_DIR / "ratings_test.parquet"),
    ]
    return pd.concat(parts, ignore_index=True)


def plot_rating_distribution(ratings: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#667eea")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating (1-10)")
    ax.set_ylabel("Number of ratings")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_genre_distribution(anime: pd.DataFrame, out_path: Path, top_n: int = 15):
    # genre column is comma-separated, e.g. "Action, Adventure, Comedy"
    genre_counts = (
        anime["genre"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    genre_counts.sort_values().plot(kind="barh", ax=ax, color="#764ba2")
    ax.set_title(f"Top {top_n} Genres by Number of Anime")
    ax.set_xlabel("Number of anime")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_most_rated_anime(ratings: pd.DataFrame, anime: pd.DataFrame, out_path: Path, top_n: int = 15):
    top_ids = ratings["anime_id"].value_counts().head(top_n)
    titles = anime.set_index("anime_id").loc[top_ids.index, "name"]
    fig, ax = plt.subplots(figsize=(8, 6))
    pd.Series(top_ids.values, index=titles.values).sort_values().plot(kind="barh", ax=ax, color="#667eea")
    ax.set_title(f"Top {top_n} Most-Rated Anime")
    ax.set_xlabel("Number of ratings")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sparsity(ratings: pd.DataFrame, out_path: Path):
    n_users = ratings["user_id"].nunique()
    n_anime = ratings["anime_id"].nunique()
    n_ratings = len(ratings)
    possible = n_users * n_anime
    density_pct = 100 * n_ratings / possible
    sparsity_pct = 100 - density_pct

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [density_pct, sparsity_pct],
        labels=[f"Filled\n{density_pct:.3f}%", f"Empty\n{sparsity_pct:.3f}%"],
        colors=["#667eea", "#2a2a3a"],
        autopct=None,
    )
    ax.set_title(f"User-Item Matrix Sparsity\n({n_users:,} users x {n_anime:,} anime)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {"n_users": n_users, "n_anime": n_anime, "n_ratings": n_ratings, "density_pct": density_pct}


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed data...")
    ratings = load_full_ratings()
    anime = pd.read_parquet(PROCESSED_DIR / "anime_clean.parquet")

    print("Plotting rating distribution...")
    plot_rating_distribution(ratings, FIGURES_DIR / "rating_distribution.png")

    print("Plotting genre distribution...")
    plot_genre_distribution(anime, FIGURES_DIR / "genre_distribution.png")

    print("Plotting most-rated anime...")
    plot_most_rated_anime(ratings, anime, FIGURES_DIR / "most_rated_anime.png")

    print("Plotting user-item matrix sparsity...")
    stats = plot_sparsity(ratings, FIGURES_DIR / "sparsity.png")

    print(f"\nSaved 4 charts to {FIGURES_DIR}/")
    print("\nSummary stats (useful for the Day 13 write-up):")
    print(f"  users:        {stats['n_users']:,}")
    print(f"  anime:        {stats['n_anime']:,}")
    print(f"  ratings:      {stats['n_ratings']:,}")
    print(f"  matrix density: {stats['density_pct']:.4f}%  (i.e. {100 - stats['density_pct']:.4f}% sparse)")
    print(f"  avg rating:   {ratings['rating'].mean():.2f}")
    print(f"  median rating count per anime: {ratings['anime_id'].value_counts().median():.0f}")
    print(f"  median rating count per user:  {ratings['user_id'].value_counts().median():.0f}")


if __name__ == "__main__":
    main()
