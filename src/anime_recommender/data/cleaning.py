"""
Day 9 — Clean the raw ratings/anime data.

Cleaning decisions (documented here, and mirrored into the README on Day 55):

1. UNSCORED WATCHES (rating == -1): the source dataset uses -1 to mean
   "user watched this but never gave it a score", not "user rated it -1".
   We split these out rather than dropping them silently — they're useful
   later as implicit-feedback signal (Day 15+, ALS-style models), but they
   would corrupt anything that treats `rating` as an explicit 1-10 score
   (e.g. SVD, or any average-rating computation). `clean_ratings()` returns
   only the explicitly-scored rows by default; unscored rows are exposed
   via `explicit_only=False` for anyone building an implicit model later.

2. EXACT DUPLICATES: a handful of (user_id, anime_id) pairs appear more
   than once with the same rating. These are almost certainly export
   duplicates rather than a user re-rating something, so we drop repeats
   and keep the first occurrence.

3. CONFLICTING DUPLICATES: a smaller number of (user_id, anime_id) pairs
   appear more than once with *different* ratings. Rather than guessing
   intent, we keep the most recent-looking (last) row for that pair — the
   dataset has no timestamp, so "last row in file" is our best proxy for
   "most recent". This is a judgment call, so it's called out explicitly
   here rather than being buried in a groupby.

4. SPARSE USERS/ANIME: users with very few ratings and anime with very
   few ratings don't give a collaborative-filtering model enough signal,
   and inflate the user-item matrix without adding useful structure. We
   drop users with fewer than `min_user_ratings` ratings and anime with
   fewer than `min_anime_ratings` ratings. This is done in two passes
   (drop anime, then re-check users, since removing anime can push some
   users below the threshold too) and repeated until stable.

5. MISSING ANIME METADATA: some anime_id values in rating.csv don't have
   a matching row in anime.csv (delisted/renamed titles). Ratings for
   anime we have no metadata for are dropped, since we can't display or
   feature-ize them anyway.

6. MISSING GENRE/TYPE: anime.csv has some blank `genre`/`type` fields.
   We fill these with the literal string "Unknown" rather than dropping
   the anime — we still want it available for CF, just not for
   genre-based content filtering.
"""

import pandas as pd


def clean_ratings(
    ratings_df: pd.DataFrame,
    anime_df: pd.DataFrame,
    min_user_ratings: int = 5,
    min_anime_ratings: int = 5,
    explicit_only: bool = True,
) -> pd.DataFrame:
    """Apply decisions 1-5 above. Returns a cleaned ratings dataframe.

    Set explicit_only=False to keep the -1 (unscored/implicit) rows too.
    """
    df = ratings_df.copy()

    # (1) split out unscored watches
    if explicit_only:
        df = df[df["rating"] != -1]

    # (2) + (3) de-duplicate (user_id, anime_id), keeping the last row
    # for conflicting duplicates and dropping exact repeats along the way
    df = df.drop_duplicates(subset=["user_id", "anime_id", "rating"])
    df = df.drop_duplicates(subset=["user_id", "anime_id"], keep="last")

    # (5) drop ratings for anime we have no metadata for
    known_anime_ids = set(anime_df["anime_id"])
    df = df[df["anime_id"].isin(known_anime_ids)]

    # (4) iteratively drop sparse users/anime until stable
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        anime_counts = df["anime_id"].value_counts()
        df = df[df["anime_id"].isin(anime_counts[anime_counts >= min_anime_ratings].index)]
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_user_ratings].index)]

    return df.reset_index(drop=True)


def clean_anime_metadata(anime_df: pd.DataFrame) -> pd.DataFrame:
    """Apply decision 6 above. Returns a cleaned anime metadata dataframe."""
    df = anime_df.copy()
    df["genre"] = df["genre"].fillna("Unknown")
    df["type"] = df["type"].fillna("Unknown")
    # episodes is sometimes "Unknown" as a literal string in this dataset;
    # coerce to numeric, leaving true unknowns as NaN rather than guessing
    df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
    return df


if __name__ == "__main__":
    from anime_recommender.data.dataset import load_raw_dataset

    anime_df, ratings_df = load_raw_dataset()
    anime_clean = clean_anime_metadata(anime_df)
    ratings_clean = clean_ratings(ratings_df, anime_clean)

    print(f"anime:   {len(anime_df):,} -> {len(anime_clean):,} rows")
    print(f"ratings: {len(ratings_df):,} -> {len(ratings_clean):,} rows")
    print(f"unique users:  {ratings_clean['user_id'].nunique():,}")
    print(f"unique anime:  {ratings_clean['anime_id'].nunique():,}")
