"""
Day 10 — Split cleaned ratings into train/validation/test, by user.

WHY NOT A PLAIN RANDOM ROW SPLIT:
A naive `train_test_split(ratings_df)` shuffles rows globally. For a
collaborative-filtering model that's a problem in both directions:
  - a user can end up with ALL of their ratings in train and none in
    test (or vice versa), so we can't fairly evaluate recommendations
    for them, or we're "testing" on a user the model never learned about
    for reasons unrelated to genuine cold-start
  - it's easy to end up leaking a large chunk of a low-activity user's
    signal into test, starving train of exactly the users hardest to
    model well

Instead, we split *within* each user's own ratings: every user who has
enough ratings to be split at all contributes a proportional slice to
train, validation, and test. This keeps every user represented in train
(so CF has something to learn per user) while still holding out enough
of their real ratings to evaluate against. True cold-start behaviour
(a user/anime with too few ratings to split meaningfully) is handled
separately by the cold-start fallback rule (Day 29) — this split isn't
trying to simulate that case, just to avoid leaking the same
interactions across splits.

Users with too few ratings to give every split at least one row keep
all their ratings in train only (see min_ratings_for_split below).
"""

import numpy as np
import pandas as pd


def train_val_test_split_by_user(
    ratings_df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    min_ratings_for_split: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings into (train_df, val_df, test_df), splitting each
    user's ratings proportionally rather than splitting rows globally.

    Users with fewer than `min_ratings_for_split` ratings go entirely
    into train (too few rows to hold anything out meaningfully).
    """
    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []

    for _, user_rows in ratings_df.groupby("user_id"):
        n = len(user_rows)
        if n < min_ratings_for_split:
            train_parts.append(user_rows)
            continue

        shuffled = user_rows.sample(frac=1.0, random_state=rng.integers(0, 2**32 - 1))
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round(n * val_size)))
        # guard against over-allocating on very small groups
        n_test = min(n_test, n - 2)
        n_val = min(n_val, n - n_test - 1)

        test_parts.append(shuffled.iloc[:n_test])
        val_parts.append(shuffled.iloc[n_test:n_test + n_val])
        train_parts.append(shuffled.iloc[n_test + n_val:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True) if val_parts else ratings_df.iloc[0:0].copy()
    test_df = pd.concat(test_parts).reset_index(drop=True) if test_parts else ratings_df.iloc[0:0].copy()

    return train_df, val_df, test_df


if __name__ == "__main__":
    from anime_recommender.data.dataset import load_raw_dataset
    from anime_recommender.data.cleaning import clean_anime_metadata, clean_ratings

    anime_df, ratings_df = load_raw_dataset()
    anime_clean = clean_anime_metadata(anime_df)
    ratings_clean = clean_ratings(ratings_df, anime_clean)

    train_df, val_df, test_df = train_val_test_split_by_user(ratings_clean)

    print(f"train: {len(train_df):,} rows, {train_df['user_id'].nunique():,} users")
    print(f"val:   {len(val_df):,} rows, {val_df['user_id'].nunique():,} users")
    print(f"test:  {len(test_df):,} rows, {test_df['user_id'].nunique():,} users")

    leak = set(zip(train_df.user_id, train_df.anime_id)) & set(zip(test_df.user_id, test_df.anime_id))
    print(f"train/test (user_id, anime_id) overlap: {len(leak)} (should be 0)")
