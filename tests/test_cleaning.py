"""
Tests for src/anime_recommender/data/cleaning.py, checked against small
hand-built dataframes where the expected outcome is worked out by hand,
not just re-running the same code twice.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_recommender.data.cleaning import clean_ratings, clean_anime_metadata


def _anime(ids):
    return pd.DataFrame({"anime_id": ids})


def test_unscored_ratings_dropped_by_default():
    ratings = pd.DataFrame({
        "user_id": [1, 1, 2],
        "anime_id": [10, 20, 10],
        "rating": [8, -1, 7],
    })
    cleaned = clean_ratings(ratings, _anime([10, 20]), min_user_ratings=1, min_anime_ratings=1)
    assert -1 not in cleaned["rating"].values
    assert len(cleaned) == 2


def test_unscored_ratings_kept_when_explicit_only_false():
    ratings = pd.DataFrame({
        "user_id": [1, 1, 2],
        "anime_id": [10, 20, 10],
        "rating": [8, -1, 7],
    })
    cleaned = clean_ratings(ratings, _anime([10, 20]), min_user_ratings=1,
                             min_anime_ratings=1, explicit_only=False)
    assert -1 in cleaned["rating"].values
    assert len(cleaned) == 3


def test_exact_duplicates_collapsed_to_one_row():
    ratings = pd.DataFrame({
        "user_id": [1, 1, 2],
        "anime_id": [10, 10, 20],
        "rating": [8, 8, 5],
    })
    cleaned = clean_ratings(ratings, _anime([10, 20]), min_user_ratings=1, min_anime_ratings=1)
    assert len(cleaned) == 2
    assert cleaned[(cleaned.user_id == 1) & (cleaned.anime_id == 10)]["rating"].iloc[0] == 8


def test_conflicting_duplicate_keeps_last_row():
    # same (user, anime) rated twice with different scores - last row wins
    ratings = pd.DataFrame({
        "user_id": [1, 1],
        "anime_id": [10, 10],
        "rating": [5, 9],
    })
    cleaned = clean_ratings(ratings, _anime([10]), min_user_ratings=1, min_anime_ratings=1)
    assert len(cleaned) == 1
    assert cleaned["rating"].iloc[0] == 9


def test_ratings_for_unknown_anime_dropped():
    ratings = pd.DataFrame({
        "user_id": [1, 1],
        "anime_id": [10, 999],  # 999 has no metadata row
        "rating": [8, 7],
    })
    cleaned = clean_ratings(ratings, _anime([10]), min_user_ratings=1, min_anime_ratings=1)
    assert set(cleaned["anime_id"]) == {10}


def test_sparse_users_and_anime_removed_iteratively():
    # user 2 only rates once (fails min_user_ratings=3)
    # anime 30 is only ever rated once total (fails min_anime_ratings=2)
    # once anime 30 is dropped, user 1 is left with only 2 ratings,
    # which then also fails min_user_ratings=3 - everything ends up empty
    ratings = pd.DataFrame({
        "user_id": [1, 1, 1, 2],
        "anime_id": [10, 20, 30, 10],
        "rating": [8, 7, 6, 9],
    })
    cleaned = clean_ratings(ratings, _anime([10, 20, 30]), min_user_ratings=3, min_anime_ratings=2)
    assert cleaned.empty


def test_metadata_missing_genre_and_type_filled_unknown():
    anime = pd.DataFrame({
        "anime_id": [1, 2],
        "genre": ["Action", None],
        "type": [None, "TV"],
        "episodes": ["12", "24"],
    })
    cleaned = clean_anime_metadata(anime)
    assert cleaned.loc[1, "genre"] == "Unknown"
    assert cleaned.loc[0, "type"] == "Unknown"


def test_metadata_episodes_coerced_to_numeric():
    anime = pd.DataFrame({
        "anime_id": [1, 2],
        "genre": ["Action", "Comedy"],
        "type": ["TV", "TV"],
        "episodes": ["12", "Unknown"],
    })
    cleaned = clean_anime_metadata(anime)
    assert cleaned.loc[0, "episodes"] == 12.0
    assert pd.isna(cleaned.loc[1, "episodes"])
