"""
Tests for src/anime_recommender/models/predict.py (Day 12)

STATUS: scaffolded from PROJECT_BRIEF.md's spec:
  - similar_items() returns correct length and never includes the query item
  - get_user_top_n() excludes items the user already rated

Trains a tiny SVD model on synthetic data rather than using the real
committed model, so this suite is fast and independent of the real dataset.

Requires scikit-surprise (already in requirements.txt per the brief).
"""

import joblib
import pandas as pd
import pytest
from surprise import SVD, Dataset, Reader

from src.anime_recommender.models.predict import get_user_top_n, similar_items


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    # 5 users x 6 anime, dense enough that every user has some unrated items
    rows = []
    for user_id in range(1, 6):
        for anime_id in range(1, 7):
            if (user_id + anime_id) % 3 != 0:  # leave gaps -> unrated items exist
                rows.append({"user_id": user_id, "anime_id": anime_id,
                             "rating": ((user_id + anime_id) % 10) + 1})
    df = pd.DataFrame(rows)

    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(df[["user_id", "anime_id", "rating"]], reader)
    trainset = data.build_full_trainset()

    algo = SVD(n_factors=5, n_epochs=5, random_state=42)
    algo.fit(trainset)

    model_path = tmp_path_factory.mktemp("model") / "tiny_svd.pkl"
    joblib.dump(algo, model_path)
    return str(model_path), df


def test_similar_items_correct_length_and_excludes_query(tiny_model):
    model_path, df = tiny_model
    result = similar_items(anime_id=1, k=3, model_path=model_path)
    assert len(result) <= 3
    assert 1 not in result["anime_id"].values


def test_similar_items_unknown_id_raises(tiny_model):
    model_path, _ = tiny_model
    with pytest.raises(ValueError):
        similar_items(anime_id=99999, k=3, model_path=model_path)


def test_get_user_top_n_excludes_already_rated(tiny_model):
    model_path, df = tiny_model
    user_id = 1
    already_rated = set(df.loc[df["user_id"] == user_id, "anime_id"])

    result = get_user_top_n(user_id=user_id, k=10, model_path=model_path)

    recommended_ids = set(result["anime_id"])
    assert recommended_ids.isdisjoint(already_rated)


def test_get_user_top_n_unknown_user_raises(tiny_model):
    model_path, _ = tiny_model
    with pytest.raises(ValueError):
        get_user_top_n(user_id=99999, k=5, model_path=model_path)


def test_get_user_top_n_sorted_descending(tiny_model):
    model_path, _ = tiny_model
    result = get_user_top_n(user_id=2, k=10, model_path=model_path)
    ratings = result["predicted_rating"].tolist()
    assert ratings == sorted(ratings, reverse=True)
