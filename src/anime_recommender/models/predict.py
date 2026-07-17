"""
Inference helpers for the trained SVD collaborative-filtering model.

STATUS: drafted from PROJECT_BRIEF.md without sight of your actual
dataset.py / cleaning.py. Two spots are marked TODO — swap in your
real column names / loader calls there and this should run as-is.

Depends on:
    - models/svd_cf_model.pkl  (trained via scripts/train_cf.py, loaded with joblib)
    - the anime metadata table (for id -> title lookups), loaded via
      src.anime_recommender.data.dataset.load_raw_dataset() per the brief

surprise's SVD stores learned item factors in `algo.qi` (an [n_items, n_factors]
array) and maps raw item ids to internal indices via `trainset.to_inner_iid()` /
`trainset.to_raw_iid()`. Both functions below lean on that.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/svd_cf_model.pkl"


def _load_model(model_path: str = MODEL_PATH):
    """Load the trained surprise SVD model (and its trainset) via joblib."""
    return joblib.load(model_path)


def get_user_top_n(user_id: int, k: int = 10, model_path: str = MODEL_PATH,
                    anime_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Top-k predicted unrated anime for a KNOWN training-set user_id.

    Returns a DataFrame with columns: anime_id, title (if anime_df given), predicted_rating
    sorted descending by predicted_rating. Raises ValueError if user_id was not
    in the training set (cold-start — this model only supports known users,
    per the brief's cold-start reasoning for keeping it out of app.py).
    """
    algo = _load_model(model_path)
    trainset = algo.trainset

    try:
        inner_uid = trainset.to_inner_uid(user_id)
    except ValueError:
        raise ValueError(
            f"user_id {user_id} was not seen during training — this model "
            "only supports known training-set users (cold-start limitation)."
        )

    rated_inner_iids = {iid for (iid, _rating) in trainset.ur[inner_uid]}
    all_inner_iids = set(trainset.all_items())
    unrated_inner_iids = all_inner_iids - rated_inner_iids

    predictions = []
    for inner_iid in unrated_inner_iids:
        raw_iid = trainset.to_raw_iid(inner_iid)
        est = algo.predict(user_id, raw_iid).est
        predictions.append((raw_iid, est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top = predictions[:k]

    result = pd.DataFrame(top, columns=["anime_id", "predicted_rating"])

    if anime_df is not None:
        # TODO: confirm the id/title column names match your anime metadata table
        # (brief implies a Kaggle anime.csv-style schema — commonly anime_id / name).
        result = result.merge(
            anime_df[["anime_id", "name"]].rename(columns={"name": "title"}),
            on="anime_id", how="left",
        )
        result = result[["anime_id", "title", "predicted_rating"]]

    return result.reset_index(drop=True)


def similar_items(anime_id: int, k: int = 10, model_path: str = MODEL_PATH,
                   anime_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Top-k anime most similar to `anime_id` by cosine similarity of the
    SVD model's LEARNED ITEM EMBEDDINGS (algo.qi rows) — not genre/theme
    text similarity (that's the Jikan-based "Similar Vibes" page in app.py,
    a separate and unrelated similarity metric).

    Returns a DataFrame with columns: anime_id, title (if anime_df given), similarity.
    Never includes the query item itself. Raises ValueError if anime_id was
    not in the training set.
    """
    algo = _load_model(model_path)
    trainset = algo.trainset

    try:
        inner_iid = trainset.to_inner_iid(anime_id)
    except ValueError:
        raise ValueError(f"anime_id {anime_id} was not seen during training.")

    item_factors = algo.qi  # [n_items, n_factors]
    query_vec = item_factors[inner_iid].reshape(1, -1)

    norms = np.linalg.norm(item_factors, axis=1)
    query_norm = np.linalg.norm(query_vec)
    sims = (item_factors @ query_vec.T).flatten() / (norms * query_norm + 1e-10)

    order = np.argsort(-sims)
    order = order[order != inner_iid][:k]

    raw_ids = [trainset.to_raw_iid(i) for i in order]
    scores = [float(sims[i]) for i in order]

    result = pd.DataFrame({"anime_id": raw_ids, "similarity": scores})

    if anime_df is not None:
        # TODO: same column-name caveat as get_user_top_n above.
        result = result.merge(
            anime_df[["anime_id", "name"]].rename(columns={"name": "title"}),
            on="anime_id", how="left",
        )
        result = result[["anime_id", "title", "similarity"]]

    return result.reset_index(drop=True)
