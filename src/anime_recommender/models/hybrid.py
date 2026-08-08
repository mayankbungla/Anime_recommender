"""
Combines the CF model, the content model, and raw popularity into one
ranked score per anime. See reports/hybrid_design.md for the reasoning
behind the formula and the weights.

Anime with no CF factors (too few ratings to have been trained on) fall
back to content plus popularity only, renormalized so the weights still
sum to 1 rather than silently treating "unknown" as "predicted low."
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def build_popularity_scores(train: pd.DataFrame) -> dict:
    """Log-scaled, min-max normalized rating count per anime, 0 to 1."""
    counts = train.groupby("anime_id").size()
    log_counts = np.log1p(counts)
    normalized = (log_counts - log_counts.min()) / (log_counts.max() - log_counts.min())
    return normalized.to_dict()


def build_user_taste_vector(liked_ids: list, content) -> np.ndarray | None:
    """Mean embedding of everything the user rated highly, L2-normalised.
    Returns None if none of the user's liked anime are in the content catalogue."""
    rows = [content._row_by_id[a] for a in liked_ids if a in content._row_by_id]
    if not rows:
        return None
    vec = content.embeddings[rows].mean(axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cf_score(user_id: int, anime_id: int, algo) -> float | None:
    """CF's predicted rating rescaled to 0-1, or None if the anime was
    never seen during training (the cold-start case)."""
    trainset = algo.trainset
    try:
        trainset.to_inner_iid(anime_id)
    except ValueError:
        return None
    est = algo.predict(user_id, anime_id).est
    return (est - 1) / 9


def content_score(anime_id: int, taste_vector: np.ndarray | None, content) -> float:
    """Cosine similarity between the anime and the user's taste vector,
    rescaled from -1..1 to 0..1. Zero if either side is unavailable."""
    if taste_vector is None or anime_id not in content._row_by_id:
        return 0.0
    vec = content.embeddings[content._row_by_id[anime_id]]
    sim = float(np.dot(vec, taste_vector))
    return (sim + 1) / 2


def hybrid_score(user_id: int, anime_id: int, algo, content, taste_vector,
                  popularity: dict, alpha: float = 0.5, beta: float = 0.3,
                  gamma: float = 0.2) -> float:
    """One anime's final blended score for one user. See module docstring
    for the cold-start renormalization when CF has no signal."""
    cf = cf_score(user_id, anime_id, algo)
    content_s = content_score(anime_id, taste_vector, content)
    pop = popularity.get(anime_id, 0.0)

    if cf is None:
        total = beta + gamma
        return (beta / total) * content_s + (gamma / total) * pop

    return alpha * cf + beta * content_s + gamma * pop


def hybrid_top_n(user_id: int, k: int, algo, content, popularity: dict,
                  liked_ids: list, candidate_ids: list,
                  seen_ids: set, alpha: float = 0.5, beta: float = 0.3,
                  gamma: float = 0.2) -> list:
    """Ranks candidate_ids for one user and returns the top-k anime_ids,
    skipping anything the user has already rated."""
    taste_vector = build_user_taste_vector(liked_ids, content)
    scored = [
        (aid, hybrid_score(user_id, aid, algo, content, taste_vector,
                            popularity, alpha, beta, gamma))
        for aid in candidate_ids if aid not in seen_ids
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [aid for aid, _ in scored[:k]]


def cf_item_similarity(query_id: int, candidate_id: int, algo) -> float | None:
    """Cosine similarity between two anime's learned CF embeddings,
    rescaled to 0-1. None if either anime has no CF factors."""
    trainset = algo.trainset
    try:
        qi = trainset.to_inner_iid(query_id)
        ci = trainset.to_inner_iid(candidate_id)
    except ValueError:
        return None
    a, b = algo.qi[qi], algo.qi[ci]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    sim = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    return (sim + 1) / 2


def item_content_similarity(query_id: int, candidate_id: int, content) -> float:
    """Cosine similarity between two anime's synopsis embeddings,
    rescaled to 0-1. Zero if either anime is missing from the catalogue."""
    if query_id not in content._row_by_id or candidate_id not in content._row_by_id:
        return 0.0
    a = content.embeddings[content._row_by_id[query_id]]
    b = content.embeddings[content._row_by_id[candidate_id]]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    sim = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    return (sim + 1) / 2


def item_hybrid_score(query_id: int, candidate_id: int, algo, content,
                       popularity: dict, alpha: float = 0.34, beta: float = 0.33,
                       gamma: float = 0.33) -> float:
    """
    Blended anime-to-anime score, no user required. Same cold-start
    renormalization as hybrid_score when CF has no factors for either
    anime. Default weights are the tuned result from
    reports/hybrid_weight_tuning.csv.
    """
    cf = cf_item_similarity(query_id, candidate_id, algo)
    content_s = item_content_similarity(query_id, candidate_id, content)
    pop = popularity.get(candidate_id, 0.0)

    if cf is None:
        total = beta + gamma
        return (beta / total) * content_s + (gamma / total) * pop

    return alpha * cf + beta * content_s + gamma * pop


def item_hybrid_top_n(query_id: int, k: int, algo, content, popularity: dict,
                       candidate_ids: list, alpha: float = 0.34,
                       beta: float = 0.33, gamma: float = 0.33) -> list:
    """Ranks candidate_ids against one query anime, no user needed.
    Returns a list of (anime_id, score) tuples, highest first."""
    scored = [
        (cid, item_hybrid_score(query_id, cid, algo, content, popularity, alpha, beta, gamma))
        for cid in candidate_ids if cid != query_id
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
