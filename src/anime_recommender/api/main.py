"""
FastAPI backend for hybrid recommendations. Loads everything once at
import time, not per request, same reasoning as the batch evaluation
scripts - reloading anything from disk on every call would be too slow.

Uses the lightweight exported factors (models/factors/) instead of the
full ~162MB surprise model object. The item-to-item hybrid scoring this
API actually does (query anime -> similar anime) only ever needs the
learned item vectors, never a fitted model's predict() or biases, so
the full pickle was never necessary here - it was just what got wired
up first. Loading it anyway was blowing well past a 512MB memory limit,
since joblib unpickling a surprise Trainset costs far more in live
Python objects than its file size suggests.

Popularity is read from a small precomputed file (models/popularity_scores.parquet,
see scripts/export_popularity.py) instead of loading the full multi-million-row
ratings_train.parquet just to compute it once at startup.

Scoring the full catalogue against every request would be slow, so each
request first pulls a shortlist from the CF and content neighbours,
then re-ranks that smaller pool with the full hybrid formula.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from anime_recommender.api.schemas import (
    AnimeInfo,
    HealthResponse,
    RecommendationItem,
    RecommendRequest,
    RecommendResponse,
)
from anime_recommender.features.content_model import ContentRecommender
from anime_recommender.models.hybrid import item_content_similarity

ROOT = Path(__file__).resolve().parents[3]
FACTORS_DIR = ROOT / "models" / "factors"
CONTENT_DIR = ROOT / "models" / "content"
POPULARITY_PATH = ROOT / "models" / "popularity_scores.parquet"
# content_catalog.parquet only carries anime_id/name/genre (what the
# embeddings needed), so /anime/{id} reads the rest from here instead.
ANIME_CLEAN_PATH = ROOT / "data" / "processed" / "anime_clean.parquet"

SHORTLIST_SIZE = 50

app = FastAPI(title="Anime Recommender API")

item_factors = np.load(FACTORS_DIR / "item_factors.npy")
_item_id_map = pd.read_parquet(FACTORS_DIR / "item_id_map.parquet")
_row_by_anime_id = dict(zip(_item_id_map["raw_anime_id"], _item_id_map["inner_iid"]))
_anime_id_by_row = dict(zip(_item_id_map["inner_iid"], _item_id_map["raw_anime_id"]))
_item_norms = np.linalg.norm(item_factors, axis=1)

content = ContentRecommender.load(CONTENT_DIR)
_popularity_df = pd.read_parquet(POPULARITY_PATH)
popularity = dict(zip(_popularity_df["anime_id"], _popularity_df["popularity"]))
anime_clean = pd.read_parquet(ANIME_CLEAN_PATH).set_index("anime_id")


def resolve_anime_id(name: str) -> int | None:
    """Case-insensitive substring match against the content catalogue."""
    matches = content.catalog[content.catalog["name"].str.contains(name, case=False, na=False, regex=False)]
    return int(matches.iloc[0]["anime_id"]) if len(matches) else None


def cf_shortlist(anime_id: int, n: int) -> list:
    """Nearest CF neighbours by learned embedding, using the lightweight
    exported factors already in memory rather than a fitted model."""
    if anime_id not in _row_by_anime_id:
        return []
    idx = _row_by_anime_id[anime_id]
    sims = (item_factors @ item_factors[idx]) / (_item_norms * _item_norms[idx] + 1e-10)
    order = np.argsort(-sims)
    order = [i for i in order if i != idx][:n]
    return [_anime_id_by_row[i] for i in order]


def content_shortlist(anime_id: int, n: int) -> list:
    if anime_id not in content._row_by_id:
        return []
    # can't ask for more neighbours than the catalogue actually has
    n = min(n, len(content.catalog) - 1)
    neighbours = content.recommend(anime_id, k=n)
    return neighbours["anime_id"].tolist()


def cf_item_similarity(query_id: int, candidate_id: int) -> float | None:
    """Cosine similarity between two anime's learned CF embeddings,
    rescaled to 0-1. None if either anime has no CF factors."""
    if query_id not in _row_by_anime_id or candidate_id not in _row_by_anime_id:
        return None
    a = item_factors[_row_by_anime_id[query_id]]
    b = item_factors[_row_by_anime_id[candidate_id]]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    sim = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    return (sim + 1) / 2


def item_hybrid_score(query_id: int, candidate_id: int, alpha: float = 0.34,
                       beta: float = 0.33, gamma: float = 0.33) -> float:
    """Same formula as hybrid.item_hybrid_score, just sourced from the
    lightweight factors above instead of a full model object."""
    cf = cf_item_similarity(query_id, candidate_id)
    content_s = item_content_similarity(query_id, candidate_id, content)
    pop = popularity.get(candidate_id, 0.0)

    if cf is None:
        total = beta + gamma
        return (beta / total) * content_s + (gamma / total) * pop

    return alpha * cf + beta * content_s + gamma * pop


def item_hybrid_top_n(query_id: int, k: int, candidate_ids: list,
                       alpha: float = 0.34, beta: float = 0.33, gamma: float = 0.33) -> list:
    scored = [
        (cid, item_hybrid_score(query_id, cid, alpha, beta, gamma))
        for cid in candidate_ids if cid != query_id
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def _none_if_nan(value):
    """Parquet-sourced numeric fields come back as NaN, not missing,
    NaN isn't valid JSON so this turns it into a real null."""
    return None if pd.isna(value) else value


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=item_factors is not None, catalog_size=len(content.catalog))


@app.get("/anime/{anime_id}", response_model=AnimeInfo)
def get_anime(anime_id: int):
    match = content.catalog[content.catalog["anime_id"] == anime_id]
    if not len(match):
        raise HTTPException(status_code=404, detail=f"No anime with id {anime_id}")

    row = match.iloc[0]
    clean_row = anime_clean.loc[anime_id] if anime_id in anime_clean.index else None

    return AnimeInfo(
        anime_id=int(row["anime_id"]),
        name=row["name"],
        genre=_none_if_nan(row.get("genre")),
        type=_none_if_nan(clean_row["type"]) if clean_row is not None else None,
        episodes=_none_if_nan(clean_row["episodes"]) if clean_row is not None else None,
        rating=_none_if_nan(clean_row["rating"]) if clean_row is not None else None,
        members=_none_if_nan(clean_row["members"]) if clean_row is not None else None,
        synopsis=_none_if_nan(row.get("synopsis")),
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    anime_id = resolve_anime_id(request.anime_name)
    if anime_id is None:
        raise HTTPException(status_code=404, detail=f"No anime matching '{request.anime_name}'")

    candidates = list(dict.fromkeys(
        cf_shortlist(anime_id, SHORTLIST_SIZE) + content_shortlist(anime_id, SHORTLIST_SIZE)
    ))
    if not candidates:
        raise HTTPException(status_code=404, detail="No candidates found for this anime")

    ranked = item_hybrid_top_n(anime_id, request.k, candidates)

    titles = content.catalog.set_index("anime_id")["name"]
    return RecommendResponse(
        query_anime_id=anime_id,
        query_title=titles.get(anime_id, request.anime_name),
        recommendations=[
            RecommendationItem(anime_id=aid, title=titles.get(aid, "Unknown"), hybrid_score=round(score, 4))
            for aid, score in ranked
        ],
    )
