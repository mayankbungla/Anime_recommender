"""
FastAPI backend for hybrid recommendations. Loads the trained model and
content embeddings once at import time, not per request, same reasoning
as the batch evaluation scripts, reloading a 162MB model on every call
would make this unusable.

Scoring the full catalogue against every request would be slow, so each
request first pulls a shortlist from the CF and content neighbours,
then re-ranks that smaller pool with the full hybrid formula.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
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
from anime_recommender.models.hybrid import build_popularity_scores, item_hybrid_top_n

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "svd_cf_model.pkl"
CONTENT_DIR = ROOT / "models" / "content"
TRAIN_PATH = ROOT / "data" / "processed" / "ratings_train.parquet"
# content_catalog.parquet only carries anime_id/name/genre (what the
# embeddings needed), so /anime/{id} reads the rest from here instead.
ANIME_CLEAN_PATH = ROOT / "data" / "processed" / "anime_clean.parquet"

SHORTLIST_SIZE = 50

app = FastAPI(title="Anime Recommender API")

algo = joblib.load(MODEL_PATH)
content = ContentRecommender.load(CONTENT_DIR)
train = pd.read_parquet(TRAIN_PATH)
popularity = build_popularity_scores(train)
anime_clean = pd.read_parquet(ANIME_CLEAN_PATH).set_index("anime_id")


def resolve_anime_id(name: str) -> int | None:
    """Case-insensitive substring match against the content catalogue."""
    matches = content.catalog[content.catalog["name"].str.contains(name, case=False, na=False, regex=False)]
    return int(matches.iloc[0]["anime_id"]) if len(matches) else None


def cf_shortlist(anime_id: int, n: int) -> list:
    """Nearest CF neighbours by learned embedding, using the model
    already in memory rather than reading factor files from disk."""
    trainset = algo.trainset
    try:
        idx = trainset.to_inner_iid(anime_id)
    except ValueError:
        return []
    qi = algo.qi
    norms = np.linalg.norm(qi, axis=1)
    sims = (qi @ qi[idx]) / (norms * norms[idx] + 1e-10)
    order = np.argsort(-sims)
    order = [i for i in order if i != idx][:n]
    return [trainset.to_raw_iid(i) for i in order]


def content_shortlist(anime_id: int, n: int) -> list:
    if anime_id not in content._row_by_id:
        return []
    # can't ask for more neighbours than the catalogue actually has
    n = min(n, len(content.catalog) - 1)
    neighbours = content.recommend(anime_id, k=n)
    return neighbours["anime_id"].tolist()


def _none_if_nan(value):
    """Parquet-sourced numeric fields come back as NaN, not missing,
    NaN isn't valid JSON so this turns it into a real null."""
    return None if pd.isna(value) else value


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=algo is not None, catalog_size=len(content.catalog))


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

    ranked = item_hybrid_top_n(anime_id, request.k, algo, content, popularity, candidates)

    titles = content.catalog.set_index("anime_id")["name"]
    return RecommendResponse(
        query_anime_id=anime_id,
        query_title=titles.get(anime_id, request.anime_name),
        recommendations=[
            RecommendationItem(anime_id=aid, title=titles.get(aid, "Unknown"), hybrid_score=round(score, 4))
            for aid, score in ranked
        ],
    )
