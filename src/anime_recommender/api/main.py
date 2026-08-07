"""
FastAPI app exposing the hybrid recommender.

POST /recommend takes an anime name and returns hybrid recommendations
for an anonymous request whose only known preference is that one anime.
"""

from fastapi import FastAPI, HTTPException

from anime_recommender.api.dependencies import (
    get_cf_model,
    get_content_model,
    get_popularity,
    get_rating_counts,
)
from anime_recommender.api.schemas import RecommendRequest
from anime_recommender.models.confidence import compute_confidence
from anime_recommender.models.explain import explain_recommendation
from anime_recommender.models.hybrid import build_user_taste_vector, hybrid_top_n

app = FastAPI(title="Anime Recommender API")

# There's no account system yet, an anonymous request builds its taste
# vector from the single anime it names. This id is never a real
# training-set user, it just lets hybrid_score's CF term fall back to
# the model's global bias instead of erroring on an unknown user.
ANONYMOUS_USER_ID = -1


def _resolve_anime_id(name: str, catalog) -> int | None:
    """Case-insensitive exact match on the anime's name."""
    match = catalog[catalog["name"].str.lower() == name.strip().lower()]
    return int(match.iloc[0]["anime_id"]) if len(match) else None


@app.post("/recommend")
def recommend(request: RecommendRequest):
    content = get_content_model()
    algo = get_cf_model()
    popularity = get_popularity()
    rating_counts = get_rating_counts()

    anime_id = _resolve_anime_id(request.anime_name, content.catalog)
    if anime_id is None:
        raise HTTPException(status_code=404, detail=f"No anime found matching '{request.anime_name}'")

    liked_ids = [anime_id]
    candidate_ids = content.catalog["anime_id"].tolist()
    seen_ids = {anime_id}

    top_ids = hybrid_top_n(
        ANONYMOUS_USER_ID, request.k, algo, content, popularity,
        liked_ids, candidate_ids, seen_ids,
    )

    taste_vector = build_user_taste_vector(liked_ids, content)
    recommendations = []
    for aid in top_ids:
        title_row = content.catalog[content.catalog["anime_id"] == aid]
        title = title_row.iloc[0]["name"] if len(title_row) else None
        confidence = compute_confidence(aid, rating_counts, liked_ids, content)
        reason = explain_recommendation(
            ANONYMOUS_USER_ID, aid, algo, content, taste_vector,
            popularity, liked_ids, content.catalog,
        )
        recommendations.append({
            "anime_id": aid,
            "title": title,
            "confidence": confidence["label"],
            "reason": reason,
        })

    return {"query_anime": request.anime_name, "recommendations": recommendations}
