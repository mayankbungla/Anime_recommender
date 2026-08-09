"""
Request/response models for the API. Response models on every route now
(Day 37), so /docs shows real schemas instead of untyped dicts.
"""

from pydantic import BaseModel


class RecommendRequest(BaseModel):
    anime_name: str
    k: int = 10


class RecommendationItem(BaseModel):
    anime_id: int
    title: str
    hybrid_score: float


class RecommendResponse(BaseModel):
    query_anime_id: int
    query_title: str
    recommendations: list[RecommendationItem]


class AnimeInfo(BaseModel):
    anime_id: int
    name: str
    genre: str | None = None
    type: str | None = None
    episodes: float | None = None
    rating: float | None = None
    members: int | None = None
    synopsis: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    catalog_size: int
