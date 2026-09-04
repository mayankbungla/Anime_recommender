"""
Tests for the FastAPI backend (src/anime_recommender/api/main.py), using
FastAPI's TestClient against the real artifacts already on disk
(models/factors/, models/content/, models/popularity_scores.parquet,
data/processed/anime_clean.parquet) - the same files the live deployment
depends on, so these tests only pass when the actual API would too.
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_recommender.api.main import app

client = TestClient(app)

KNOWN_ANIME = "Death Note"
UNKNOWN_ANIME = "Zzz Totally Fake Anime Title Nobody Made Up 999"


def test_health_reports_ok_and_loaded_catalog():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["catalog_size"] > 0


def test_recommend_known_anime_returns_ranked_results():
    response = client.post("/recommend", json={"anime_name": KNOWN_ANIME, "k": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["query_title"]
    assert 0 < len(data["recommendations"]) <= 5
    for rec in data["recommendations"]:
        assert "anime_id" in rec and "title" in rec and "hybrid_score" in rec


def test_recommend_unknown_anime_returns_404():
    response = client.post("/recommend", json={"anime_name": UNKNOWN_ANIME})
    assert response.status_code == 404


def test_anime_lookup_by_id_matches_recommend_query_id():
    # resolves a real anime_id via /recommend rather than hardcoding one,
    # so this test doesn't depend on knowing exact ids ahead of time
    recommend_response = client.post("/recommend", json={"anime_name": KNOWN_ANIME, "k": 1})
    anime_id = recommend_response.json()["query_anime_id"]

    response = client.get(f"/anime/{anime_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["anime_id"] == anime_id
    assert data["name"]


def test_anime_lookup_unknown_id_returns_404():
    response = client.get("/anime/999999999")
    assert response.status_code == 404
