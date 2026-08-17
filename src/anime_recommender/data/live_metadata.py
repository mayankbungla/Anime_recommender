"""
Live data from the Jikan API, used only for display metadata and for
widening the candidate pool with brand-new anime the local Kaggle-based
catalogue doesn't know about yet. Never used to score or rank anything,
that stays entirely with the trained CF, content, and popularity signals.

Uses the same retrying JikanClient as jikan_client.py (Day 39) instead
of making its own raw requests calls.
"""

from anime_recommender.data.jikan_client import client


def get_live_metadata(mal_id: int) -> dict:
    """Poster, current score, and airing status for one anime, for
    display only, has no effect on any score used elsewhere."""
    data = client.anime(mal_id)
    if not data:
        return {}
    return {
        "poster_url": (data.get("images", {}).get("jpg") or {}).get("image_url"),
        "score": data.get("score"),
        "airing_status": data.get("status"),
    }


def get_cold_start_candidates(n: int = 20) -> list[dict]:
    """Currently trending anime from Jikan, used to widen recommendation
    candidates for titles too new to be in the local dataset at all."""
    return client.top(n)


def add_cold_start_candidates(local_candidate_ids: list, n: int = 20) -> list:
    """
    Merges live trending anime into a local candidate list before scoring.
    Jikan is only ever used to widen who gets *considered*, the actual
    hybrid_score() call in hybrid.py still decides who ranks well, so a
    trending anime with no local data just falls through the same
    content+popularity cold-start path as any other unrated title.
    """
    live = get_cold_start_candidates(n)
    # Kaggle's anime_id and Jikan's mal_id are the same MyAnimeList id,
    # so this assumes they line up directly rather than needing a lookup table
    live_ids = [item["mal_id"] for item in live if "mal_id" in item]
    return list(dict.fromkeys(local_candidate_ids + live_ids))
