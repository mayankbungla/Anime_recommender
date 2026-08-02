"""
Live data from the Jikan API, used only for display metadata and for
widening the candidate pool with brand-new anime the local Kaggle-based
catalogue doesn't know about yet. Never used to score or rank anything,
that stays entirely with the trained CF, content, and popularity signals.
"""

import requests

JIKAN = "https://api.jikan.moe/v4"


def get_live_metadata(mal_id: int) -> dict:
    """Poster, current score, and airing status for one anime, for
    display only, has no effect on any score used elsewhere."""
    try:
        r = requests.get(f"{JIKAN}/anime/{mal_id}", timeout=10)
        r.raise_for_status()
        data = r.json().get("data", {})
        return {
            "poster_url": (data.get("images", {}).get("jpg") or {}).get("image_url"),
            "score": data.get("score"),
            "airing_status": data.get("status"),
        }
    except Exception:
        return {}


def get_cold_start_candidates(n: int = 20) -> list[dict]:
    """Currently trending anime from Jikan, used to widen recommendation
    candidates for titles too new to be in the local dataset at all."""
    try:
        r = requests.get(f"{JIKAN}/top/anime", params={"limit": n}, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception:
        return []


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
