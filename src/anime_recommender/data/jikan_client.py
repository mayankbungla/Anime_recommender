"""
Jikan API client (live MyAnimeList data).

This was previously a set of functions living directly inside app.py.
Pulling it into its own module means:
  - app.py stays focused on UI/rendering
  - these functions are unit-testable in isolation (Week 9)
  - they can later be swapped for the FastAPI-wrapped client with
    retries/timeouts planned for Day 39, without touching the UI code

Each function is defensive: on any request failure it returns an empty
result rather than raising, so a flaky/rate-limited API call degrades
the UI gracefully instead of crashing the app.
"""

import streamlit as st
import requests

JIKAN = "https://api.jikan.moe/v4"


@st.cache_data(ttl=3600, show_spinner=False)
def jikan_search(query: str, limit: int = 12):
    try:
        r = requests.get(f"{JIKAN}/anime", params={"q": query, "limit": limit, "sfw": True}, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def jikan_anime(mal_id: int):
    try:
        r = requests.get(f"{JIKAN}/anime/{mal_id}", timeout=10)
        r.raise_for_status()
        return r.json().get("data", {})
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def jikan_recommendations(mal_id: int):
    try:
        r = requests.get(f"{JIKAN}/anime/{mal_id}/recommendations", timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])[:12]
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def jikan_top(limit: int = 50):
    try:
        r = requests.get(f"{JIKAN}/top/anime", params={"limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def jikan_season_now(limit: int = 20):
    try:
        r = requests.get(f"{JIKAN}/seasons/now", params={"limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def jikan_genre(genre_id: int, limit: int = 20):
    try:
        r = requests.get(
            f"{JIKAN}/anime",
            params={"genres": genre_id, "order_by": "score", "sort": "desc", "limit": limit, "sfw": True},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception:
        return []
