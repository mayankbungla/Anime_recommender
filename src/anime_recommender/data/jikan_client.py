"""
Jikan API client (live MyAnimeList data).

JikanClient wraps every call in one requests.Session with automatic
retries on timeouts, connection errors, 429s and 5xx (Day 39). The
cached functions below keep the same names app.py already imports,
they just delegate to the shared client now instead of calling
requests directly.
"""

import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

JIKAN = "https://api.jikan.moe/v4"


class JikanClient:
    """Thin wrapper around Jikan's REST API. Retries transient failures
    with backoff before giving up, everything else returns an empty
    result instead of raising, so a bad request never crashes the app."""

    def __init__(self, base_url: str = JIKAN, timeout: int = 10,
                 max_retries: int = 3, backoff_factor: float = 0.5):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()

        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def search(self, query: str, limit: int = 12) -> list:
        try:
            r = self.session.get(f"{self.base_url}/anime", params={"q": query, "limit": limit, "sfw": True}, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("data", [])
        except requests.exceptions.RequestException:
            return []

    def anime(self, mal_id: int) -> dict:
        try:
            r = self.session.get(f"{self.base_url}/anime/{mal_id}", timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("data", {})
        except requests.exceptions.RequestException:
            return {}

    def recommendations(self, mal_id: int) -> list:
        try:
            r = self.session.get(f"{self.base_url}/anime/{mal_id}/recommendations", timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("data", [])[:12]
        except requests.exceptions.RequestException:
            return []

    def top(self, limit: int = 50) -> list:
        try:
            r = self.session.get(f"{self.base_url}/top/anime", params={"limit": limit}, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("data", [])
        except requests.exceptions.RequestException:
            return []

    def season_now(self, limit: int = 20) -> list:
        try:
            r = self.session.get(f"{self.base_url}/seasons/now", params={"limit": limit}, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("data", [])
        except requests.exceptions.RequestException:
            return []

    def genre(self, genre_id: int, limit: int = 20) -> list:
        try:
            r = self.session.get(
                f"{self.base_url}/anime",
                params={"genres": genre_id, "order_by": "score", "sort": "desc", "limit": limit, "sfw": True},
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json().get("data", [])
        except requests.exceptions.RequestException:
            return []


client = JikanClient()


@st.cache_data(ttl=3600, show_spinner=False)
def jikan_search(query: str, limit: int = 12):
    return client.search(query, limit)

@st.cache_data(ttl=3600, show_spinner=False)
def jikan_anime(mal_id: int):
    return client.anime(mal_id)

@st.cache_data(ttl=3600, show_spinner=False)
def jikan_recommendations(mal_id: int):
    return client.recommendations(mal_id)

@st.cache_data(ttl=3600, show_spinner=False)
def jikan_top(limit: int = 50):
    return client.top(limit)

@st.cache_data(ttl=3600, show_spinner=False)
def jikan_season_now(limit: int = 20):
    return client.season_now(limit)

@st.cache_data(ttl=3600, show_spinner=False)
def jikan_genre(genre_id: int, limit: int = 20):
    return client.genre(genre_id, limit)
