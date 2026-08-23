"""
Jikan API client (live MyAnimeList data).

JikanClient wraps every call in one requests.Session with automatic
retries on timeouts, connection errors, 429s and 5xx (Day 39), and
caches successful responses in a shared TTLCache so repeat lookups
skip the network entirely (Day 40). Only successful responses are
cached, a failed request is never memoized as "no results", so a
transient outage doesn't get stuck looking like empty data for the
next hour.

The module functions below keep the same names app.py already
imports, they just delegate to the client now instead of calling
requests directly.
"""

import operator
import threading

import requests
from cachetools import TTLCache, cachedmethod
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

JIKAN = "https://api.jikan.moe/v4"


class JikanClient:
    """Thin wrapper around Jikan's REST API. Retries transient failures
    with backoff before giving up. Each public method calls a cached
    fetch, catching failures outside the cache so they're never stored."""

    def __init__(self, base_url: str = JIKAN, timeout: int = 10,
                 max_retries: int = 3, backoff_factor: float = 0.5,
                 cache_maxsize: int = 500, cache_ttl: int = 3600):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self.lock = threading.Lock()

        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # -- cached fetchers: raise on failure, so cachedmethod only ever
    # stores a real successful response, never an error

    @cachedmethod(operator.attrgetter("cache"), lock=operator.attrgetter("lock"))
    def _fetch_search(self, query: str, limit: int) -> list:
        r = self.session.get(f"{self.base_url}/anime", params={"q": query, "limit": limit, "sfw": True}, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("data", [])

    @cachedmethod(operator.attrgetter("cache"), lock=operator.attrgetter("lock"))
    def _fetch_anime(self, mal_id: int) -> dict:
        r = self.session.get(f"{self.base_url}/anime/{mal_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("data", {})

    @cachedmethod(operator.attrgetter("cache"), lock=operator.attrgetter("lock"))
    def _fetch_recommendations(self, mal_id: int) -> list:
        r = self.session.get(f"{self.base_url}/anime/{mal_id}/recommendations", timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("data", [])[:12]

    @cachedmethod(operator.attrgetter("cache"), lock=operator.attrgetter("lock"))
    def _fetch_top(self, limit: int) -> list:
        r = self.session.get(f"{self.base_url}/top/anime", params={"limit": limit}, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("data", [])

    @cachedmethod(operator.attrgetter("cache"), lock=operator.attrgetter("lock"))
    def _fetch_season_now(self, limit: int) -> list:
        r = self.session.get(f"{self.base_url}/seasons/now", params={"limit": limit}, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("data", [])

    @cachedmethod(operator.attrgetter("cache"), lock=operator.attrgetter("lock"))
    def _fetch_genre(self, genre_id: int, limit: int) -> list:
        r = self.session.get(
            f"{self.base_url}/anime",
            params={"genres": genre_id, "order_by": "score", "sort": "desc", "limit": limit, "sfw": True},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json().get("data", [])

    # -- public methods: uncached, catch whatever the fetch above raised

    def search(self, query: str, limit: int = 12) -> list:
        try:
            return self._fetch_search(query, limit)
        except requests.exceptions.RequestException:
            return []

    def anime(self, mal_id: int) -> dict:
        try:
            return self._fetch_anime(mal_id)
        except requests.exceptions.RequestException:
            return {}

    def recommendations(self, mal_id: int) -> list:
        try:
            return self._fetch_recommendations(mal_id)
        except requests.exceptions.RequestException:
            return []

    def top(self, limit: int = 50) -> list:
        try:
            return self._fetch_top(limit)
        except requests.exceptions.RequestException:
            return []

    def season_now(self, limit: int = 20) -> list:
        try:
            return self._fetch_season_now(limit)
        except requests.exceptions.RequestException:
            return []

    def genre(self, genre_id: int, limit: int = 20) -> list:
        try:
            return self._fetch_genre(genre_id, limit)
        except requests.exceptions.RequestException:
            return []


client = JikanClient()


def jikan_search(query: str, limit: int = 12):
    return client.search(query, limit)

def jikan_anime(mal_id: int):
    return client.anime(mal_id)

def jikan_recommendations(mal_id: int):
    return client.recommendations(mal_id)

def jikan_top(limit: int = 50):
    return client.top(limit)

def jikan_season_now(limit: int = 20):
    return client.season_now(limit)

def jikan_genre(genre_id: int, limit: int = 20):
    return client.genre(genre_id, limit)
