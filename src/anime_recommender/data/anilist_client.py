"""
AniList client. Independent anime database, unrelated to MyAnimeList.
Used as the first source for live anime data, with Jikan and the
local dataset as fallbacks when unavailable.
"""

import operator
import threading
import time

import requests
from cachetools import TTLCache, cachedmethod
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ANILIST_URL = "https://graphql.anilist.co"

# AniList caps each page well short of what a browse view needs, and its
# public API is currently rate-limited to 30 requests/minute (degraded
# state, see docs.anilist.co/guide/rate-limiting), so pagination here is
# deliberately shallow rather than trying to pull the whole catalogue.
PER_PAGE = 50
MAX_PAGES = 4
PAGE_DELAY = 0.35  # stays clear of AniList's burst limiter between pages

TOP_QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage }
    media(type: ANIME, sort: SCORE_DESC) {
      idMal
      title { romaji english }
      coverImage { large }
      averageScore
      episodes
      genres
      siteUrl
    }
  }
}
"""

GENRE_QUERY = """
query ($genre: String, $page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage }
    media(type: ANIME, genre: $genre, sort: SCORE_DESC) {
      idMal
      title { romaji english }
      coverImage { large }
      averageScore
      episodes
      genres
      siteUrl
    }
  }
}
"""

SEARCH_QUERY = """
query ($search: String, $perPage: Int) {
  Page(perPage: $perPage) {
    media(type: ANIME, search: $search) {
      idMal
      title { romaji english }
      coverImage { large }
      averageScore
      episodes
      genres
      siteUrl
    }
  }
}
"""


def _to_card(item: dict) -> dict:
    """Reshapes one AniList media record into the card format used
    across the app."""
    score = item.get("averageScore")
    return {
        "mal_id": item.get("idMal"),
        "title": (item.get("title") or {}).get("english") or (item.get("title") or {}).get("romaji") or "Unknown",
        "images": {"jpg": {"large_image_url": (item.get("coverImage") or {}).get("large", "")}},
        "score": round(score / 10, 2) if score else None,
        "episodes": item.get("episodes"),
        "genres": [{"name": g} for g in item.get("genres") or []],
        "url": item.get("siteUrl", "#"),
    }


class AniListClient:
    """Wrapper around the AniList GraphQL endpoint. Retries transient
    failures with backoff. Fetch methods raise on final failure so
    caching only ever stores a real successful response."""

    def __init__(self, base_url: str = ANILIST_URL, timeout: int = 10,
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
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)

    def _post(self, query: str, variables: dict) -> dict:
        r = self.session.post(self.base_url, json={"query": query, "variables": variables}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _fetch_paged(self, query: str, variables: dict, limit: int) -> list:
        """Collects up to `limit` items across multiple AniList pages.
        A single Page(perPage=...) call caps out well short of a real
        browse-sized list, so this loops until enough items are in hand,
        the API says there's no next page, or MAX_PAGES is hit."""
        items = []
        page = 1
        while len(items) < limit and page <= MAX_PAGES:
            if page > 1:
                time.sleep(PAGE_DELAY)
            data = self._post(query, {**variables, "page": page, "perPage": PER_PAGE})
            page_data = data.get("data", {}).get("Page", {})
            media = page_data.get("media", [])
            items.extend(_to_card(item) for item in media if item.get("idMal"))
            if not page_data.get("pageInfo", {}).get("hasNextPage"):
                break
            page += 1
        return items[:limit]

    @cachedmethod(operator.attrgetter("cache"), lock=operator.attrgetter("lock"))
    def _fetch_top(self, limit: int) -> list:
        return self._fetch_paged(TOP_QUERY, {}, limit)

    @cachedmethod(operator.attrgetter("cache"), lock=operator.attrgetter("lock"))
    def _fetch_genre(self, genre: str, limit: int) -> list:
        return self._fetch_paged(GENRE_QUERY, {"genre": genre}, limit)

    @cachedmethod(operator.attrgetter("cache"), lock=operator.attrgetter("lock"))
    def _fetch_search(self, query: str, limit: int) -> list:
        data = self._post(SEARCH_QUERY, {"search": query, "perPage": limit})
        media = data.get("data", {}).get("Page", {}).get("media", [])
        return [_to_card(item) for item in media if item.get("idMal")]

    def top(self, limit: int = 50) -> list:
        try:
            return self._fetch_top(limit)
        except requests.exceptions.RequestException:
            return []

    def genre(self, genre: str, limit: int = 20) -> list:
        try:
            return self._fetch_genre(genre, limit)
        except requests.exceptions.RequestException:
            return []

    def search(self, query: str, limit: int = 12) -> list:
        try:
            return self._fetch_search(query, limit)
        except requests.exceptions.RequestException:
            return []


client = AniListClient()
