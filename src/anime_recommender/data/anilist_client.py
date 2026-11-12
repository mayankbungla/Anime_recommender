"""
AniList client. Independent anime database, unrelated to MyAnimeList.
Used as the first source for live anime data, with Jikan and the
local dataset as fallbacks when unavailable.
"""

import logging
import operator
import threading
import time

import requests
from cachetools import TTLCache, cachedmethod
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

ANILIST_URL = "https://graphql.anilist.co"

# AniList caps each page well short of what a browse view needs, and its
# public API is currently rate-limited to 30 requests/minute (degraded
# state, see docs.anilist.co/guide/rate-limiting), so pagination here is
# deliberately shallow rather than trying to pull the whole catalogue.
PER_PAGE = 50
MAX_PAGES = 4
DEEP_PAGES = 8  # airing and genre lists can run to a few hundred titles
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

AIRING_QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage }
    media(type: ANIME, status: RELEASING, sort: POPULARITY_DESC) {
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

POSTER_QUERY = """
query ($ids: [Int], $perPage: Int) {
  Page(perPage: $perPage) {
    media(type: ANIME, idMal_in: $ids) {
      idMal
      coverImage { large }
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
        # one cache per fetch method, a shared one would mix up results
        # whenever two methods are called with the same arguments
        self.cache_top = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self.cache_genre = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self.cache_search = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self.cache_airing = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self.lock = threading.Lock()
        self.poster_cache = {}
        self.poster_missing = set()

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

    def _fetch_paged(self, query: str, variables: dict, limit: int, max_pages: int = MAX_PAGES) -> list:
        """Collects up to `limit` items across multiple AniList pages.
        A single Page(perPage=...) call caps out well short of a real
        browse-sized list, so this loops until enough items are in hand,
        the API says there's no next page, or MAX_PAGES is hit."""
        items = []
        page = 1
        while len(items) < limit and page <= max_pages:
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

    @cachedmethod(operator.attrgetter("cache_top"), lock=operator.attrgetter("lock"))
    def _fetch_top(self, limit: int) -> list:
        return self._fetch_paged(TOP_QUERY, {}, limit)

    @cachedmethod(operator.attrgetter("cache_genre"), lock=operator.attrgetter("lock"))
    def _fetch_genre(self, genre: str, limit: int) -> list:
        return self._fetch_paged(GENRE_QUERY, {"genre": genre}, limit, DEEP_PAGES)

    @cachedmethod(operator.attrgetter("cache_search"), lock=operator.attrgetter("lock"))
    def _fetch_search(self, query: str, limit: int) -> list:
        data = self._post(SEARCH_QUERY, {"search": query, "perPage": limit})
        media = data.get("data", {}).get("Page", {}).get("media", [])
        return [_to_card(item) for item in media if item.get("idMal")]

    @cachedmethod(operator.attrgetter("cache_airing"), lock=operator.attrgetter("lock"))
    def _fetch_airing(self, limit: int) -> list:
        return self._fetch_paged(AIRING_QUERY, {}, limit, DEEP_PAGES)

    def airing(self, limit: int = 40) -> list:
        try:
            return self._fetch_airing(limit)
        except requests.exceptions.RequestException as exc:
            logger.warning("airing request failed: %s", exc)
            return []

    def posters(self, ids: list) -> dict:
        """mal_id -> cover url. Successful lookups are kept for the life of
        the process, failures are never stored."""
        wanted = [i for i in dict.fromkeys(ids) if i not in self.poster_cache and i not in self.poster_missing]
        for start in range(0, len(wanted), PER_PAGE):
            chunk = wanted[start:start + PER_PAGE]
            if start:
                time.sleep(PAGE_DELAY)
            try:
                data = self._post(POSTER_QUERY, {"ids": chunk, "perPage": len(chunk)})
            except requests.exceptions.RequestException as exc:
                logger.warning("poster request failed: %s", exc)
                break
            media = ((data.get("data") or {}).get("Page") or {}).get("media") or []
            for item in media:
                url = (item.get("coverImage") or {}).get("large")
                if item.get("idMal") and url:
                    self.poster_cache[item["idMal"]] = url
            self.poster_missing.update(i for i in chunk if i not in self.poster_cache)
        return {i: self.poster_cache[i] for i in ids if i in self.poster_cache}

    def top(self, limit: int = 50) -> list:
        try:
            return self._fetch_top(limit)
        except requests.exceptions.RequestException as exc:
            logger.warning("top request failed: %s", exc)
            return []

    def genre(self, genre: str, limit: int = 20) -> list:
        try:
            return self._fetch_genre(genre, limit)
        except requests.exceptions.RequestException as exc:
            logger.warning("genre request failed: %s", exc)
            return []

    def search(self, query: str, limit: int = 12) -> list:
        try:
            return self._fetch_search(query, limit)
        except requests.exceptions.RequestException as exc:
            logger.warning("search request failed: %s", exc)
            return []


client = AniListClient()
