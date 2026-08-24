"""
AniList client. Independent anime database, unrelated to MyAnimeList.
Used as the first source for live anime data, with Jikan and the
local dataset as fallbacks when unavailable.
"""

import operator
import threading

import requests
from cachetools import TTLCache, cachedmethod
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ANILIST_URL = "https://graphql.anilist.co"

TOP_QUERY = """
query ($perPage: Int) {
  Page(perPage: $perPage) {
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

    @cachedmethod(operator.attrgetter("cache"), lock=operator.attrgetter("lock"))
    def _fetch_top(self, limit: int) -> list:
        data = self._post(TOP_QUERY, {"perPage": limit})
        media = data.get("data", {}).get("Page", {}).get("media", [])
        return [_to_card(item) for item in media if item.get("idMal")]

    def top(self, limit: int = 50) -> list:
        try:
            return self._fetch_top(limit)
        except requests.exceptions.RequestException:
            return []


client = AniListClient()
