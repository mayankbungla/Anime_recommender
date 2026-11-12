"""
Combines live and local anime sources with a fallback order: AniList
first, Jikan second, local dataset last. Keeps a page working even
when one or more live sources are unavailable.
"""

import html
from pathlib import Path

import pandas as pd

from anime_recommender.data.anilist_client import client as anilist_client
from anime_recommender.data.jikan_client import client as jikan_client

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
_anime_clean = pd.read_parquet(PROCESSED_DIR / "anime_clean.parquet")


def _local_top(limit: int) -> list:
    """Highest-rated anime from the stored dataset. Used only when
    live sources are unavailable, carries no poster image."""
    top = _anime_clean.sort_values("rating", ascending=False).head(limit)

    return [
        {
            "mal_id": int(row["anime_id"]),
            "title": html.unescape(str(row["name"])),
            "images": {"jpg": {"large_image_url": ""}},
            "score": row.get("rating"),
            "episodes": row.get("episodes"),
            "genres": [{"name": html.unescape(g.strip())} for g in str(row.get("genre") or "").split(",") if g.strip()],
            "url": "#",
        }
        for _, row in top.iterrows()
    ]


def get_airing(limit: int = 40) -> list:
    """Anime airing now. Tries AniList, then Jikan. No local fallback
    since the stored dataset has no airing information."""
    result = anilist_client.airing(limit)
    if result:
        return result
    return jikan_client.season_now(limit)


_no_cover = set()


def attach_posters(cards: list) -> list:
    """Fills in missing cover images from AniList, one batched request
    per 50 cards. Cards stay as they are if the lookup fails."""
    def has_image(card):
        return bool(((card.get("images") or {}).get("jpg") or {}).get("large_image_url"))

    missing = [c["mal_id"] for c in cards if c.get("mal_id") and not has_image(c)]
    if not missing:
        return cards
    found = dict(anilist_client.posters(missing))

    # second source for the few covers AniList does not have, capped so a
    # page of obscure titles does not wait on many slow lookups
    tried = 0
    for mal_id in missing:
        if mal_id in found or mal_id in _no_cover or tried >= 6:
            continue
        tried += 1
        url = ((jikan_client.anime(mal_id).get("images") or {}).get("jpg") or {}).get("large_image_url")
        if url:
            found[mal_id] = url
        else:
            _no_cover.add(mal_id)

    for card in cards:
        url = found.get(card.get("mal_id"))
        if url and not has_image(card):
            card["images"] = {"jpg": {"large_image_url": url}}
    return cards


def get_genres_for_ids(anime_ids: list) -> dict:
    """anime_id -> genre string, for the given ids. Local-data only,
    used by Day 42's analytics page, not by any live-display path."""
    subset = _anime_clean[_anime_clean["anime_id"].isin(anime_ids)]
    return dict(zip(subset["anime_id"], subset["genre"]))


def get_top(limit: int = 50) -> list:
    """Top anime by score. Tries AniList, then Jikan, then the local
    dataset, in that order."""
    result = anilist_client.top(limit)
    if result:
        return result

    result = jikan_client.top(limit)
    if result:
        return result

    return _local_top(limit)


def _local_search(query: str, limit: int) -> list:
    """Search the local dataset by title, case-insensitive substring.
    Used only when live sources are unavailable, carries no poster image."""
    matches = _anime_clean[_anime_clean["name"].str.contains(query, case=False, na=False, regex=False)].head(limit)
    
    return [
        {
            "mal_id": int(row["anime_id"]),
            "title": html.unescape(str(row["name"])),
            "images": {"jpg": {"large_image_url": ""}},
            "score": row.get("rating"),
            "episodes": row.get("episodes"),
            "genres": [{"name": html.unescape(g.strip())} for g in str(row.get("genre") or "").split(",") if g.strip()],
            "url": "#",
        }
        for _, row in matches.iterrows()
    ]


def get_search(query: str, limit: int = 12) -> list:
    """Search by title. Tries AniList, then Jikan, then the local
    dataset, in that order."""
    result = anilist_client.search(query, limit)
    if result:
        return result

    result = jikan_client.search(query, limit)
    if result:
        return result

    return _local_search(query, limit)


def _local_by_genre(genre_name: str, limit: int) -> list:
    """Anime tagged with one genre from the stored dataset. Used only
    when live sources are unavailable, carries no poster image."""
    matches = _anime_clean[_anime_clean["genre"].str.contains(genre_name, case=False, na=False, regex=False)]
    matches = matches.sort_values("rating", ascending=False, na_position="last").head(limit)

    return [
        {
            "mal_id": int(row["anime_id"]),
            "title": html.unescape(str(row["name"])),
            "images": {"jpg": {"large_image_url": ""}},
            "score": row.get("rating"),
            "episodes": row.get("episodes"),
            "genres": [{"name": html.unescape(g.strip())} for g in str(row.get("genre") or "").split(",") if g.strip()],
            "url": "#",
        }
        for _, row in matches.iterrows()
    ]


def get_by_genre(anilist_genre: str, jikan_genre_id: int, limit: int = 50) -> list:
    """Anime tagged with one genre/mood. Tries AniList (paginated, so a
    real browse-sized list comes back), then Jikan (single page, ~25
    max), then the local dataset (unlimited, just a genre substring
    match), in that order."""
    result = anilist_client.genre(anilist_genre, limit)
    if result:
        return result

    result = jikan_client.genre(jikan_genre_id, limit)
    if result:
        return result

    return _local_by_genre(anilist_genre, limit)


def get_mood_list(anilist_genre: str, jikan_genre_id: int, limit: int = 400) -> list:
    """Everything for one genre: live results first (newest titles), then
    the rest of the stored catalogue, without duplicates."""
    live = get_by_genre(anilist_genre, jikan_genre_id, limit)
    seen = {a["mal_id"] for a in live if a.get("mal_id")}
    rest = [a for a in _local_by_genre(anilist_genre, 5000) if a["mal_id"] not in seen]
    return live + rest


def get_catalogue(sort_by: str = "rating", genre_filter: str = None, page: int = 1, per_page: int = 50) -> tuple:
    """Get paginated full catalogue. Returns (list of anime, total_count, total_pages).
    sort_by: 'rating', 'popularity' (members), 'episodes', 'title'
    genre_filter: filter by genre name (case-insensitive substring match)
    page: 1-indexed page number
    per_page: how many per page"""
    pool = _anime_clean.copy()

    if genre_filter and genre_filter.strip():
        pool = pool[pool["genre"].str.contains(genre_filter, case=False, na=False, regex=False)]

    if sort_by == "rating":
        pool = pool.sort_values("rating", ascending=False)
    elif sort_by == "popularity":
        pool = pool.sort_values("members", ascending=False)
    elif sort_by == "episodes":
        pool = pool.sort_values("episodes", ascending=False)
    elif sort_by == "title":
        pool = pool.sort_values("name", ascending=True)

    total = len(pool)
    total_pages = (total + per_page - 1) // per_page
    page = max(1, min(page, total_pages)) if total_pages > 0 else 1

    start = (page - 1) * per_page
    end = start + per_page
    page_data = pool.iloc[start:end]

    cards = [
        {
            "mal_id": int(row["anime_id"]),
            "title": html.unescape(str(row["name"])),
            "images": {"jpg": {"large_image_url": ""}},
            "score": row.get("rating"),
            "episodes": row.get("episodes"),
            "genres": [{"name": html.unescape(g.strip())} for g in str(row.get("genre") or "").split(",") if g.strip()],
            "url": "#",
        }
        for _, row in page_data.iterrows()
    ]

    return cards, total, total_pages


def get_all_paginated(page: int = 1, page_size: int = 50, sort_by: str = "rating", 
                      sort_order: str = "desc", genre_filter: str = "", type_filter: str = "") -> tuple:
    """Paginate through the entire local dataset with filtering and sorting.
    Returns (total_anime, total_pages, page_data)."""
    df = _anime_clean.copy()
    
    if genre_filter:
        df = df[df["genre"].str.contains(genre_filter, case=False, na=False, regex=False)]
    
    if type_filter:
        df = df[df["type"].str.lower() == type_filter.lower()]
    
    if sort_by == "rating":
        df = df.sort_values("rating", ascending=(sort_order == "asc"), na_position="last")
    elif sort_by == "members":
        df = df.sort_values("members", ascending=(sort_order == "asc"), na_position="last")
    elif sort_by == "episodes":
        df = df.sort_values("episodes", ascending=(sort_order == "asc"), na_position="last")
    elif sort_by == "name":
        df = df.sort_values("name", ascending=(sort_order == "asc"), na_position="last")
    
    total = len(df)
    total_pages = (total + page_size - 1) // page_size
    page = max(1, min(page, total_pages if total_pages > 0 else 1))
    
    offset = (page - 1) * page_size
    page_data = df.iloc[offset:offset + page_size]
    
    results = [
        {
            "mal_id": int(row["anime_id"]),
            "title": html.unescape(str(row["name"])),
            "images": {"jpg": {"large_image_url": ""}},
            "score": row.get("rating"),
            "episodes": row.get("episodes"),
            "genres": [{"name": html.unescape(g.strip())} for g in str(row.get("genre") or "").split(",") if g.strip()],
            "url": "#",
            "type": row.get("type"),
            "members": int(row.get("members") or 0),
        }
        for _, row in page_data.iterrows()
    ]
    
    return total, total_pages, page, results
