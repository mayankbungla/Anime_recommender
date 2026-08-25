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
