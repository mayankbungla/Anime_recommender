from pathlib import Path
import html
import logging
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv

from anime_recommender.data.jikan_client import jikan_anime, jikan_season_now
from anime_recommender.data.metadata_provider import (
    get_top, get_search, get_catalogue, get_airing, get_mood_list, attach_posters, get_all_paginated, get_by_genre, get_genres_for_ids,
)
from anime_recommender.data import tracking

load_dotenv()
logger = logging.getLogger(__name__)

# set API_BASE in the deployment environment (or a local .env, see .env.example)
# once the FastAPI backend has a real address; localhost only works when both
# run on one machine
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

if "session_id" not in st.session_state:
    st.session_state.session_id = tracking.new_session_id()

st.set_page_config(
    page_title="Anime Recs",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "🎌 Anime Recommender - discover your next favourite series.",
        "Get help": "https://github.com/mayankbungla/Anime_recommender",
        "Report a bug": "https://github.com/mayankbungla/Anime_recommender/issues"
    }
)

# -- Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --accent: #667eea;
    --accent-hover: #764ba2;
    --bg: #0f1419;
    --bg-sidebar: #1a1f2e;
    --border: #2a2a3a;
    --text: #e0e6ff;
    --text-muted: #a8afc7;
}

html, body, [class*="css"] {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-sidebar) 0%, var(--bg) 100%);
    border-right: 2px solid var(--accent);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-sidebar); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; }

.sidebar-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.15em;
    color: var(--accent) !important;
    margin-bottom: 0;
}
.sidebar-sub {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    color: #888 !important;
    text-transform: uppercase;
    margin-top: 0;
    margin-bottom: 1.5rem;
}

/* Page header */
.page-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    letter-spacing: 0.12em;
    color: var(--accent);
    margin-bottom: 0;
    line-height: 1;
}
.page-sub {
    font-size: 0.78rem;
    letter-spacing: 0.28em;
    color: #888;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 1.5rem;
}
hr.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1rem 0 1.8rem 0;
}

/* Info box */
.info-box {
    background: var(--bg-sidebar);
    border-left: 3px solid var(--accent);
    padding: 0.75rem 1rem;
    border-radius: 4px;
    font-size: 0.85rem;
    color: #aaa;
    margin-bottom: 1.5rem;
}

/* Anime cards */
.anime-card {
    background: var(--bg-sidebar);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.anime-card:hover {
    transform: translateY(-4px);
    border-color: var(--accent);
}
.card-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text);
    padding: 0.5rem 0.6rem 0.2rem;
    line-height: 1.3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.card-meta {
    font-size: 0.7rem;
    color: #777;
    padding: 0 0.6rem 0.6rem;
}
.score-badge {
    display: inline-block;
    background: rgba(102, 126, 234, 0.13);
    color: var(--accent);
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 0.68rem;
    font-weight: 600;
    margin-right: 4px;
}

/* Inputs */
div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] > div > div {
    background: var(--bg-sidebar) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
}

/* Button */
div[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.12em !important;
    padding: 0.5rem 2rem !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    background: var(--accent-hover) !important;
}

/* Spinner / status */
div[data-testid="stSpinner"] { color: var(--accent) !important; }

/* Hide Streamlit branding, keep the header so the sidebar toggle still works */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# -- Our own API (the trained hybrid model)

def api_search(query: str, limit: int = 30) -> list:
    """Title search against the local catalogue (handles typos)."""
    try:
        r = requests.get(f"{API_BASE}/search", params={"q": query, "limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as exc:
        logger.warning("search call failed: %s", exc)
        return []


def api_recommend(anime_id: int | None = None, anime_name: str = "", k: int = 10):
    """Calls our trained hybrid model. Returns the parsed response, or
    a dict with an 'error' key if the anime wasn't found or the API
    isn't reachable."""
    try:
        r = requests.post(f"{API_BASE}/recommend", json={"anime_id": anime_id, "anime_name": anime_name, "k": k}, timeout=20)
        if r.status_code == 404:
            return {"error": r.json().get("detail", "That anime isn't in our trained dataset.")}
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as exc:
        logger.warning("recommend call failed: %s", exc)
        return {"error": "Couldn't reach the recommendation API. Make sure it's running."}

def api_anime(anime_id: int):
    """Fallback lookup against our own dataset, used when Jikan doesn't
    have (or no longer has) an anime_id our model recommended."""
    try:
        r = requests.get(f"{API_BASE}/anime/{anime_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as exc:
        logger.warning("anime lookup failed: %s", exc)
        return {}


# -- UI helpers

def page_header(title: str, subtitle: str):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

def render_cards(anime_list: list, cols: int = 5, source_page: str = "unknown"):
    if not anime_list:
        st.warning("Nothing to show right now. Try a different search.")
        return

    anime_list = attach_posters(anime_list)
    tracking.log_shown(anime_list, source_page, st.session_state.session_id)

    groups = [anime_list[i:i+cols] for i in range(0, len(anime_list), cols)]
    idx = 0
    for group in groups:
        columns = st.columns(cols)
        for col, a in zip(columns, group):
            idx += 1
            img = a.get("images", {}).get("jpg", {}).get("large_image_url", "")
            title = a.get("title", "Unknown")
            anime_id = a.get("mal_id")
            score = a.get("score")
            match = a.get("hybrid_match")
            genres = ", ".join(g["name"] for g in a.get("genres", [])[:2])
            episodes = a.get("episodes")
            episodes = "?" if episodes is None or pd.isna(episodes) else int(episodes)
            url = a.get("url", "#")
            with col:
                if img:
                    st.image(img, width="stretch")
                else:
                    # no poster available (Browse Catalogue's local data has
                    # no image column) - show a placeholder instead of blank space
                    st.markdown(
                        '<div style="aspect-ratio:2/3;border-radius:6px;'
                        'background:linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);'
                        'display:flex;align-items:center;justify-content:center;">'
                        f'<span style="font-family:\'Bebas Neue\',sans-serif;font-size:2.4rem;color:#fff;opacity:0.85;">{title[:1].upper()}</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                match_html = f'<span class="score-badge">🎯 {match:.0%}</span>' if match is not None else ""
                score_html = f'<span class="score-badge">★ {score}</span>' if score else ""
                st.markdown(
                    f'<div class="card-title"><a href="{url}" target="_blank" style="color:var(--text);text-decoration:none;">{title}</a></div>'
                    f'<div class="card-meta">{match_html}{score_html}{episodes} ep · {genres}</div>',
                    unsafe_allow_html=True
                )
                # a real MAL/AniList link opens in a new tab, so a click on
                # it can't be logged server-side without a JS component;
                # this button is a lightweight, trackable stand-in signal
                if anime_id is not None and st.button("👍", key=f"like_{source_page}_{anime_id}_{idx}"):
                    tracking.log_click(anime_id, title, source_page, st.session_state.session_id)
                    st.toast(f"Noted — {title}")


# -- Pages

def page_browse_all():
    page_header("Browse Catalogue", "Explore the full collection")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sort_by = st.selectbox("Sort by", ["rating", "members", "episodes", "name"], key="sort_by")
    with col2:
        sort_order = st.selectbox("Order", ["Highest to Lowest", "Lowest to Highest"], key="sort_order")
    with col3:
        page_size = st.selectbox("Per page", [25, 50, 100], key="page_size")
    
    col_g, col_t = st.columns(2)
    with col_g:
        genre_filter = st.text_input("Filter by genre (optional)", key="genre_filter")
    with col_t:
        type_filter = st.selectbox("Filter by type", ["All", "TV", "Movie", "OVA", "Special"], key="type_filter")
    
    if type_filter == "All":
        type_filter = ""
    
    sort_order_value = "asc" if "Lowest" in sort_order else "desc"
    
    total, total_pages, current_page, results = get_all_paginated(
        page=st.session_state.get("current_page", 1),
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order_value,
        genre_filter=genre_filter,
        type_filter=type_filter,
    )
    
    st.write(f"Showing {len(results)} of {total} anime | Page {current_page}/{total_pages}")
    
    render_cards(results, cols=5, source_page="browse_catalogue")
    with col1:
        if current_page > 1 and st.button("← Previous"):
            st.session_state.current_page = current_page - 1
            st.rerun()
    with col3:
        st.write(f"Page {current_page}/{total_pages}")
    with col5:
        if current_page < total_pages and st.button("Next →"):
            st.session_state.current_page = current_page + 1
            st.rerun()


def render_paged(cards: list, key: str, source_page: str, step: int = 40):
    """Shows a long list in steps, with a count and a Show more button."""
    shown = st.session_state.get(key, step)
    st.caption(f"Showing {min(shown, len(cards))} of {len(cards)}")
    render_cards(cards[:shown], cols=5, source_page=source_page)
    if shown < len(cards) and st.button("Show more", key=f"{key}_more"):
        st.session_state[key] = shown + step
        st.rerun()


def rec_to_card(rec: dict) -> dict:
    genres = [{"name": g.strip()} for g in (rec.get("genre") or "").split(",") if g.strip()]
    return {
        "mal_id": rec["anime_id"],
        "title": html.unescape(rec["title"]),
        "images": {"jpg": {"large_image_url": ""}},
        "score": rec.get("rating"),
        "episodes": rec.get("episodes"),
        "genres": genres,
        "url": "#",
    }


def search_options(query: str) -> dict:
    """label -> anime id, from live results (English titles) and the catalogue (typo tolerant)."""
    by_id = {}
    for a in get_search(query, limit=12):
        if a.get("mal_id"):
            year = a.get("year")
            by_id[a["mal_id"]] = f"{a['title']}  ({year})" if year else a["title"]
    for a in api_search(query, 30):
        by_id.setdefault(a["anime_id"], html.unescape(a["title"]))
    labels, seen = {}, {}
    for anime_id, label in by_id.items():
        seen[label] = seen.get(label, 0) + 1
        labels[label if seen[label] == 1 else f"{label} #{seen[label]}"] = anime_id
    return labels


def page_community():
    page_header("Because You Liked...", "Pick a show, get similar picks")

    query = st.text_input("", placeholder="🔍  Search for an anime title...", label_visibility="collapsed")
    if not query.strip():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🔥 Popular right now")
        with st.spinner("Loading..."):
            top = get_top(limit=10)
        render_cards(top, cols=5, source_page="because_you_liked_popular")
        return

    with st.spinner("Searching..."):
        options = search_options(query)

    if not options:
        st.error("No anime found. Check the spelling or try another title.")
        return

    chosen_id = options[st.selectbox("", list(options.keys()), label_visibility="collapsed")]

    if st.button("Find Recommendations"):
        with st.spinner("Finding recommendations..."):
            payload = api_recommend(anime_id=chosen_id, k=300)
        if "error" in payload:
            st.error(payload["error"])
            return
        if not payload["recommendations"]:
            st.warning("No recommendations found for this title.")
            return
        st.session_state.rec_cards = [rec_to_card(r) for r in payload["recommendations"]]
        st.session_state.rec_title = payload["query_title"]
        st.session_state.rec_for = chosen_id
        st.session_state[f"shown_rec_{chosen_id}"] = 40

    if st.session_state.get("rec_for") == chosen_id and st.session_state.get("rec_cards"):
        st.markdown(f"#### Because you liked **{html.unescape(st.session_state.rec_title)}**:")
        render_paged(st.session_state.rec_cards, f"shown_rec_{chosen_id}", "because_you_liked_recommend")


def page_browse():
    page_header("Browse by Mood", "Not sure what you want? Start with a feeling.")

    MOODS = {
        "⚔️  Action & Hype": ("Action", 1),
        "💘  Romance & Feels": ("Romance", 22),
        "😂  Comedy & Chill": ("Comedy", 4),
        "🔮  Fantasy & Magic": ("Fantasy", 10),
        "🤯  Mystery & Thriller": ("Mystery", 7),
        "🤖  Sci-Fi & Mecha": ("Sci-Fi", 24),
        "👻  Horror & Dark": ("Horror", 14),
        "🏆  Sports & Hustle": ("Sports", 30),
    }

    cols = st.columns(4)
    for i, (label, genre_ids) in enumerate(MOODS.items()):
        if cols[i % 4].button(label, width="stretch"):
            st.session_state["mood_label"] = label
            st.session_state["mood_genre"] = genre_ids

    if "mood_genre" in st.session_state:
        label = st.session_state["mood_label"]
        anilist_genre, jikan_genre_id = st.session_state["mood_genre"]
        st.markdown(f"#### Top picks for: **{label}**")
        with st.spinner("Loading..."):
            results = get_mood_list(anilist_genre, jikan_genre_id)
        render_paged(results, f"shown_mood_{label}", "browse_by_mood")


def page_airing():
    page_header("Airing Now", "The freshest shows and what everyone is watching this season")

    with st.spinner("Fetching this season's anime..."):
        results = get_airing(limit=400)

    if not results:
        st.warning("Couldn't load seasonal data right now. Please try again in a moment.")
        return

    results_sorted = sorted(results, key=lambda x: x.get("score") or 0, reverse=True)
    render_paged(results_sorted, "shown_airing", "airing_now")


def page_top():
    page_header("All-Time Greatest", "The highest-rated anime of all time, as voted by millions")

    if "top_page" not in st.session_state:
        st.session_state.top_page = 1
    if "top_results" not in st.session_state:
        st.session_state.top_results = []

    if st.session_state.top_page == 1 or not st.session_state.top_results:
        with st.spinner("Loading..."):
            st.session_state.top_results = get_top(limit=500)

    if not st.session_state.top_results:
        st.warning("Couldn't load rankings right now. Please try again in a moment.")
        return

    per_page = 50
    total = len(st.session_state.top_results)
    total_pages = (total + per_page - 1) // per_page
    current_page = st.session_state.top_page

    start = (current_page - 1) * per_page
    end = start + per_page
    page_results = st.session_state.top_results[start:end]

    st.markdown(f"<div style='text-align:center;color:#888;font-size:0.9rem;'>Page {current_page} of {total_pages} ({total} total)</div>", unsafe_allow_html=True)

    render_cards(page_results, cols=5, source_page="all_time_greatest")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if current_page > 1:
            if st.button("← Previous", key="top_prev"):
                st.session_state.top_page -= 1
                st.rerun()
    with col3:
        if current_page < total_pages:
            if st.button("Next →", key="top_next"):
                st.session_state.top_page += 1
                st.rerun()


def page_catalogue():
    page_header("Browse Catalogue", "Find something new to watch")

    col1, col2, col3 = st.columns(3)
    with col1:
        sort_by = st.selectbox("Sort by", ["Rating", "Popularity", "Episodes", "Title"], key="cat_sort")
    with col2:
        genre_filter = st.text_input("Filter by genre", placeholder="e.g., Action", key="cat_genre")
    with col3:
        per_page = st.number_input("Per page", min_value=10, max_value=500, value=50, step=10, key="cat_perpage")

    sort_map = {"Rating": "rating", "Popularity": "popularity", "Episodes": "episodes", "Title": "title"}
    sort_key = sort_map[sort_by]

    if "catalogue_page" not in st.session_state:
        st.session_state.catalogue_page = 1

    with st.spinner("Loading..."):
        results, total, total_pages = get_catalogue(sort_by=sort_key, genre_filter=genre_filter or None, page=st.session_state.catalogue_page, per_page=per_page)

    if total == 0:
        st.warning("No anime found matching that filter.")
        return

    st.markdown(f"<div style='text-align:center;color:#888;font-size:0.9rem;'>{total} total • Page {st.session_state.catalogue_page} of {total_pages}</div>", unsafe_allow_html=True)

    render_cards(results, cols=5, source_page="browse_catalogue")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.session_state.catalogue_page > 1:
            if st.button("← Previous", key="cat_prev"):
                st.session_state.catalogue_page -= 1
                st.rerun()
    with col3:
        if st.session_state.catalogue_page < total_pages:
            if st.button("Next →", key="cat_next"):
                st.session_state.catalogue_page += 1
                st.rerun()


def page_analytics():
    page_header("Analytics", "What's actually getting shown and clicked")

    events = tracking.fetch_all_events()
    if not events:
        st.info("Nothing logged yet — browse a few pages first, then come back here.")
        return

    df = pd.DataFrame(events)
    shown = df[df["event_type"] == "shown"]
    clicked = df[df["event_type"] == "clicked"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shown", len(shown))
    c2.metric("Clicked", len(clicked))
    c3.metric("Distinct anime shown", shown["anime_id"].nunique())
    c4.metric("Sessions", df["session_id"].nunique())

    st.markdown("---")
    st.markdown("##### Most-recommended anime")
    counts = (
        shown.groupby(["anime_id", "title"]).size().rename("times_shown")
        .reset_index().sort_values("times_shown", ascending=False).head(10)
    )
    if counts.empty:
        st.write("No impressions logged yet.")
    else:
        st.dataframe(counts[["title", "times_shown"]], hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("##### Genre coverage")
    st.caption("Genres of every anime shown so far, from the local dataset.")
    genre_map = get_genres_for_ids(shown["anime_id"].unique().tolist())
    genre_rows = []
    for aid in shown["anime_id"]:
        for g in str(genre_map.get(aid) or "").split(","):
            g = g.strip()
            if g:
                genre_rows.append(g)

    if not genre_rows:
        st.write("None of the shown anime matched the local dataset (all came from live-only sources).")
    else:
        genre_counts = pd.Series(genre_rows).value_counts().head(15)
        st.bar_chart(genre_counts)
        st.caption(f"{genre_counts.index.nunique()} distinct genres represented so far.")


# -- Sidebar

with st.sidebar:
    st.markdown('<p class="sidebar-title">Anime Recs</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">Discover your next series</p>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "",
        [
            "🎯  Because You Liked...",
            "🎲  Browse by Mood",
            "📚  Browse Catalogue",
            "📡  Airing Now",
            "🏆  All-Time Greatest",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")


# -- Router

if "Because You Liked" in page:
    page_community()
elif "Browse by Mood" in page:
    page_browse()
elif "Browse Catalogue" in page:
    page_catalogue()
elif "Airing Now" in page:
    page_airing()
elif "All-Time Greatest" in page:
    page_top()
