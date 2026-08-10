import streamlit as st
import requests

JIKAN = "https://api.jikan.moe/v4"
API_BASE = "http://127.0.0.1:8000"  # FastAPI backend, see src/anime_recommender/api

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

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# -- Jikan API helpers (live display metadata, never used for ranking)

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
        r = requests.get(f"{JIKAN}/anime", params={"genres": genre_id, "order_by": "score", "sort": "desc", "limit": limit, "sfw": True}, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception:
        return []


# -- Our own API (the trained hybrid model)

def api_recommend(anime_name: str, k: int = 10):
    """Calls our trained hybrid model. Returns the parsed response, or
    a dict with an 'error' key if the anime wasn't found or the API
    isn't reachable."""
    try:
        r = requests.post(f"{API_BASE}/recommend", json={"anime_name": anime_name, "k": k}, timeout=10)
        if r.status_code == 404:
            return {"error": r.json().get("detail", "That anime isn't in our trained dataset.")}
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return {"error": "Couldn't reach the recommendation API. Make sure it's running."}

def api_anime(anime_id: int):
    """Fallback lookup against our own dataset, used when Jikan doesn't
    have (or no longer has) an anime_id our model recommended."""
    try:
        r = requests.get(f"{API_BASE}/anime/{anime_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return {}


# -- UI helpers

def page_header(title: str, subtitle: str):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

def info_box(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def render_cards(anime_list: list, cols: int = 5):
    if not anime_list:
        st.warning("Nothing to show right now. Try a different search.")
        return
    groups = [anime_list[i:i+cols] for i in range(0, len(anime_list), cols)]
    for group in groups:
        columns = st.columns(cols)
        for col, a in zip(columns, group):
            img = a.get("images", {}).get("jpg", {}).get("large_image_url", "")
            title = a.get("title", "Unknown")
            score = a.get("score")
            match = a.get("hybrid_match")
            genres = ", ".join(g["name"] for g in a.get("genres", [])[:2])
            episodes = a.get("episodes") or "?"
            url = a.get("url", "#")
            with col:
                if img:
                    st.image(img, use_container_width=True)
                match_html = f'<span class="score-badge">🎯 {match:.0%}</span>' if match is not None else ""
                score_html = f'<span class="score-badge">★ {score}</span>' if score else ""
                st.markdown(
                    f'<div class="card-title"><a href="{url}" target="_blank" style="color:var(--text);text-decoration:none;">{title}</a></div>'
                    f'<div class="card-meta">{match_html}{score_html}{episodes} ep · {genres}</div>',
                    unsafe_allow_html=True
                )


# -- Pages

def page_community(n_recs=10):
    page_header("Because You Liked...", "Recommendations from our own trained hybrid model")
    info_box("Pick any anime and our hybrid model, collaborative filtering, content embeddings, and popularity trained on real MyAnimeList ratings, finds what to watch next. The 🎯 badge shows how strong the match is.")

    query = st.text_input("", placeholder="🔍  Search for an anime title...", label_visibility="collapsed")
    if not query:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🔥 Popular right now")
        with st.spinner("Loading..."):
            top = jikan_top(limit=10)
        render_cards(top, cols=5)
        return

    with st.spinner("Searching..."):
        results = jikan_search(query, limit=10)

    if not results:
        st.error("No anime found. Try a different spelling.")
        return

    options = {f"{a['title']}  ({a.get('year') or '?'})": a["title"] for a in results}
    chosen_label = st.selectbox("", list(options.keys()), label_visibility="collapsed")
    chosen_title = options[chosen_label]

    if st.button("Find Recommendations"):
        with st.spinner("Scoring against our trained model..."):
            payload = api_recommend(chosen_title, n_recs)

        if "error" in payload:
            st.error(payload["error"])
            return
        if not payload["recommendations"]:
            st.warning("No recommendations found for this title.")
            return

        with st.spinner("Fetching details..."):
            details = []
            for rec in payload["recommendations"]:
                d = jikan_anime(rec["anime_id"])
                if not d:
                    # Jikan doesn't have this id (delisted/renamed), fall
                    # back to our own dataset so the card still shows.
                    local = api_anime(rec["anime_id"])
                    if local:
                        d = {
                            "title": local.get("name", rec["title"]),
                            "score": local.get("rating"),
                            "episodes": local.get("episodes"),
                            "genres": [{"name": g.strip()} for g in (local.get("genre") or "").split(",") if g.strip()],
                        }
                if d:
                    d = dict(d)
                    d["hybrid_match"] = rec["hybrid_score"]
                    details.append(d)

        st.markdown(f"#### Because you liked **{payload['query_title']}**:")
        render_cards(details, cols=5)


def page_browse():
    page_header("Browse by Mood", "Not sure what you want? Start with a feeling.")

    MOODS = {
        "⚔️  Action & Hype": 1,
        "💘  Romance & Feels": 22,
        "😂  Comedy & Chill": 4,
        "🔮  Fantasy & Magic": 10,
        "🤯  Mystery & Thriller": 7,
        "🤖  Sci-Fi & Mecha": 24,
        "👻  Horror & Dark": 14,
        "🏆  Sports & Hustle": 30,
    }

    cols = st.columns(4)
    selected_mood = None
    for i, (label, gid) in enumerate(MOODS.items()):
        if cols[i % 4].button(label, use_container_width=True):
            selected_mood = (label, gid)
            st.session_state["mood_label"] = label
            st.session_state["mood_id"] = gid

    if "mood_id" in st.session_state:
        label = st.session_state["mood_label"]
        gid = st.session_state["mood_id"]
        st.markdown(f"#### Top picks for: **{label}**")
        with st.spinner("Loading..."):
            results = jikan_genre(gid, limit=20)
        render_cards(results, cols=5)


def page_airing():
    page_header("Airing Now", "The freshest shows and what everyone is watching this season")

    with st.spinner("Fetching this season's anime..."):
        results = jikan_season_now(limit=25)

    if not results:
        st.warning("Couldn't load seasonal data right now. Please try again in a moment.")
        return

    results_sorted = sorted(results, key=lambda x: x.get("score") or 0, reverse=True)
    render_cards(results_sorted, cols=5)


def page_top():
    page_header("All-Time Greatest", "The highest-rated anime of all time, as voted by millions")

    with st.spinner("Loading..."):
        results = jikan_top(limit=50)

    if not results:
        st.warning("Couldn't load rankings right now. Please try again in a moment.")
        return

    render_cards(results, cols=5)


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
            "📡  Airing Now",
            "🏆  All-Time Greatest",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    n_recs = st.slider("Recommendations to show", min_value=5, max_value=20, value=10, step=1)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("GitHub", "https://github.com/mayankbungla/Anime_recommender", use_container_width=True)
    with col2:
        st.link_button("Dataset", "https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database", use_container_width=True)
    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem;color:#555;text-align:center;">Powered by MyAnimeList</p>', unsafe_allow_html=True)


# -- Router

if "Because You Liked" in page:
    page_community(n_recs)
elif "Browse by Mood" in page:
    page_browse()
elif "Airing Now" in page:
    page_airing()
elif "All-Time Greatest" in page:
    page_top()
