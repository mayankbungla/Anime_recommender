import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

JIKAN = "https://api.jikan.moe/v4"

st.set_page_config(
    page_title="Anime Recs",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    background-color: #0d0d0d;
    color: #e8e0d5;
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #2a2a2a;
}
section[data-testid="stSidebar"] * { color: #e8e0d5 !important; }

.sidebar-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.15em;
    color: #e05a2b !important;
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
    color: #e05a2b;
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
    border-top: 1px solid #2a2a2a;
    margin: 1rem 0 1.8rem 0;
}

/* Info box */
.info-box {
    background: #181818;
    border-left: 3px solid #e05a2b;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    font-size: 0.85rem;
    color: #aaa;
    margin-bottom: 1.5rem;
}

/* Anime cards */
.anime-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.anime-card:hover {
    transform: translateY(-4px);
    border-color: #e05a2b;
}
.card-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #e8e0d5;
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
    background: #e05a2b22;
    color: #e05a2b;
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 0.68rem;
    font-weight: 600;
    margin-right: 4px;
}

/* Inputs */
div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] > div > div {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    color: #e8e0d5 !important;
}

/* Button */
div[data-testid="stButton"] > button {
    background: #e05a2b !important;
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
    background: #c44d24 !important;
}

/* Spinner / status */
div[data-testid="stSpinner"] { color: #e05a2b !important; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Jikan API helpers ──────────────────────────────────────────────────────────

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
        r = requests.get(f"{JIKAN}/anime", params={"genres": genre_id, "order_by": "score", "sort": "desc", "limit": limit, "sfw": True}, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception:
        return []


# ── UI helpers ─────────────────────────────────────────────────────────────────

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
            genres = ", ".join(g["name"] for g in a.get("genres", [])[:2])
            episodes = a.get("episodes") or "?"
            url = a.get("url", "#")
            with col:
                if img:
                    st.image(img, use_container_width=True)
                score_html = f'<span class="score-badge">★ {score}</span>' if score else ""
                st.markdown(
                    f'<div class="card-title"><a href="{url}" target="_blank" style="color:#e8e0d5;text-decoration:none;">{title}</a></div>'
                    f'<div class="card-meta">{score_html}{episodes} ep · {genres}</div>',
                    unsafe_allow_html=True
                )


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_community(n_recs=10):
    page_header("Who Else Watched This?", "Find what fans of your favourite anime also loved")
    info_box("Pick any anime and we'll show you what its fans recommend — based on real MyAnimeList community votes.")

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

    options = {f"{a['title']}  ({a.get('year') or '?'})": a["mal_id"] for a in results}
    chosen_label = st.selectbox("", list(options.keys()), label_visibility="collapsed")
    chosen_id = options[chosen_label]
    chosen_title = chosen_label.split("  (")[0]

    if st.button("Find Recommendations"):
        with st.spinner("Fetching community picks..."):
            recs = jikan_recommendations(chosen_id)

        if not recs:
            st.warning("No community recommendations found for this title yet.")
            return

        st.markdown(f"#### Fans of **{chosen_title}** also loved:")
        details = []
        for rec in recs[:n_recs]:
            mid = rec.get("entry", {}).get("mal_id")
            if mid:
                d = jikan_anime(mid)
                if d:
                    details.append(d)

        render_cards(details, cols=5)


def page_similar(n_recs=10):
    page_header("Similar Vibes", "Discover anime that feel just like the one you love")
    info_box("We match anime by their genres and themes. Great for when you want \"more of this energy\" but don't know what to watch next.")

    query = st.text_input("", placeholder="🔍  Search for an anime title...", label_visibility="collapsed")
    if not query:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🏆 Top rated anime")
        with st.spinner("Loading..."):
            top = jikan_top(limit=10)
        render_cards(top, cols=5)
        return

    with st.spinner("Searching..."):
        results = jikan_search(query, limit=10)

    if not results:
        st.error("No anime found. Try a different spelling.")
        return

    options = {f"{a['title']}  ({a.get('year') or '?'})": a for a in results}
    chosen_label = st.selectbox("", list(options.keys()), label_visibility="collapsed")
    chosen = options[chosen_label]

    if st.button("Find Similar Anime"):
        with st.spinner("Analysing genres..."):
            pool = jikan_top(limit=50)
            chosen_id = chosen["mal_id"]
            pool_ids = {a["mal_id"] for a in pool}
            if chosen_id not in pool_ids:
                pool = [chosen] + pool

            def genre_str(a):
                g = " ".join(x["name"] for x in a.get("genres", []))
                t = " ".join(x["name"] for x in a.get("themes", []))
                return f"{g} {t}".strip() or "unknown"

            df = pd.DataFrame([{
                "mal_id": a["mal_id"],
                "title": a["title"],
                "g": genre_str(a),
                "_d": a
            } for a in pool]).drop_duplicates("mal_id").reset_index(drop=True)

            tfidf = TfidfVectorizer(stop_words="english")
            matrix = tfidf.fit_transform(df["g"])
            sim = linear_kernel(matrix, matrix)

            idx_list = df.index[df["mal_id"] == chosen_id].tolist()
            if not idx_list:
                st.warning("Couldn't find this anime in our current pool. Try again shortly.")
                return

            idx = idx_list[0]
            scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)
            top_idx = [i for i, _ in scores if i != idx][:n_recs]
            similar = [df.iloc[i]["_d"] for i in top_idx]
            top_scores = [scores[i][1] for i in range(len(scores)) if scores[i][0] != idx][:n_recs]

        st.markdown(f"#### Anime with a similar vibe to **{chosen['title']}**:")
        render_cards(similar, cols=5)

        if top_scores:
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Match Score", f"{sum(top_scores)/len(top_scores):.0%}")
            c2.metric("Top Match", f"{top_scores[0]:.0%}", similar[0]["title"] if similar else "")
            c3.metric("Results Found", len(similar))


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
    page_header("Airing Now", "The freshest shows — what everyone is watching this season")

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


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="sidebar-title">Anime Recs</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">Discover your next series</p>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "",
        [
            "🎯  Because You Liked...",
            "🎭  Similar Vibes",
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


# ── Router ─────────────────────────────────────────────────────────────────────

if "Because You Liked" in page:
    page_community(n_recs)
elif "Similar Vibes" in page:
    page_similar(n_recs)
elif "Browse by Mood" in page:
    page_browse()
elif "Airing Now" in page:
    page_airing()
elif "All-Time Greatest" in page:
    page_top()
