# Run: streamlit run app.py
# Install: pip install streamlit pandas scikit-learn requests

import os
import pickle
import requests
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ─────────────────────────────────────────────
# Page Config  (must be FIRST streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS – dark cinematic theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    background-color: #0d0d0d;
    color: #e8e8e8;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111 0%, #1a0a0a 100%);
    border-right: 1px solid #2a2a2a;
}
[data-testid="stSidebar"] * { color: #e8e8e8 !important; }

/* ── Hero title ── */
.hero-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 3.8rem;
    letter-spacing: 0.12em;
    background: linear-gradient(90deg, #ff4e4e, #ff9900);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1;
}
.hero-sub {
    color: #888;
    font-size: 0.9rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Divider ── */
.divider {
    height: 2px;
    background: linear-gradient(90deg, #ff4e4e33, #ff990033, transparent);
    border: none;
    margin: 1rem 0 2rem 0;
}

/* ── Selectbox label ── */
label { color: #aaa !important; font-size: 0.8rem !important; letter-spacing: 0.1em; }

/* ── Recommend button ── */
.stButton > button {
    background: linear-gradient(90deg, #ff4e4e, #cc2200);
    color: white !important;
    border: none;
    border-radius: 4px;
    padding: 0.55rem 2.2rem;
    font-family: 'Bebas Neue', cursive;
    font-size: 1.1rem;
    letter-spacing: 0.1em;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

/* ── Anime card ── */
.anime-card {
    background: #181818;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s, transform 0.15s;
    position: relative;
    overflow: hidden;
}
.anime-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #ff4e4e, #ff9900);
    border-radius: 8px 0 0 8px;
}
.anime-card:hover { border-color: #ff4e4e55; transform: translateX(3px); }
.card-rank {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.6rem;
    color: #ff4e4e44;
    position: absolute;
    right: 12px; top: 8px;
    line-height: 1;
}
.card-name {
    font-size: 1rem;
    font-weight: 600;
    color: #f0f0f0;
    margin: 0 0 4px 0;
}
.card-meta {
    font-size: 0.75rem;
    color: #666;
    margin: 0;
}
.match-badge {
    display: inline-block;
    background: #ff4e4e18;
    border: 1px solid #ff4e4e55;
    color: #ff9966;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-top: 6px;
    letter-spacing: 0.05em;
}
.genre-chip {
    display: inline-block;
    background: #ffffff0d;
    color: #aaa;
    font-size: 0.65rem;
    padding: 2px 7px;
    border-radius: 20px;
    margin: 3px 3px 0 0;
    border: 1px solid #2a2a2a;
}

/* ── Info box ── */
.info-box {
    background: #ff4e4e0d;
    border: 1px solid #ff4e4e33;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #bbb;
    margin-bottom: 1.5rem;
}

/* ── Selectbox styling ── */
[data-testid="stSelectbox"] > div > div {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #e8e8e8 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data Loading  (cached so it runs ONCE)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load anime metadata and precomputed similarity matrix from pickle files."""
    data_dir = Path(".")
    anime_df = pd.DataFrame(pickle.load(open(data_dir / "anime.pkl", "rb")))
    sim_df   = pd.DataFrame(pickle.load(open(data_dir / "anime_sim.pkl", "rb")))
    return anime_df, sim_df


@st.cache_data(show_spinner=False)
def build_content_matrix(anime_df: pd.DataFrame):
    """
    Build TF-IDF cosine similarity matrix from genre column.
    Cached so it is computed only ONCE per session, not on every button click.
    """
    tfidf   = TfidfVectorizer(stop_words="english")
    matrix  = tfidf.fit_transform(anime_df["genre"].fillna(""))
    cos_sim = linear_kernel(matrix, matrix)
    return cos_sim


# ─────────────────────────────────────────────
# Jikan API  – fetch poster image (best-effort)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_poster(anime_name: str) -> str | None:
    """
    Query the free Jikan v4 API (MyAnimeList) for a poster image URL.
    Returns None if the request fails or no image is found.
    """
    try:
        resp = requests.get(
            "https://api.jikan.moe/v4/anime",
            params={"q": anime_name, "limit": 1},
            timeout=4,
        )
        if resp.status_code == 200:
            results = resp.json().get("data", [])
            if results:
                return results[0].get("images", {}).get("jpg", {}).get("image_url")
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# Recommendation logic
# ─────────────────────────────────────────────
def get_collaborative_recs(anime_name: str, sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return top-10 anime by collaborative similarity score.
    Raises KeyError if anime_name not in similarity matrix.
    """
    results = []
    for title in sim_df.sort_values(by=anime_name, ascending=False).index[1:11]:
        score = round(sim_df[title][anime_name] * 100, 2)
        results.append({"name": title, "match": score})
    return pd.DataFrame(results)


def get_content_recs(anime_name: str, anime_df: pd.DataFrame, cos_sim) -> pd.DataFrame:
    """
    Return top-10 anime by TF-IDF genre cosine similarity.
    Raises IndexError if anime_name not found in dataframe.
    """
    idx        = anime_df[anime_df["name"] == anime_name].index[0]
    scores     = list(enumerate(cos_sim[idx]))
    scores     = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]
    indices    = [i[0] for i in scores]
    sim_vals   = [round(s[1] * 100, 2) for s in scores]
    recs       = anime_df.iloc[indices][["name", "genre"]].copy()
    recs["match"] = sim_vals
    return recs.reset_index(drop=True)


# ─────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────
def render_card(rank: int, name: str, match: float | None = None, genres: str = ""):
    """Render a single styled anime result card."""
    match_html = f'<span class="match-badge">⚡ {match}% match</span>' if match is not None else ""

    genre_chips = ""
    if genres:
        for g in str(genres).split(",")[:4]:
            genre_chips += f'<span class="genre-chip">{g.strip()}</span>'

    st.markdown(f"""
    <div class="anime-card">
        <span class="card-rank">#{rank:02d}</span>
        <p class="card-name">{name}</p>
        {match_html}
        <div style="margin-top:6px">{genre_chips}</div>
    </div>
    """, unsafe_allow_html=True)


def render_poster_grid(recs: pd.DataFrame):
    """Show results as a 5-column poster grid when Jikan images are available."""
    cols = st.columns(5)
    for i, row in recs.iterrows():
        poster = fetch_poster(row["name"])
        with cols[i % 5]:
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown(
                    f'<div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;'
                    f'height:180px;display:flex;align-items:center;justify-content:center;'
                    f'font-size:2rem;">🎌</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(f'<p style="font-size:0.72rem;color:#ccc;margin:4px 0 0 0;text-align:center">'
                        f'{row["name"]}</p>', unsafe_allow_html=True)
            if "match" in row:
                st.markdown(f'<p style="font-size:0.68rem;color:#ff9966;text-align:center;margin:2px 0">'
                            f'⚡ {row["match"]}%</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Page renderers
# ─────────────────────────────────────────────
def page_collaborative(anime_titles, sim_df: pd.DataFrame):
    st.markdown("""
    <div class="info-box">
        📊 <b>Collaborative Filtering</b> — recommends anime that users with similar tastes also enjoyed.
        Based on user-rating patterns rather than genre or content.
    </div>
    """, unsafe_allow_html=True)

    selected = st.selectbox("Choose an anime", anime_titles, key="collab_select")
    view_mode = st.radio("Display as", ["Cards", "Poster Grid"], horizontal=True)

    if st.button("Get Recommendations", key="collab_btn"):
        with st.spinner("Finding similar anime…"):
            try:
                recs = get_collaborative_recs(selected, sim_df)
            except KeyError:
                st.error("❌ This anime isn't in the similarity matrix. Try another title.")
                return
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                return

        st.markdown(f"<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#888;font-size:0.8rem;letter-spacing:0.1em;text-transform:uppercase'>"
                    f"Top 10 similar to <b style='color:#ff9966'>{selected}</b></p>",
                    unsafe_allow_html=True)

        if view_mode == "Poster Grid":
            render_poster_grid(recs)
        else:
            for i, row in recs.iterrows():
                render_card(i + 1, row["name"], row["match"])


def page_content_based(anime_titles, anime_df: pd.DataFrame, cos_sim):

    selected = st.selectbox("Choose an anime", anime_titles, key="content_select")
    view_mode = st.radio("Display as", ["Cards", "Poster Grid"], horizontal=True)

    if st.button("Recommend", key="content_btn"):
        with st.spinner("Analysing genre patterns…"):
            try:
                recs = get_content_recs(selected, anime_df, cos_sim)
            except IndexError:
                st.error("❌ Anime not found in dataset.")
                return
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                return

        st.markdown(f"<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#888;font-size:0.8rem;letter-spacing:0.1em;text-transform:uppercase'>"
                    f"Top 10 genre matches for <b style='color:#ff9966'>{selected}</b></p>",
                    unsafe_allow_html=True)

        if view_mode == "Poster Grid":
            render_poster_grid(recs)
        else:
            for i, row in recs.iterrows():
                genres = row.get("genre", "")
                render_card(i + 1, row["name"], row["match"], genres)


# ─────────────────────────────────────────────
# App entry point
# ─────────────────────────────────────────────
def main():
    # Load data
    with st.spinner("Loading data…"):
        anime_df, sim_df = load_data()
        cos_sim          = build_content_matrix(anime_df)

    anime_titles = anime_df["name"].values

    # ── Sidebar ──
    with st.sidebar:
        st.markdown('<p class="hero-title">ANIME\nRECS</p>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub">Discover your next series</p>', unsafe_allow_html=True)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        page = st.radio(
            "Algorithm",
            ["📊 Collaborative Filtering", "🎭 Content-Based Filtering"],
            label_visibility="collapsed",
        )

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.7rem;color:#444;text-align:center'>"
            "Built with Streamlit · MyAnimeList data</p>",
            unsafe_allow_html=True,
        )

    # ── Main area header ──
    st.markdown('<p class="hero-title">ANIME RECOMMENDER</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Find your next obsession</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Route pages ──
    if page == "📊 Collaborative Filtering":
        page_collaborative(anime_titles, sim_df)
    else:
        page_content_based(anime_titles, anime_df, cos_sim)


if __name__ == "__main__":
    main()
