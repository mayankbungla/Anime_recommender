"""
Genre/theme similarity calculation.

This was previously inlined inside app.py's `page_similar` function.
Pulling it out means the same logic can be reused by:
  - the live Streamlit page (small in-memory pool from Jikan)
  - the offline evaluation scripts (Week 4), once we're scoring a
    content-based model against a real dataset instead of a live pool

NOTE: this TF-IDF-on-genre-strings approach is the placeholder content
model referenced in the plan. Day 18 replaces it with sentence-transformer
embeddings over synopses, and Day 19 replaces the dense cosine matrix here
with approximate nearest neighbours (NearestNeighbors/FAISS) so it scales
past a small in-memory pool.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def genre_theme_string(anime: dict) -> str:
    """Flatten an anime record's genres + themes into one string for TF-IDF."""
    g = " ".join(x["name"] for x in anime.get("genres", []))
    t = " ".join(x["name"] for x in anime.get("themes", []))
    return f"{g} {t}".strip() or "unknown"


def build_pool_dataframe(pool: list[dict]) -> pd.DataFrame:
    """Turn a list of raw Jikan anime dicts into a de-duplicated dataframe
    with a genre/theme text column ready for vectorising."""
    df = pd.DataFrame([{
        "mal_id": a["mal_id"],
        "title": a["title"],
        "g": genre_theme_string(a),
        "_d": a,
    } for a in pool]).drop_duplicates("mal_id").reset_index(drop=True)
    return df


def compute_similarity_matrix(df: pd.DataFrame):
    """Fit TF-IDF over the genre/theme text column and return the full
    cosine similarity matrix for the pool."""
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["g"])
    return linear_kernel(matrix, matrix)


def top_similar(df: pd.DataFrame, sim_matrix, chosen_id: int, n_recs: int = 10):
    """Return (similar_records, scores) for the n_recs closest anime to
    chosen_id, excluding chosen_id itself. Returns ([], []) if chosen_id
    isn't in the pool."""
    idx_list = df.index[df["mal_id"] == chosen_id].tolist()
    if not idx_list:
        return [], []

    idx = idx_list[0]
    scores = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)
    top_idx = [i for i, _ in scores if i != idx][:n_recs]
    top_scores = [s for i, s in scores if i != idx][:n_recs]
    similar = [df.iloc[i]["_d"] for i in top_idx]
    return similar, top_scores
