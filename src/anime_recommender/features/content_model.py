"""
Content-based model over anime synopses.

Day 18: represent each anime by a sentence-transformer embedding of its
synopsis, instead of TF-IDF over genre strings. Synopses carry tone,
setting and plot that bare genre tags miss, so "two rival chefs" and
"two rival duelists" no longer look identical just because both are
tagged Shounen.

Day 19: retrieve similar anime with sklearn NearestNeighbors (cosine)
over those embeddings, instead of building a full dense N x N cosine
matrix. NearestNeighbors keeps only the vectors and searches on demand,
so this scales to the whole 12k catalogue without a 12k x 12k matrix.
FAISS would be the next step if the catalogue grew into the millions;
at this size exact search is instant and not worth the extra dependency.

The live Streamlit app still uses the small-pool TF-IDF in similarity.py.
This module is the offline, catalogue-scale content model used by the
Week 4 evaluation and the Week 5 hybrid.

sentence-transformers is imported lazily inside embed_texts so the rest
of this module (the retriever, load/save) works without it installed,
which keeps the sanity check and tests runnable on a machine that only
has the pre-built embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CONTENT_DIR = Path(__file__).resolve().parents[3] / "models" / "content"


def build_catalog_text(row: pd.Series) -> str:
    """Text we embed for one anime.

    Prefer the synopsis. When it's missing or too short to be useful
    (some obscure entries have none), fall back to title + genres so the
    anime still gets a sensible vector and stays in the catalogue. This
    is what lets the content model cover the ~4k anime that never made it
    into the CF model for lack of ratings.
    """
    synopsis = str(row.get("synopsis") or "").strip()
    if len(synopsis) >= 40:
        return synopsis

    name = str(row.get("name") or "").strip()
    genre = str(row.get("genre") or "").replace(",", " ").strip()
    return f"{name}. {genre}".strip(". ").strip() or "unknown"


def embed_texts(texts: list[str], model_name: str = MODEL_NAME,
                batch_size: int = 64) -> np.ndarray:
    """Encode texts into L2-normalised float32 embeddings.

    Normalised vectors mean a plain dot product equals cosine similarity,
    which is what the retriever below relies on.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)


class ContentRecommender:
    """Nearest-neighbour retrieval over precomputed anime embeddings."""

    def __init__(self, embeddings: np.ndarray, catalog: pd.DataFrame):
        # catalog is aligned row-for-row with embeddings and carries at
        # least anime_id and name for lookups and readable output.
        if len(embeddings) != len(catalog):
            raise ValueError("embeddings and catalog must have the same length")

        self.embeddings = embeddings
        self.catalog = catalog.reset_index(drop=True)
        self._row_by_id = {aid: i for i, aid in enumerate(self.catalog["anime_id"])}

        # cosine on already-normalised vectors; brute force is exact and
        # fast enough for a catalogue this size.
        self._nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._nn.fit(self.embeddings)

    def recommend(self, anime_id: int, k: int = 10) -> pd.DataFrame:
        """Top-k anime closest to anime_id, excluding itself."""
        if anime_id not in self._row_by_id:
            raise ValueError(f"anime_id {anime_id} is not in the content catalogue.")

        row = self._row_by_id[anime_id]
        # ask for k+1 because the query item is its own nearest neighbour
        distances, indices = self._nn.kneighbors(
            self.embeddings[row].reshape(1, -1), n_neighbors=k + 1
        )

        out = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == row:
                continue
            rec = self.catalog.iloc[idx]
            out.append({
                "anime_id": int(rec["anime_id"]),
                "title": rec["name"],
                "similarity": float(1.0 - dist),  # cosine distance -> similarity
            })
            if len(out) == k:
                break

        return pd.DataFrame(out)

    def save(self, content_dir: Path | str = CONTENT_DIR) -> None:
        content_dir = Path(content_dir)
        content_dir.mkdir(parents=True, exist_ok=True)
        np.save(content_dir / "synopsis_embeddings.npy", self.embeddings)
        self.catalog.to_parquet(content_dir / "content_catalog.parquet", index=False)

    @classmethod
    def load(cls, content_dir: Path | str = CONTENT_DIR) -> "ContentRecommender":
        content_dir = Path(content_dir)
        embeddings = np.load(content_dir / "synopsis_embeddings.npy")
        catalog = pd.read_parquet(content_dir / "content_catalog.parquet")
        return cls(embeddings, catalog)
