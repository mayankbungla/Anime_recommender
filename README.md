# Anime Recommender

Two intentionally separate recommendation systems, built for different purposes:

1. **A live app**: real-time recommendations for any anime, powered entirely by the Jikan API
2. **An offline ML pipeline**: a leak-checked, evaluated collaborative-filtering model trained on 6M+ historical ratings

They are not wired together. See [Why two separate systems?](#why-two-separate-systems) below.

---

## 1. The live app

`app.py` is a Streamlit app with five pages, all backed by live calls to the [Jikan API](https://jikan.moe/) (a public MyAnimeList wrapper), with no local dataset, no trained model:

- **Because You Liked...**: community recommendations via Jikan's own `/recommendations` endpoint
- **Similar Vibes**: genre/theme similarity computed on-the-fly with TF-IDF + cosine similarity over the current top-anime pool
- **Browse by Mood**: genre-tag browsing (Action, Romance, Horror, etc.)
- **Airing Now**: current-season anime, sorted by score
- **All-Time Greatest**: the all-time top-rated list

All Jikan calls are cached (`st.cache_data`, 1hr TTL) to stay within the public API's rate limits.

### Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

No API keys or local data required, it talks to Jikan directly.

---

## 2. The ML pipeline

A classic offline collaborative-filtering pipeline on the [CooperUnion/anime-recommendations-database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) Kaggle dataset: ingest → clean → leak-checked split → train an SVD model (`scikit-surprise`) → evaluate on a held-out test set.

### Pipeline stats

```
ratings: 7,813,737 -> 6,314,631 rows retained (~81%)
users:   60,970
anime:   8,027
train:   5,048,185 rows
val:       633,223 rows
test:      633,223 rows
Leak check (train ∩ test on (user_id, anime_id)): 0, confirmed clean
```

### EDA highlights

The user-item matrix is 98.71% sparse: the typical user has rated well under 1% of the catalog, which is the central challenge collaborative filtering has to work around. Median anime has 109 ratings; median user has rated 57 anime. Average rating given is 7.81/10, consistent with the well-known positivity bias in rating datasets. Charts (rating distribution, genre distribution, most-rated anime, sparsity) are in `reports/figures/`.

### Model & evaluation

SVD (`n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42`):

```
val:  RMSE=1.1247  MAE=0.8422  n=633,223
test: RMSE=1.1226  MAE=0.8424  n=633,223
```

Val/test RMSE agree within 0.002, with a leak count of zero on the (user_id, anime_id) key, so no sign of leakage or split bias. RMSE ~1.0 to 1.3 on a 1 to 10 rating scale is the expected range for this dataset.

**Tuning.** A small grid over `n_factors` and `reg_all` (see `scripts/tune_cf.py`, results in `reports/hyperparameter_tuning.csv`) moves validation RMSE by less than 0.003, so the baseline `n_factors=50, reg_all=0.02` is kept. The takeaway is that this dataset is not factor-starved; accuracy is capped by sparsity, not model capacity.

**Lightweight factors.** The full surprise model pickles its whole trainset (~162MB). `scripts/export_factors.py` saves just the learned user/item embeddings and id maps to `models/factors/` (~27MB, an 84% cut). That is everything inference actually needs, and `predict.similar_items_from_factors()` uses it directly instead of unpickling the full model.

### Content model

A second, catalogue-scale content model (`src/anime_recommender/features/content_model.py`) represents each anime by a sentence-transformer embedding of its synopsis (`all-MiniLM-L6-v2`) and retrieves neighbours with `NearestNeighbors` (cosine). Synopses come from Jikan (`scripts/fetch_synopses.py`); anime without one fall back to title + genres. Because it needs no ratings, it covers the full ~12k catalogue, including the ~4k anime the CF model never sees for lack of ratings. This is the content half of the Week 5 hybrid.

### Reproduce it

```bash
pip install -r requirements.txt
python scripts/download_dataset.py            # -> data/raw/
python scripts/prepare_data.py                # -> data/processed/ (cleaned, leak-checked split, Parquet)
python scripts/eda.py                         # -> reports/figures/ (EDA charts + summary stats)
python -m src.anime_recommender.models.train_cf   # -> models/svd_cf_model.pkl
python scripts/export_factors.py              # -> models/factors/ (lightweight embeddings)
python scripts/fetch_synopses.py              # -> data/processed/anime_synopses.parquet (slow, resumable)
python scripts/build_content_model.py         # -> models/content/ (synopsis embeddings + retriever)
python scripts/sanity_check.py                # -> reports/sanity_check.md
```

Sample CF neighbours from the learned item factors (`scripts/sanity_check.py`):

```
Cowboy Bebop           -> Cowboy Bebop: Tengoku no Tobira, Samurai Champloo, Ghost in the Shell, Trigun
One Punch Man          -> Mob Psycho 100, Boku no Hero Academia, Hunter x Hunter (2011)
Steins;Gate            -> Kimi no Na wa., Fate/Zero, Psycho-Pass, Hunter x Hunter (2011)
Death Note             -> Shingeki no Kyojin, Kiseijuu, Code Geass, Fullmetal Alchemist: Brotherhood
```

These track taste, not just genre tags: Cowboy Bebop pulls in Samurai Champloo (same director) and One Punch Man pulls in Mob Psycho 100 (same creator), neither of which shares its main genre string.

---

## Why two separate systems?

The trained SVD model is **user-based collaborative filtering**: it only produces meaningful recommendations for `user_id`s that existed in its training set. The live app serves anonymous visitors searching arbitrary titles, who have no such `user_id`: a textbook cold-start mismatch.

Rather than force an awkward integration (fake user IDs, forced mappings, degraded quality), the two systems are kept deliberately separate, each evaluated on its own terms:

- The **app** is a product-engineering exercise: live API integration, caching, UX, on-the-fly similarity.
- The **model** is a methodology exercise: data cleaning, leak-checked splitting, trained collaborative filtering, real held-out evaluation metrics.

---

## Repo structure

```
app.py                      # Live Jikan-API Streamlit app
AnimeRecommender.ipynb      # Original prototype/EDA notebook (unmodified)
notebooks/                  # MAL-data notebook, pure DS narrative, separate from the above
data/
├── raw/                    # anime.csv, rating.csv (regenerate via scripts/download_dataset.py)
└── processed/              # splits + anime_synopses.parquet (regenerate via scripts)
models/
├── svd_cf_model.pkl        # Trained SVD model (full surprise object, ~162MB via Git LFS)
├── factors/                # Lightweight user/item embeddings + id maps (export_factors.py)
└── content/                # Synopsis embeddings + retriever (build_content_model.py)
reports/
├── figures/                # EDA charts from scripts/eda.py
├── hyperparameter_tuning.csv   # Day 16 tuning grid
└── sanity_check.md         # Day 20 CF vs content spot-check
scripts/
├── download_dataset.py
├── prepare_data.py
├── eda.py
├── tune_cf.py              # Day 16: SVD hyperparameter grid
├── export_factors.py       # Day 17: shrink the model to just its factors
├── fetch_synopses.py       # Day 18: pull synopses from Jikan (resumable)
├── build_content_model.py  # Day 18/19: embed synopses + fit retriever
├── sanity_check.py         # Day 20: eyeball both models
└── find_demo_users.py
src/anime_recommender/
├── data/                   # dataset.py, cleaning.py, split.py, jikan_client.py
├── features/               # similarity.py (live app), content_model.py (offline)
└── models/                 # train_cf.py, predict.py
tests/
```

## Development

```bash
pip install -r requirements.txt
pytest tests/
```

## Links

- Dataset: [CooperUnion/anime-recommendations-database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
- Live app: <!-- TODO: paste Streamlit Cloud URL once verified -->

---

## Author

**Mayank Bungla**

GitHub: https://github.com/mayankbungla

---

⭐ If you found this project useful, consider giving it a star!
