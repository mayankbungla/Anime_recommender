"""
Scores the CF model and the content model against the test split, using
precision/recall/ndcg at k plus catalogue coverage. Loads the SVD model
once and reuses it across every user, since reloading a 162MB pickle per
user would make this impractically slow.

Day 43 — logs each model's precision/recall/ndcg/coverage to MLflow as
its own run, so the Week 4 comparison has real experiment history too,
not just the CSV.

USAGE
    python scripts/evaluate_models.py
"""

import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anime_recommender.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k  # noqa: E402
from anime_recommender.models.predict import get_user_top_n  # noqa: E402
from anime_recommender.features.content_model import ContentRecommender  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "svd_cf_model.pkl"
CONTENT_DIR = ROOT / "models" / "content"
OUT_PATH = ROOT / "reports" / "model_comparison.csv"

# a rating of 7+ counts as "liked" for precision/recall's binary relevant set.
# ndcg uses the actual rating as graded relevance, this threshold only
# matters for precision and recall.
REL_THRESHOLD = 7
K = 10
N_USERS = 200
SEED = 42


def content_recommend_for_user(content, liked_ids, seen_ids, k):
    """
    ContentRecommender only does item-to-item lookup, so a user-level
    recommendation is built by pooling neighbours of everything the user
    liked in train, keeping each candidate's best similarity score, and
    dropping anything already seen.
    """
    scores = {}
    for anime_id in liked_ids:
        if anime_id not in content._row_by_id:
            continue
        neighbours = content.recommend(anime_id, k=k * 3)
        for row in neighbours.itertuples():
            if row.anime_id in seen_ids:
                continue
            scores[row.anime_id] = max(scores.get(row.anime_id, 0.0), row.similarity)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [anime_id for anime_id, _ in ranked[:k]]


def main():
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("anime-recommender-eval")

    print("Loading data...")
    train = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "ratings_test.parquet")
    catalog = pd.read_parquet(PROCESSED_DIR / "anime_clean.parquet")[["anime_id", "name"]]

    print("Loading CF model (one time)...")
    algo = joblib.load(MODEL_PATH)

    print("Loading content model...")
    content = ContentRecommender.load(CONTENT_DIR)

    seen_by_user = train.groupby("user_id")["anime_id"].apply(set).to_dict()
    liked_by_user_train = (
        train[train["rating"] >= REL_THRESHOLD].groupby("user_id")["anime_id"].apply(list).to_dict()
    )
    test_by_user = test.groupby("user_id").apply(
        lambda df: dict(zip(df["anime_id"], df["rating"]))
    ).to_dict()

    # only evaluate users who actually have something worth hitting in test,
    # scoring against a user with nothing relevant just adds noise
    eligible_users = [
        u for u, ratings in test_by_user.items()
        if any(r >= REL_THRESHOLD for r in ratings.values())
    ]
    rng = np.random.default_rng(SEED)
    sample = rng.choice(eligible_users, size=min(N_USERS, len(eligible_users)), replace=False)
    print(f"Evaluating on {len(sample)} users (out of {len(eligible_users):,} eligible).")

    rows = []
    cf_recommended_ever = set()
    content_recommended_ever = set()

    for i, user_id in enumerate(sample, 1):
        user_id = int(user_id)
        relevance = test_by_user[user_id]
        relevant_set = {a for a, r in relevance.items() if r >= REL_THRESHOLD}
        seen_ids = seen_by_user.get(user_id, set())

        cf_recs = get_user_top_n(user_id, k=K, anime_df=catalog, algo=algo)["anime_id"].tolist()
        cf_recommended_ever.update(cf_recs)
        rows.append({
            "user_id": user_id, "model": "cf",
            "precision": precision_at_k(cf_recs, relevant_set, K),
            "recall": recall_at_k(cf_recs, relevant_set, K),
            "ndcg": ndcg_at_k(cf_recs, relevance, K),
        })

        liked_ids = liked_by_user_train.get(user_id, [])
        content_recs = content_recommend_for_user(content, liked_ids, seen_ids, K)
        content_recommended_ever.update(content_recs)
        rows.append({
            "user_id": user_id, "model": "content",
            "precision": precision_at_k(content_recs, relevant_set, K),
            "recall": recall_at_k(content_recs, relevant_set, K),
            "ndcg": ndcg_at_k(content_recs, relevance, K),
        })

        if i % 25 == 0:
            print(f"  {i}/{len(sample)} users done")

    per_user = pd.DataFrame(rows)
    summary = per_user.groupby("model")[["precision", "recall", "ndcg"]].mean().reset_index()

    # coverage: % of the CF-trained catalogue that showed up in any
    # recommendation. Using the CF catalogue as the shared denominator
    # keeps the comparison fair, the content model can technically reach
    # a wider ~12k-anime catalogue, but only the CF-overlapping portion is
    # ever checkable against this test split anyway.
    catalog_size = train["anime_id"].nunique()
    summary["coverage"] = summary["model"].map({
        "cf": len(cf_recommended_ever) / catalog_size,
        "content": len(content_recommended_ever) / catalog_size,
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False)

    for _, row in summary.iterrows():
        with mlflow.start_run(run_name=f"eval_{row['model']}"):
            mlflow.log_param("model", row["model"])
            mlflow.log_param("k", K)
            mlflow.log_param("n_users", len(sample))
            mlflow.log_metric("precision", row["precision"])
            mlflow.log_metric("recall", row["recall"])
            mlflow.log_metric("ndcg", row["ndcg"])
            mlflow.log_metric("coverage", row["coverage"])

    print(f"\n{summary.to_string(index=False)}")
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
