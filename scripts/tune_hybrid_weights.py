"""
Tries a handful of alpha/beta/gamma weight combinations for the hybrid
scorer and evaluates each with the same precision/recall/ndcg/coverage
metrics used for the CF and content models, so the comparison is
apples to apples with reports/model_comparison.csv.

Loads the CF model and content model once and reuses them across every
user and every weight combo, same reasoning as evaluate_models.py.

USAGE
    python scripts/tune_hybrid_weights.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anime_recommender.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k  # noqa: E402
from anime_recommender.features.content_model import ContentRecommender  # noqa: E402
from anime_recommender.models.hybrid import (  # noqa: E402
    build_popularity_scores, build_user_taste_vector, hybrid_score,
)

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "svd_cf_model.pkl"
CONTENT_DIR = ROOT / "models" / "content"
OUT_PATH = ROOT / "reports" / "hybrid_weight_tuning.csv"

REL_THRESHOLD = 7
K = 10
N_USERS = 200
SEED = 42

# same weights sum to 1 each time, just shifting who gets the most say
WEIGHT_COMBOS = [
    {"alpha": 0.5, "beta": 0.3, "gamma": 0.2},   # design doc default
    {"alpha": 0.7, "beta": 0.2, "gamma": 0.1},   # lean on CF more
    {"alpha": 0.3, "beta": 0.5, "gamma": 0.2},   # lean on content more
    {"alpha": 0.34, "beta": 0.33, "gamma": 0.33},  # roughly equal
]


def main():
    print("Loading data...")
    train = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "ratings_test.parquet")

    algo = joblib.load(MODEL_PATH)
    content = ContentRecommender.load(CONTENT_DIR)
    popularity = build_popularity_scores(train)
    catalog_ids = train["anime_id"].unique().tolist()

    seen_by_user = train.groupby("user_id")["anime_id"].apply(set).to_dict()
    liked_by_user = train[train["rating"] >= REL_THRESHOLD].groupby("user_id")["anime_id"].apply(list).to_dict()
    test_by_user = test.groupby("user_id").apply(lambda df: dict(zip(df["anime_id"], df["rating"]))).to_dict()

    eligible_users = [u for u, r in test_by_user.items() if any(v >= REL_THRESHOLD for v in r.values())]
    rng = np.random.default_rng(SEED)
    sample = rng.choice(eligible_users, size=min(N_USERS, len(eligible_users)), replace=False)
    print(f"Evaluating {len(WEIGHT_COMBOS)} weight combos on {len(sample)} users.")

    results = []
    for combo in WEIGHT_COMBOS:
        rows = []
        recommended_ever = set()

        for user_id in sample:
            user_id = int(user_id)
            relevance = test_by_user[user_id]
            relevant_set = {a for a, r in relevance.items() if r >= REL_THRESHOLD}
            seen_ids = seen_by_user.get(user_id, set())
            liked_ids = liked_by_user.get(user_id, [])

            # taste vector doesn't depend on the weights, but recomputing
            # it once per combo keeps this loop simple, it's cheap either way
            taste = build_user_taste_vector(liked_ids, content)
            scored = [
                (aid, hybrid_score(user_id, aid, algo, content, taste, popularity, **combo))
                for aid in catalog_ids if aid not in seen_ids
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            recs = [aid for aid, _ in scored[:K]]
            recommended_ever.update(recs)

            rows.append({
                "precision": precision_at_k(recs, relevant_set, K),
                "recall": recall_at_k(recs, relevant_set, K),
                "ndcg": ndcg_at_k(recs, relevance, K),
            })

        avg = pd.DataFrame(rows).mean()
        results.append({
            **combo,
            "precision": avg["precision"],
            "recall": avg["recall"],
            "ndcg": avg["ndcg"],
            "coverage": len(recommended_ever) / len(catalog_ids),
        })
        print(f"  alpha={combo['alpha']} beta={combo['beta']} gamma={combo['gamma']}  "
              f"ndcg={avg['ndcg']:.4f}")

    summary = pd.DataFrame(results).sort_values("ndcg", ascending=False)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False)

    print(f"\n{summary.to_string(index=False)}")
    best = summary.iloc[0]
    print(f"\nBest combo by ndcg: alpha={best['alpha']} beta={best['beta']} gamma={best['gamma']}")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
