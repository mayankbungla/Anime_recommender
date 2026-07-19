"""
Day 17 — Save the trained model's learned factors as .npy, not the whole
surprise model object.

surprise's SVD model pickles the ENTIRE Trainset alongside the learned
factors (raw ratings, id mappings, etc.) — that's why svd_cf_model.pkl is
162MB. But everything predict.py actually needs to make predictions is:
  - algo.pu   (user factors,  shape [n_users, n_factors])
  - algo.qi   (item factors,  shape [n_anime, n_factors])
  - the raw_id <-> inner_id mappings, so we can look up a real user_id /
    anime_id's row in pu / qi

This script extracts just those pieces and saves them separately. Much
smaller, and it's a natural first step toward not depending on the full
surprise object at inference time (predict.py can be updated to use these
instead, in a later pass — not done automatically here, since predict.py
already works and shouldn't be changed without you reviewing the diff).

USAGE
-----
    python scripts/export_factors.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "svd_cf_model.pkl"
FACTORS_DIR = Path(__file__).resolve().parents[1] / "models" / "factors"


def main():
    print(f"Loading {MODEL_PATH} (this is the slow part, ~30s+)...")
    algo = joblib.load(MODEL_PATH)
    trainset = algo.trainset

    FACTORS_DIR.mkdir(parents=True, exist_ok=True)

    print("Extracting learned factors...")
    np.save(FACTORS_DIR / "user_factors.npy", algo.pu)
    np.save(FACTORS_DIR / "item_factors.npy", algo.qi)

    # id mappings: inner index -> raw id, so a lookup by real user_id/anime_id
    # is just "find the raw id in this table, use that row number in pu/qi"
    user_map = pd.DataFrame({
        "inner_uid": range(trainset.n_users),
        "raw_user_id": [trainset.to_raw_uid(i) for i in range(trainset.n_users)],
    })
    item_map = pd.DataFrame({
        "inner_iid": range(trainset.n_items),
        "raw_anime_id": [trainset.to_raw_iid(i) for i in range(trainset.n_items)],
    })
    user_map.to_parquet(FACTORS_DIR / "user_id_map.parquet", index=False)
    item_map.to_parquet(FACTORS_DIR / "item_id_map.parquet", index=False)

    # size comparison
    old_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    new_size_mb = sum(f.stat().st_size for f in FACTORS_DIR.glob("*")) / (1024 * 1024)

    print(f"\nOld full model:     {old_size_mb:.1f} MB  ({MODEL_PATH})")
    print(f"New factors only:   {new_size_mb:.1f} MB  ({FACTORS_DIR}/)")
    print(f"Shrunk by:          {100 * (1 - new_size_mb / old_size_mb):.1f}%")
    print(f"\nuser_factors.npy shape: {algo.pu.shape}")
    print(f"item_factors.npy shape: {algo.qi.shape}")


if __name__ == "__main__":
    main()
