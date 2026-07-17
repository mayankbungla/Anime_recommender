"""
Day 11 — Train a collaborative filtering model (SVD) on the cleaned,
per-user-split ratings, and evaluate on val/test.
"""

from pathlib import Path
import joblib
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"


def main():
    print("Loading processed data...")
    train_df = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "ratings_val.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "ratings_test.parquet")

    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(train_df[["user_id", "anime_id", "rating"]], reader)
    trainset = data.build_full_trainset()

    print("Training SVD model (this can take a few minutes)...")
    model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    model.fit(trainset)

    def evaluate(df, label):
        preds = [model.predict(row.user_id, row.anime_id, r_ui=row.rating) for row in df.itertuples()]
        rmse = accuracy.rmse(preds, verbose=False)
        mae = accuracy.mae(preds, verbose=False)
        print(f"{label}: RMSE={rmse:.4f}  MAE={mae:.4f}  n={len(df):,}")

    evaluate(val_df, "val")
    evaluate(test_df, "test")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "svd_cf_model.pkl")
    print(f"Saved model -> {MODEL_DIR / 'svd_cf_model.pkl'}")


if __name__ == "__main__":
    main()