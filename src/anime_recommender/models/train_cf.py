"""
Day 11 — Train a collaborative filtering model (SVD) on the cleaned,
per-user-split ratings, and evaluate on val/test.

Day 43 — logs hyperparameters and val/test metrics to MLflow, so each
training run leaves a real record instead of just terminal output.
Runs are stored locally in a SQLite file (mlflow.db) at the repo root -
MLflow's plain file-based store is deprecated in current versions, so
this uses the format MLflow itself now recommends. View runs with
`mlflow ui --backend-store-uri sqlite:///mlflow.db` from the repo root.
The model file itself stays on disk as before - it's not duplicated
into MLflow's run storage.
"""

from pathlib import Path
import joblib
import mlflow
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
MLFLOW_DB = Path(__file__).resolve().parents[3] / "mlflow.db"

PARAMS = {"n_factors": 50, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02, "random_state": 42}


def main():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("anime-recommender-cf")

    print("Loading processed data...")
    train_df = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "ratings_val.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "ratings_test.parquet")

    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(train_df[["user_id", "anime_id", "rating"]], reader)
    trainset = data.build_full_trainset()

    with mlflow.start_run(run_name="svd_baseline"):
        mlflow.log_params(PARAMS)

        print("Training SVD model (this can take a few minutes)...")
        model = SVD(**PARAMS)
        model.fit(trainset)

        def evaluate(df, label):
            preds = [model.predict(row.user_id, row.anime_id, r_ui=row.rating) for row in df.itertuples()]
            rmse = accuracy.rmse(preds, verbose=False)
            mae = accuracy.mae(preds, verbose=False)
            print(f"{label}: RMSE={rmse:.4f}  MAE={mae:.4f}  n={len(df):,}")
            mlflow.log_metric(f"{label}_rmse", rmse)
            mlflow.log_metric(f"{label}_mae", mae)

        evaluate(val_df, "val")
        evaluate(test_df, "test")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "svd_cf_model.pkl")
    print(f"Saved model -> {MODEL_DIR / 'svd_cf_model.pkl'}")


if __name__ == "__main__":
    main()