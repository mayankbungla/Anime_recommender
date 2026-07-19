"""
Day 16 — Tune hyperparameters of the SVD model.

Tries a handful of (n_factors, reg_all) combinations, evaluates each on
the validation split (never touches test — that's saved for the very
end, per the train/val/test discipline from Day 10), and logs everything
to a CSV so you have a record of what was tried and why the final choice
was picked.

n_factors: how many "hidden traits" each user/anime gets represented by.
  Too few -> the model can't capture enough nuance (underfitting).
  Too many -> the model can memorize noise in the training data
  (overfitting) and val performance gets worse even as train gets better.

reg_all: regularization strength. Higher values penalize large factor
  values more, which fights overfitting at the cost of some accuracy on
  data the model has already seen. Lower values let the model fit the
  training data more closely, risking overfitting.

USAGE
-----
    python scripts/tune_cf.py
"""

from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# Day 16 asks for 2-3 hyperparameters. We vary n_factors and reg_all,
# holding n_epochs and lr_all fixed at the Day 15 baseline values.
GRID = [
    {"n_factors": 50, "reg_all": 0.02},   # Day 15 baseline, for comparison
    {"n_factors": 20, "reg_all": 0.02},   # fewer factors -> simpler model
    {"n_factors": 100, "reg_all": 0.02},  # more factors -> more capacity
    {"n_factors": 50, "reg_all": 0.08},   # same factors, stronger regularization
]


def evaluate(model, df, label):
    preds = [model.predict(row.user_id, row.anime_id, r_ui=row.rating) for row in df.itertuples()]
    rmse = accuracy.rmse(preds, verbose=False)
    mae = accuracy.mae(preds, verbose=False)
    return rmse, mae


def main():
    print("Loading processed data...")
    train_df = pd.read_parquet(PROCESSED_DIR / "ratings_train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "ratings_val.parquet")

    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(train_df[["user_id", "anime_id", "rating"]], reader)
    trainset = data.build_full_trainset()

    results = []
    for i, params in enumerate(GRID, 1):
        print(f"\n[{i}/{len(GRID)}] Training with n_factors={params['n_factors']}, reg_all={params['reg_all']}...")
        model = SVD(
            n_factors=params["n_factors"],
            n_epochs=20,
            lr_all=0.005,
            reg_all=params["reg_all"],
            random_state=42,
        )
        model.fit(trainset)

        val_rmse, val_mae = evaluate(model, val_df, "val")
        print(f"    val RMSE={val_rmse:.4f}  MAE={val_mae:.4f}")

        results.append({
            "n_factors": params["n_factors"],
            "reg_all": params["reg_all"],
            "val_rmse": val_rmse,
            "val_mae": val_mae,
        })

    results_df = pd.DataFrame(results).sort_values("val_rmse")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORTS_DIR / "hyperparameter_tuning.csv", index=False)

    print("\n=== Results (sorted by val RMSE, best first) ===")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest combo: n_factors={int(best['n_factors'])}, reg_all={best['reg_all']}"
          f"  (val RMSE={best['val_rmse']:.4f})")
    print(f"Saved full comparison to {REPORTS_DIR / 'hyperparameter_tuning.csv'}")
    print("\nNext step: if the best combo beats the Day 15 baseline, update "
          "train_cf.py's SVD(...) call with these values and retrain the "
          "final model (evaluated on test, once, at the end).")


if __name__ == "__main__":
    main()
