"""
Tests for the committed model artifact: models/svd_cf_model.pkl

STATUS: scaffolded from PROJECT_BRIEF.md. These tests load the real
committed model file (small, portfolio-sized per the brief) rather than
training a fresh one, since the point is to guard against a broken/corrupt
commit, not to re-validate training.

If the model isn't present in a given environment (e.g. Git LFS not pulled),
these are skipped rather than failed, so the rest of the suite still runs.
"""

import os

import joblib
import pytest

MODEL_PATH = "models/svd_cf_model.pkl"

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason=f"{MODEL_PATH} not present in this environment",
)


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


def test_model_loads(model):
    assert model is not None
    assert hasattr(model, "predict")


def test_model_has_trainset(model):
    # needed by predict.py's get_user_top_n / similar_items
    assert hasattr(model, "trainset")


def test_predict_returns_plausible_estimate(model):
    trainset = model.trainset
    # grab a real user/item pair the model actually saw during training
    inner_uid = next(iter(trainset.all_users()))
    raw_uid = trainset.to_raw_uid(inner_uid)
    inner_iid = next(iter(trainset.all_items()))
    raw_iid = trainset.to_raw_iid(inner_iid)

    pred = model.predict(raw_uid, raw_iid)
    assert 1.0 <= pred.est <= 10.0


def test_item_factors_shape_matches_trainset(model):
    n_items = model.trainset.n_items
    assert model.qi.shape[0] == n_items
