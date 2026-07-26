"""
Tests for precision@k, recall@k, and ndcg@k, checked against values
worked out by hand, not just re-running the same code twice.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_recommender.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

RECOMMENDED = ["A", "B", "C", "D", "E"]
RELEVANT = {"B", "D", "F"}


def test_precision_at_k_matches_hand_calc():
    assert abs(precision_at_k(RECOMMENDED, RELEVANT, 3) - 1 / 3) < 1e-9


def test_recall_at_k_matches_hand_calc():
    assert abs(recall_at_k(RECOMMENDED, RELEVANT, 3) - 1 / 3) < 1e-9


def test_recall_with_no_relevant_items_is_zero():
    assert recall_at_k(RECOMMENDED, set(), 3) == 0.0


def test_ndcg_matches_hand_calc():
    relevance = {"B": 1, "D": 1}
    assert abs(ndcg_at_k(RECOMMENDED, relevance, 3) - 0.3868) < 1e-3


def test_ndcg_perfect_ranking_is_one():
    perfect = ndcg_at_k(["X", "Y", "Z"], {"X": 1, "Y": 1, "Z": 0}, 3)
    assert abs(perfect - 1.0) < 1e-9


def test_ndcg_with_no_relevance_data_is_zero():
    assert ndcg_at_k(["A", "B"], {}, 2) == 0.0
