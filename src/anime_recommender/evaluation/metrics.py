"""
Ranking quality metrics for evaluating a recommender's top-k output.

Everything here is written from scratch on plain lists, no sklearn or
surprise metric helpers, so the math stays visible and explainable
rather than hidden behind a library call.
"""

import math


def precision_at_k(recommended, relevant, k):
    """Fraction of the top-k recommended items that are actually relevant."""
    # divides by k even if fewer than k items were recommended, on purpose,
    # a short recommendation list should score worse, not get a free pass
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k):
    """Fraction of all relevant items that showed up in the top-k."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def dcg_at_k(gains, k):
    """Discounted cumulative gain: relevant items ranked higher count more."""
    top_k = gains[:k]
    return sum(gain / math.log2(i + 2) for i, gain in enumerate(top_k))


def ndcg_at_k(recommended, relevance, k):
    """
    Normalized DCG at k. `relevance` maps item -> a relevance score
    (use 1/0 for binary relevance, or the actual rating for graded).
    Items not in `relevance` are treated as 0.
    """
    gains = [relevance.get(item, 0) for item in recommended[:k]]
    dcg = dcg_at_k(gains, k)

    # ideal ranking sorts every known relevant score highest first
    ideal_gains = sorted(relevance.values(), reverse=True)
    idcg = dcg_at_k(ideal_gains, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg
