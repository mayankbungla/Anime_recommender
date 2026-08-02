"""
Re-ranks a scored candidate list with Maximal Marginal Relevance, so the
top-k isn't just the highest-scoring items even when several of them are
near-duplicates of each other (same franchise, same tone, same synopsis).
Trades a small amount of raw score for genuinely different picks.
"""

import numpy as np


def mmr_rerank(scored_candidates: list, embeddings, row_by_id: dict,
               k: int, diversity_weight: float = 0.3) -> list:
    """
    scored_candidates: list of (anime_id, score) tuples, already sorted
    or not, order doesn't matter going in.
    diversity_weight: 0 means ignore diversity entirely (plain top-k),
    1 means ignore the original score entirely and just spread out picks.
    """
    if not scored_candidates:
        return []

    pool = {aid: score for aid, score in scored_candidates}
    max_score = max(pool.values()) or 1.0
    normalized = {aid: score / max_score for aid, score in pool.items()}

    selected = []
    remaining = set(pool.keys())

    while remaining and len(selected) < k:
        best_aid, best_value = None, -float("inf")

        for aid in remaining:
            relevance = normalized[aid]
            if aid in row_by_id and selected:
                sims = [
                    _cosine(embeddings[row_by_id[aid]], embeddings[row_by_id[s]])
                    for s in selected if s in row_by_id
                ]
                redundancy = max(sims) if sims else 0.0
            else:
                redundancy = 0.0

            mmr_value = (1 - diversity_weight) * relevance - diversity_weight * redundancy
            if mmr_value > best_value:
                best_aid, best_value = aid, mmr_value

        selected.append(best_aid)
        remaining.remove(best_aid)

    return selected


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
