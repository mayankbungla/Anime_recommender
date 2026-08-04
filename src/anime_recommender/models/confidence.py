"""
Labels a recommendation's confidence based on how much real evidence
backed it, not the score itself. A high hybrid score built on an anime
with 3 ratings total is a guess, not a strong pick, this makes that
distinction visible instead of hiding it behind one polished number.
"""


def build_rating_counts(train) -> dict:
    """How many ratings each anime received in train, the main evidence
    signal for both the CF and popularity terms."""
    return train.groupby("anime_id").size().to_dict()


def compute_confidence(anime_id: int, rating_counts: dict, liked_ids: list,
                        content, high: int = 50, medium: int = 10) -> dict:
    """
    Returns a label plus the raw counts behind it, so the number is
    checkable rather than a black box. Thresholds are a judgment call,
    50+ ratings counts as solid CF evidence, 10-49 as workable but thin,
    under 10 as too little to trust much.
    """
    n_ratings = rating_counts.get(anime_id, 0)
    taste_support = sum(1 for a in liked_ids if a in content._row_by_id)

    if n_ratings >= high:
        label = "High confidence"
    elif n_ratings >= medium:
        label = "Medium confidence"
    else:
        label = "Low confidence, based on limited data"

    return {
        "label": label,
        "anime_rating_count": n_ratings,
        "taste_vector_support": taste_support,
    }
