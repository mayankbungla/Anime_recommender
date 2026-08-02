"""
Turns a hybrid score's components into one plain-language sentence
explaining why an anime was recommended, rather than showing a bare
number nobody can act on.
"""

from anime_recommender.models.hybrid import cf_score, content_score


def explain_recommendation(user_id: int, anime_id: int, algo, content,
                            taste_vector, popularity: dict, liked_ids: list,
                            catalog, alpha: float = 0.34, beta: float = 0.33,
                            gamma: float = 0.33) -> str:
    """Picks whichever signal (CF, content, or popularity) contributed
    the most to this anime's score and explains it in those terms."""
    cf = cf_score(user_id, anime_id, algo)
    cf_component = alpha * cf if cf is not None else 0.0
    content_component = beta * content_score(anime_id, taste_vector, content)
    pop_component = gamma * popularity.get(anime_id, 0.0)

    dominant = max(
        [("cf", cf_component), ("content", content_component), ("popularity", pop_component)],
        key=lambda x: x[1],
    )[0]

    if dominant == "cf":
        return "Recommended based on patterns from users with similar taste to yours."

    if dominant == "content":
        best_match = _most_similar_liked(anime_id, liked_ids, content, catalog)
        if best_match:
            return f"Recommended because it's similar to \"{best_match}\", which you rated highly."
        return "Recommended because it shares themes and genres with shows you've liked."

    return "Recommended because it's one of the most-watched anime in our dataset."


def _most_similar_liked(anime_id: int, liked_ids: list, content, catalog) -> str | None:
    """Which liked anime this recommendation resembles most, for a
    specific explanation instead of a generic one."""
    if anime_id not in content._row_by_id:
        return None
    target = content.embeddings[content._row_by_id[anime_id]]

    best_id, best_sim = None, -1.0
    for liked in liked_ids:
        if liked not in content._row_by_id:
            continue
        vec = content.embeddings[content._row_by_id[liked]]
        sim = float(target @ vec)
        if sim > best_sim:
            best_id, best_sim = liked, sim

    if best_id is None:
        return None
    match = catalog[catalog["anime_id"] == best_id]
    return match.iloc[0]["name"] if len(match) else None
