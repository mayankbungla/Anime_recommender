"""
Day 20: eyeball both models on anime I actually know.

Picks five well-known titles and prints, side by side:
  - CF neighbours from the learned SVD item factors
  - content neighbours from the synopsis embeddings

Writes the same thing to reports/sanity_check.md so the notes live with
the repo. The CF side always runs (it only needs the exported factors).
The content side runs only if models/content/ exists, so this is useful
before and after building the content model.

USAGE
    python scripts/sanity_check.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anime_recommender.models.predict import similar_items_from_factors  # noqa: E402

CATALOG_PATH = ROOT / "data" / "processed" / "anime_clean.parquet"
CONTENT_DIR = ROOT / "models" / "content"
OUT_PATH = ROOT / "reports" / "sanity_check.md"

# titles I know well, matched by substring against the cleaned catalogue
PICKS = ["Death Note", "Steins;Gate", "Cowboy Bebop", "One Punch Man", "Shingeki no Kyojin"]
K = 8


def resolve_id(catalog: pd.DataFrame, query: str) -> int | None:
    hits = catalog[catalog["name"].str.contains(query, case=False, na=False, regex=False)]
    return int(hits.iloc[0]["anime_id"]) if len(hits) else None


def format_recs(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "  (no results)"
    return "\n".join(f"  {r.similarity:5.3f}  {r.title}" for r in df.itertuples())


def main():
    catalog = pd.read_parquet(CATALOG_PATH)[["anime_id", "name", "genre"]]

    content = None
    if CONTENT_DIR.exists():
        from anime_recommender.features.content_model import ContentRecommender
        content = ContentRecommender.load(CONTENT_DIR)
    else:
        print("models/content/ not found: content side will be skipped.\n"
              "Run scripts/build_content_model.py to fill it in.\n")

    lines = ["# Day 20: sanity check", "",
             "Neighbours for five well-known anime, from each model.", ""]

    for title in PICKS:
        anime_id = resolve_id(catalog, title)
        header = f"## {title} (id={anime_id})"
        print("\n" + header)
        lines += [header, ""]

        if anime_id is None:
            print("  not found in catalogue")
            lines += ["_Not found in catalogue._", ""]
            continue

        try:
            cf = similar_items_from_factors(anime_id, k=K, anime_df=catalog)
        except ValueError as e:
            cf = None
            print(f"  CF: {e}")

        print("  [CF - learned SVD factors]")
        print(format_recs(cf))
        lines += ["**CF (learned SVD factors)**", "", "```", format_recs(cf), "```", ""]

        if content is not None:
            try:
                cb = content.recommend(anime_id, k=K)
            except ValueError as e:
                cb = None
                print(f"  content: {e}")
            print("  [Content - synopsis embeddings]")
            print(format_recs(cb))
            lines += ["**Content (synopsis embeddings)**", "", "```", format_recs(cb), "```", ""]

    lines += ["## What I notice", "",
              "_Write your observations here after reading the lists above:_",
              "- Do the CF neighbours share a fanbase rather than just a genre?",
              "- Do the content neighbours share plot/tone rather than just tags?",
              "- Where do the two disagree, and which looks more sensible?", ""]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
