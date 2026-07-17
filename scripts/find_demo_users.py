"""
Find candidate demo user IDs for the "Recommended For You" page (Section 3.1
of HANDOFF_SESSION3.md).

Picks users with a decent number of training ratings (default: 50+) so the
demo shows a model that actually has signal to work with, then prints what
each candidate rated highly so a human can assign a readable label
(e.g. "The Shonen Fan").
"""

import pandas as pd

train = pd.read_parquet("data/processed/ratings_train.parquet")
anime = pd.read_parquet("data/processed/anime_clean.parquet")

# anime_clean.csv has its own "rating" column (community average score),
# which collides with train's "rating" column (this user's score for that
# anime). Rename before merging so nothing gets silently overwritten.
anime = anime.rename(columns={"rating": "community_avg_rating"})

MIN_RATINGS = 50
N_CANDIDATES = 10
SEED = 42

counts = train.groupby("user_id").size()
eligible = counts[counts >= MIN_RATINGS]

if eligible.empty:
    raise SystemExit(
        f"No users with >= {MIN_RATINGS} ratings found. Lower MIN_RATINGS and try again."
    )

candidates = eligible.sample(min(N_CANDIDATES, len(eligible)), random_state=SEED)

print(f"{len(eligible):,} users have >= {MIN_RATINGS} ratings. Showing {len(candidates)} candidates:\n")

for user_id, n_ratings in candidates.items():
    user_rows = train[train["user_id"] == user_id].sort_values("rating", ascending=False)
    top_titles = user_rows.merge(anime, on="anime_id")[["name", "rating"]].head(8)

    print(f"user_id={user_id}  ({n_ratings} ratings in train)")
    for _, row in top_titles.iterrows():
        print(f"    {row['rating']:>2}  {row['name']}")  # this user's own score, not community avg
    print()
