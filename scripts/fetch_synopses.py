"""
Day 18 (data step): fetch a synopsis for every anime in the cleaned
catalogue, so the content model has real text to embed.

The Kaggle anime.csv has no synopsis column, so we pull them from Jikan.
The Kaggle anime_id is the MyAnimeList id, so we can query Jikan by that
id directly with no name matching.

This does NOT reuse src/.../jikan_client.py on purpose: that client is
built around Streamlit's cache for the live UI. A bulk offline job wants
different behaviour (plain requests, its own rate limiting, resume on
restart), so it lives here as a self-contained script.

Two things make a ~12k-call job survivable:
  - rate limiting: Jikan allows roughly 3 requests/second. We stay under
    that, so the whole run takes a bit over an hour.
  - resume: results are flushed to Parquet every SAVE_EVERY rows, and on
    restart we skip ids we already have. Kill it and rerun any time.

USAGE
    python scripts/fetch_synopses.py
"""

import time
from pathlib import Path

import pandas as pd
import requests

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
CATALOG_PATH = PROCESSED_DIR / "anime_clean.parquet"
OUT_PATH = PROCESSED_DIR / "anime_synopses.parquet"

JIKAN = "https://api.jikan.moe/v4"
REQUEST_DELAY = 0.4      # seconds between calls, comfortably under 3/s
SAVE_EVERY = 100         # flush progress this often


def fetch_one(anime_id: int) -> dict:
    """Return {anime_id, synopsis} for one id. Empty synopsis on failure
    so a single bad/missing entry never stops the whole run."""
    try:
        r = requests.get(f"{JIKAN}/anime/{anime_id}", timeout=10)
        if r.status_code == 429:  # rate limited: wait and let the caller retry
            time.sleep(2)
            r = requests.get(f"{JIKAN}/anime/{anime_id}", timeout=10)
        r.raise_for_status()
        data = r.json().get("data", {})
        return {"anime_id": anime_id, "synopsis": data.get("synopsis") or ""}
    except Exception:
        return {"anime_id": anime_id, "synopsis": ""}


def main():
    catalog = pd.read_parquet(CATALOG_PATH)
    all_ids = catalog["anime_id"].tolist()

    done = pd.DataFrame(columns=["anime_id", "synopsis"])
    if OUT_PATH.exists():
        done = pd.read_parquet(OUT_PATH)
        print(f"Resuming: {len(done):,} synopses already fetched.")

    have = set(done["anime_id"])
    todo = [a for a in all_ids if a not in have]
    print(f"{len(todo):,} to fetch out of {len(all_ids):,} total.")

    rows = done.to_dict("records")
    for i, anime_id in enumerate(todo, 1):
        rows.append(fetch_one(anime_id))
        time.sleep(REQUEST_DELAY)

        if i % SAVE_EVERY == 0 or i == len(todo):
            pd.DataFrame(rows).to_parquet(OUT_PATH, index=False)
            print(f"  {i:,}/{len(todo):,} fetched, saved to {OUT_PATH.name}")

    final = pd.DataFrame(rows)
    missing = (final["synopsis"].str.len() == 0).sum()
    print(f"\nDone. {len(final):,} rows, {missing:,} without a synopsis "
          f"(these fall back to title + genres at embedding time).")


if __name__ == "__main__":
    main()
