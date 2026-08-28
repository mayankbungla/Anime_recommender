"""
Day 41 - logs every recommendation shown and every one a user engages
with, so Day 42's analytics page has real data to summarise.

SQLite by design (the plan calls for "a small SQLite (or Postgres)
table"). On Heroku specifically the filesystem is ephemeral: this file
resets on every dyno restart or redeploy, so treat it as a rolling log
for demo/portfolio purposes, not a permanent record. Swapping to
Postgres later (a Heroku Postgres add-on) needs no interface change -
callers only ever see log_shown() / log_click(), never raw SQL.
"""

import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "recommendation_log.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN ('shown', 'clicked')),
    anime_id INTEGER NOT NULL,
    title TEXT,
    source_page TEXT,
    session_id TEXT
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def new_session_id() -> str:
    """One id per browser session, so events can be grouped later
    without storing anything that identifies a real person."""
    return uuid.uuid4().hex[:12]


def log_shown(anime_list: list, source_page: str, session_id: str) -> None:
    """One row per anime actually rendered on screen. Batched into a
    single insert since render_cards shows several at once."""
    rows = [
        (time.time(), "shown", a["mal_id"], a.get("title"), source_page, session_id)
        for a in anime_list if a.get("mal_id") is not None
    ]
    if not rows:
        return
    conn = _connect()
    try:
        conn.executemany(
            "INSERT INTO recommendation_events (ts, event_type, anime_id, title, source_page, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def log_click(anime_id: int, title: str, source_page: str, session_id: str) -> None:
    """One row for a single user engagement with a shown anime."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO recommendation_events (ts, event_type, anime_id, title, source_page, session_id) "
            "VALUES (?, 'clicked', ?, ?, ?, ?)",
            (time.time(), anime_id, title, source_page, session_id),
        )
        conn.commit()
    finally:
        conn.close()
