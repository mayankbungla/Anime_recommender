"""
Tests for the typo-tolerant title search, using a small hand-made catalogue.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anime_recommender.features.search import TitleSearch

IDS = [1, 2, 3, 4, 5, 6]
NAMES = [
    "Death Note",
    "Shingeki no Kyojin",
    "Fullmetal Alchemist: Brotherhood",
    "Kimetsu no Yaiba",
    "Steins;Gate",
    "Up",
]
POPULARITY = {1: 0.9, 2: 0.95, 3: 0.8, 4: 0.7, 5: 0.6, 6: 0.1}

search = TitleSearch(IDS, NAMES, POPULARITY)


def titles(query, **kwargs):
    return [h["title"] for h in search.search(query, **kwargs)]


def test_exact_title_ranks_first():
    assert titles("death note")[0] == "Death Note"


def test_misspelling_still_finds_title():
    assert "Death Note" in titles("Deth Note")
    assert "Fullmetal Alchemist: Brotherhood" in titles("fulmetal alchemist")


def test_english_title_maps_to_catalogue_name():
    assert titles("Attack on Titan")[0] == "Shingeki no Kyojin"
    assert titles("demon slayer")[0] == "Kimetsu no Yaiba"


def test_punctuation_and_case_are_ignored():
    assert titles("STEINS GATE")[0] == "Steins;Gate"


def test_partial_title_matches_longer_name():
    assert "Fullmetal Alchemist: Brotherhood" in titles("brotherhood")


def test_nonsense_query_returns_nothing_strong():
    hits = search.search("Zzz Totally Fake Anime Title Nobody Made Up 999")
    assert all(h["match"] < 85 for h in hits)


def test_one_character_query_returns_nothing():
    assert search.search("a") == []
