"""
Typo-tolerant title search over the anime catalogue.

Matches in tiers: exact title, titles starting with the query, titles
containing it, then fuzzy matches for misspellings. Common English
titles are mapped to the romaji names the catalogue uses.
"""

import re

import numpy as np
from rapidfuzz import fuzz, process

ALIASES = {
    "attack on titan": "shingeki no kyojin",
    "demon slayer": "kimetsu no yaiba",
    "my hero academia": "boku no hero academia",
    "your name": "kimi no na wa",
    "spirited away": "sen to chihiro no kamikakushi",
    "the promised neverland": "yakusoku no neverland",
    "promised neverland": "yakusoku no neverland",
    "assassination classroom": "ansatsu kyoushitsu",
    "your lie in april": "shigatsu wa kimi no uso",
    "food wars": "shokugeki no souma",
    "the seven deadly sins": "nanatsu no taizai",
    "seven deadly sins": "nanatsu no taizai",
    "my teen romantic comedy snafu": "yahari ore no seishun love comedy wa machigatteiru",
    "oregairu": "yahari ore no seishun love comedy wa machigatteiru",
    "rascal does not dream": "seishun buta yarou",
    "a silent voice": "koe no katachi",
    "weathering with you": "tenki no ko",
    "howls moving castle": "howl no ugoku shiro",
    "princess mononoke": "mononoke hime",
    "grave of the fireflies": "hotaru no haka",
    "the girl who leapt through time": "toki wo kakeru shoujo",
    "wolf children": "ookami kodomo no ame to yuki",
    "ghost in the shell": "koukaku kikoutai",
    "sailor moon": "bishoujo senshi sailor moon",
    "case closed": "meitantei conan",
    "detective conan": "meitantei conan",
    "fullmetal alchemist": "hagane no renkinjutsushi",
    "haikyu": "haikyuu",
    "the garden of words": "kotonoha no niwa",
    "classroom of the elite": "youkoso jitsuryoku shijou shugi no kyoushitsu e",
    "konosuba": "kono subarashii sekai ni shukufuku wo",
    "rent a girlfriend": "kanojo okarishimasu",
    "dr stone": "dr stone",
    "that time i got reincarnated as a slime": "tensei shitara slime datta ken",
    "re zero": "re zero kara hajimeru isekai seikatsu",
    "tokyo revengers": "tokyo revengers",
    "one piece": "one piece",
}


def normalise(text) -> str:
    """Lowercase, drop punctuation, collapse spaces."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


class TitleSearch:
    def __init__(self, ids, names, popularity: dict, min_fuzzy_len: int = 4):
        self.ids = [int(i) for i in ids]
        self.names = list(names)
        self._norm = [normalise(n) for n in self.names]
        self._pop = np.array([popularity.get(i, 0.0) for i in self.ids])
        self._fuzzy_rows = [i for i, n in enumerate(self._norm) if len(n) >= min_fuzzy_len]
        self._fuzzy_names = [self._norm[i] for i in self._fuzzy_rows]

    @staticmethod
    def _expand(q: str) -> str:
        for key, target in ALIASES.items():
            if q == key or q.startswith(key + " "):
                return target + q[len(key):]
        return q

    def _by_popularity(self, rows: list) -> list:
        return sorted(rows, key=lambda r: -self._pop[r])

    def search(self, query: str, limit: int = 30, fuzzy_cutoff: int = 70) -> list:
        """Best matches first. Each hit has anime_id, title and match (0-100)."""
        q = self._expand(normalise(query))
        if len(q) < 2:
            return []

        exact, starts, inside = [], [], []
        for i, name in enumerate(self._norm):
            if name == q:
                exact.append(i)
            elif name.startswith(q):
                starts.append(i)
            elif q in name:
                inside.append(i)

        hits = [(r, 100) for r in exact + self._by_popularity(starts) + self._by_popularity(inside)]
        seen = {r for r, _ in hits}

        if len(hits) < limit:
            found = process.extract(q, self._fuzzy_names, scorer=fuzz.token_sort_ratio,
                                    limit=limit * 2, score_cutoff=fuzzy_cutoff)
            for _, score, pos in found:
                row = self._fuzzy_rows[pos]
                if row not in seen:
                    hits.append((row, float(score)))
                    seen.add(row)

        return [{"anime_id": self.ids[r], "title": self.names[r], "match": m}
                for r, m in hits[:limit]]
