"""BM25 retrieval over chapter chunks.

A chapter is only a few hundred passages, so a dependency-free in-memory BM25 is
fast enough and fully deterministic, which keeps the planner's tool observations
reproducible (useful for the eval/ablation runs).
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .schemas import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the of and or to in on at for from by with is are was were be been being this that these those "
    "it its as into than then so such can could may might will would shall should do does did not no "
    "we you they he she i our your their his her them us what which who whom how why when where there here "
    "also more most very just only about over under between each other some any all".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _stem(t: str) -> str:
    # Very light stemming so "reactions"/"reaction" and "oxidised"/"oxidise" match.
    for suf in ("ations", "ation", "ings", "ing", "ies", "es", "ed", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def terms(text: str) -> list[str]:
    return [_stem(t) for t in tokenize(text)]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(terms(c.section + " " + c.text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = terms(query)
        scored: list[tuple[float, int]] = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                f = d.get(t)
                if not f:
                    continue
                s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [(self.chunks[i], s) for s, i in scored[:k]]


def overlap(a: str, b: str) -> float:
    """Fraction of content terms of ``a`` that appear in ``b`` (rule-based support check)."""
    ta = set(terms(a))
    if not ta:
        return 0.0
    tb = set(terms(b))
    return len(ta & tb) / len(ta)
