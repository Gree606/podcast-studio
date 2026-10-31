"""BM25 retrieval over chapter chunks.

A chapter is only a few hundred passages, so a dependency-free in-memory BM25 is
fast enough and fully deterministic, which keeps the planner's tool observations
reproducible (useful for the eval/ablation runs).
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

from .schemas import Chunk

log = logging.getLogger("podcast.retrieval")

# Latin words, or runs of Indic script (Devanagari to Malayalam, with their vowel signs and
# zero-width joiners, minus the danda). Python's \w would cut "चुम्बकों" apart at the vowel signs.
_TOKEN = re.compile(r"[a-z0-9]+|[\u0900-\u0963\u0966-\u0D7F\u200c\u200d]+")
_INDIC_PREFIX = 6  # code points; see terms()
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
    """Search terms. Indic words also count by their first few letters, a stand-in for stemming:
    Malayalam and Tamil glue endings on ("ജലാവസ്ഥ" / "ജലാവസ്ഥകളിലൂടെയുള്ള"), and so, less, does Hindi."""
    out = []
    for t in tokenize(text):
        if t.isascii():
            out.append(_stem(t))
        else:
            out.append(t)
            if len(t) > _INDIC_PREFIX:
                out.append(t[:_INDIC_PREFIX])
    return out


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


    async def asearch(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Same as ``search``; lets callers treat keyword-only and hybrid indexes alike."""
        return self.search(query, k)


RRF_K = 60  # reciprocal-rank-fusion constant (Cormack et al. 2009)
POOL = 20   # candidates taken from each retriever before fusing


class HybridIndex:
    """Keyword (BM25) + meaning (embedding) search, fused by reciprocal rank.

    BM25 finds exact terms ("photosynthesis", chunk ids, names); embeddings find passages that say
    the same thing in other words, or in another language/script. Passage vectors come from the
    SQLite vector store (backend/vectors.py), so a chapter is embedded once. If the embedding
    service fails, search quietly falls back to BM25 alone (``dense_ok`` turns False)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Index(chunks)
        self._mat = None
        self.dense_ok = True

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        return self.bm25.search(query, k)  # synchronous callers (rule checks) stay lexical on purpose

    @staticmethod
    def _passage_text(c: Chunk) -> str:
        return f"{c.section}\n{c.text}"

    async def _matrix(self):
        if self._mat is None:
            from .vectors import embed_cached  # local import: vectors -> clients.embed -> retrieval
            self._mat = await embed_cached([self._passage_text(c) for c in self.chunks])
        return self._mat

    async def asearch(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        lexical = self.bm25.search(query, k=POOL)
        if not self.dense_ok or not self.chunks:
            return lexical[:k]
        try:
            from .vectors import embed_cached
            mat = await self._matrix()
            qv = (await embed_cached([query]))[0]
        except Exception as e:  # noqa: BLE001 - any embedding failure: keyword search still works
            log.warning("embedding search unavailable, using BM25 only: %s", e)
            self.dense_ok = False
            return lexical[:k]
        dense = (mat @ qv).argsort()[::-1][:POOL]
        pos = {id(c): i for i, c in enumerate(self.chunks)}
        fused: dict[int, float] = {}
        for rank, (c, _) in enumerate(lexical):
            fused[pos[id(c)]] = fused.get(pos[id(c)], 0.0) + 1 / (RRF_K + rank + 1)
        for rank, i in enumerate(dense):
            fused[int(i)] = fused.get(int(i), 0.0) + 1 / (RRF_K + rank + 1)
        best = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(self.chunks[i], s) for i, s in best]


def overlap(a: str, b: str) -> float:
    """Fraction of content terms of ``a`` that appear in ``b`` (rule-based support check)."""
    ta = set(terms(a))
    if not ta:
        return 0.0
    tb = set(terms(b))
    return len(ta & tb) / len(ta)
