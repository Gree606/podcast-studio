"""Deep agent RAG: BM25 + embeddings fused by rank, vectors cached in SQLite, BM25 fallback on failure."""
import asyncio

import numpy as np

from backend import vectors
from backend.clients import embed
from backend.clients.base import ServiceError
from backend.retrieval import HybridIndex
from backend.schemas import Chunk

CHUNKS = [Chunk(id=f"c{i}", section="S", text=t, page=1) for i, t in enumerate([
    "Iron rusts when it reacts with oxygen and water.",
    "Plants make food by photosynthesis using sunlight.",
    "Magnets attract iron and have a north and a south pole.",
])]


def test_hybrid_search_finds_and_ranks(monkeypatch):
    idx = HybridIndex(CHUNKS)
    hits = asyncio.run(idx.asearch("photosynthesis in plants", k=2))
    assert hits[0][0].id == "c1" and idx.dense_ok


def test_dense_hits_count_even_without_shared_words(monkeypatch):
    # an embedding that knows "chlorophyll food" is about c1 although BM25 has no matching term
    async def fake_embed(texts):
        m = np.zeros((len(texts), 3), dtype=np.float32)
        for i, t in enumerate(texts):
            m[i, 1 if ("photosynthesis" in t or "chlorophyll" in t) else (0 if "rust" in t else 2)] = 1
        return m
    monkeypatch.setattr(vectors, "embed_cached", fake_embed)
    hits = asyncio.run(HybridIndex(CHUNKS).asearch("chlorophyll food", k=1))
    assert hits and hits[0][0].id == "c1"


def test_vectors_are_embedded_once_and_reused(monkeypatch):
    calls = []
    real = embed.embed

    async def counting(texts):
        calls.append(len(texts))
        return await real(texts)
    monkeypatch.setattr(embed, "embed", counting)
    asyncio.run(HybridIndex(CHUNKS).asearch("magnet poles"))
    asyncio.run(HybridIndex(CHUNKS).asearch("magnet poles"))  # a new episode on the same chapter
    assert sum(calls) <= len(CHUNKS) + 1  # passages + the query, only the first time


def test_falls_back_to_keywords_when_embeddings_fail(monkeypatch):
    async def broken(texts):
        raise ServiceError("Embeddings", "is down")
    monkeypatch.setattr(vectors, "embed_cached", broken)
    idx = HybridIndex(CHUNKS)
    hits = asyncio.run(idx.asearch("rusting iron oxygen", k=1))
    assert hits[0][0].id == "c0" and not idx.dense_ok
