"""Text embeddings for the deep agent's hybrid retrieval.

OpenRouter's ``POST /v1/embeddings`` (OpenAI wire format: ``{model, input: [...]}`` ->
``{data: [{index, embedding}]}``), with the OpenRouter key by default. Default model
``google/gemini-embedding-001``: multilingual, so Hindi and Malayalam chapters embed as well as
English ones. Vectors are L2-normalised so a dot product is the cosine similarity.
In MOCK_MODE a deterministic bag-of-hashed-terms vector stands in, so tests need no network.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..config import get_settings
from ..retrieval import terms
from .base import ServiceError, request_with_retry
from .usage import record

BATCH = 64
MOCK_DIM = 256


def embed_ready() -> bool:
    s = get_settings()
    return bool(s.embed_model) and (s.mock_mode or s.embed_ready())


def _normalise(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(n == 0, 1, n)).astype(np.float32)


def _mock(texts: list[str]) -> np.ndarray:
    m = np.zeros((len(texts), MOCK_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for term in terms(t):
            m[i, int(hashlib.md5(term.encode()).hexdigest()[:6], 16) % MOCK_DIM] += 1.0
    return _normalise(m)


async def embed(texts: list[str]) -> np.ndarray:
    """Embeddings for ``texts`` as an (n, d) float32 array of unit vectors."""
    s = get_settings()
    if not texts:
        return np.zeros((0, MOCK_DIM), dtype=np.float32)
    if s.mock_mode:
        return _mock(texts)
    rows: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        batch = [t[:6000] for t in texts[start:start + BATCH]]
        resp = await request_with_retry(
            "Embeddings", "POST", s.embed_base_url.rstrip("/") + "/v1/embeddings",
            json={"model": s.embed_model, "input": batch},
            headers={"Authorization": f"Bearer {s.embed_key()}", "Content-Type": "application/json"},
            timeout=30, attempts=3,
        )
        try:
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            rows += [d["embedding"] for d in data]
        except (ValueError, KeyError, TypeError) as e:
            raise ServiceError("Embeddings", "returned an unexpected response", resp.text[:200]) from e
        record(embed_calls=1)
    if len(rows) != len(texts):
        raise ServiceError("Embeddings", f"returned {len(rows)} vectors for {len(texts)} texts")
    return _normalise(np.asarray(rows, dtype=np.float32))
