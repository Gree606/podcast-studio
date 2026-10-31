"""SQLite vector store for passage and query embeddings.

One table, keyed by (embedding model, SHA-1 of the text): a chapter is embedded once and reused by
every later episode, regeneration and query that needs the same text. A chapter is tens to a few
hundred passages, so exact cosine search over its vectors (numpy, in retrieval.HybridIndex) is
instant and more accurate than an approximate index; SQLite keeps them across restarts with
nothing extra to run.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading

import numpy as np

from .clients import embed as embed_client
from .config import get_settings

_SCHEMA = """CREATE TABLE IF NOT EXISTS vectors (
    model TEXT NOT NULL, key TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL,
    PRIMARY KEY (model, key))"""


class VectorStore:
    def __init__(self, path) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute(_SCHEMA)
        self._db.commit()

    def get(self, model: str, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                rows = self._db.execute(f"SELECT key, vec FROM vectors WHERE model = ? AND key IN ({','.join('?' * len(part))})",
                                        [model, *part]).fetchall()
                out.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return out

    def put(self, model: str, items: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO vectors (model, key, dim, vec) VALUES (?, ?, ?, ?)",
                                 [(model, k, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()])
            self._db.commit()

    def count(self, model: str) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM vectors WHERE model = ?", (model,)).fetchone()[0]


_store: VectorStore | None = None
_store_path: str | None = None


def store() -> VectorStore:
    global _store, _store_path
    path = str(get_settings().data_dir / "vectors.sqlite3")
    if _store is None or _store_path != path:
        _store, _store_path = VectorStore(path), path
    return _store


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


async def embed_cached(texts: list[str]) -> np.ndarray:
    """Embeddings for ``texts``, computing (and storing) only the ones not in the store yet."""
    model = get_settings().embed_model
    keys = [_key(t) for t in texts]
    have = store().get(model, list(dict.fromkeys(keys)))
    missing = [i for i, k in enumerate(keys) if k not in have]
    if missing:
        todo = list(dict.fromkeys(keys[i] for i in missing))
        first = {keys[i]: texts[i] for i in missing}
        fresh = await embed_client.embed([first[k] for k in todo])
        new = dict(zip(todo, fresh))
        store().put(model, new)
        have.update(new)
    return np.stack([have[k] for k in keys]) if texts else np.zeros((0, 1), dtype=np.float32)
