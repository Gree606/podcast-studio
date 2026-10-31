"""Disk persistence for sources and episodes.

One JSON file per object plus audio files in the episode's folder. Writes are
atomic (tmp file + rename) so a crash mid-write never corrupts an episode, and
episodes survive restarts and redeploys (on hosts with a persistent disk).
Designed for a single worker process: per-episode asyncio locks serialise
generation and regeneration on the same episode.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path

from .config import get_settings
from .schemas import Episode, Source


def atomic_write(path: Path, data: str | bytes) -> None:
    """Write via a temp file + rename, so readers never see a half-written file."""
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    if isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


class Store:
    def __init__(self) -> None:
        self._episodes: dict[str, Episode] = {}
        self._sources: dict[str, Source] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_save: dict[str, float] = {}
        self._deleted: set[str] = set()  # a job still winding down must not write a deleted episode back

    # --- paths -------------------------------------------------------------
    def episode_dir(self, eid: str) -> Path:
        d = get_settings().episodes_dir / eid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def delete_episode(self, eid: str) -> bool:
        """Remove an episode and its audio for good (the caller stops its job first)."""
        if not eid.replace("_", "").isalnum():
            return False
        self._deleted.add(eid)
        self._episodes.pop(eid, None)
        self._last_save.pop(eid, None)
        d = get_settings().episodes_dir / eid
        if not d.is_dir():
            return False
        shutil.rmtree(d)
        return True

    def lock(self, eid: str) -> asyncio.Lock:
        if eid not in self._locks:
            self._locks[eid] = asyncio.Lock()
        return self._locks[eid]

    # --- sources -----------------------------------------------------------
    def save_source(self, src: Source) -> None:
        self._sources[src.id] = src
        atomic_write(get_settings().sources_dir / f"{src.id}.json", src.model_dump_json())

    def get_source(self, sid: str) -> Source | None:
        if sid in self._sources:
            return self._sources[sid]
        if "/" in sid or ".." in sid:
            return None
        p = get_settings().sources_dir / f"{sid}.json"
        if not p.exists():
            return None
        src = Source.model_validate_json(p.read_text(encoding="utf-8"))
        self._sources[sid] = src
        return src

    # --- episodes ----------------------------------------------------------
    def save_episode(self, ep: Episode, force: bool = True) -> None:
        """Persist. Non-forced saves (live trace updates) are throttled to ~2/s."""
        if ep.id in self._deleted:
            return
        self._episodes[ep.id] = ep
        now = time.time()
        if not force and now - self._last_save.get(ep.id, 0) < 0.5:
            return
        self._last_save[ep.id] = now
        atomic_write(self.episode_dir(ep.id) / "episode.json", ep.model_dump_json())

    def get_episode(self, eid: str) -> Episode | None:
        if eid in self._episodes:
            return self._episodes[eid]
        if not eid.replace("_", "").isalnum():
            return None
        p = get_settings().episodes_dir / eid / "episode.json"
        if not p.exists():
            return None
        ep = Episode.model_validate_json(p.read_text(encoding="utf-8"))
        self._episodes[eid] = ep
        return ep

    def recover_interrupted(self) -> int:
        """On startup, mark jobs that were running when the process died."""
        n = 0
        for d in get_settings().episodes_dir.iterdir():
            p = d / "episode.json"
            if not p.exists():
                continue
            try:
                ep = Episode.model_validate_json(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if ep.status in ("queued", "running"):
                ep.status = "interrupted"
                ep.message = "The server restarted while this was being made. Press Resume to continue where it stopped."
                for seg in ep.segments:
                    if seg.status not in ("done", "failed"):
                        seg.status = "pending"
                self.save_episode(ep)
                n += 1
        return n


store = Store()
