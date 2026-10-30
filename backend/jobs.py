"""Background job runner.

Generation takes minutes, so HTTP requests only *start* work and return an id;
the UI polls status. A global semaphore caps concurrent episodes (several TAs at
once share the LLM/TTS rate limits); extra jobs wait in a visible queue.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Awaitable, Callable

from .config import get_settings
from .store import store

log = logging.getLogger("podcast.jobs")

_tasks: set[asyncio.Task] = set()
_sem: asyncio.Semaphore | None = None
_waiting: list[str] = []
_active: dict[str, str] = {}  # episode id -> kind of job running
_by_episode: dict[str, asyncio.Task] = {}
_started: dict[str, float] = {}  # episode id -> when its current job started running


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(get_settings().max_concurrent_episodes)
    return _sem


def queue_position(eid: str) -> int | None:
    return _waiting.index(eid) + 1 if eid in _waiting else None


def is_busy(eid: str) -> bool:
    return eid in _active or eid in _waiting


def submit(eid: str, kind: str, fn: Callable[[], Awaitable[None]]) -> None:
    async def runner() -> None:
        _waiting.append(eid)
        try:
            async with _semaphore():
                _waiting.remove(eid)
                _active[eid] = kind
                _started[eid] = time.monotonic()
                await fn()
        finally:
            if eid in _waiting:
                _waiting.remove(eid)
            _active.pop(eid, None)
            _started.pop(eid, None)

    task = asyncio.create_task(runner(), name=f"{kind}:{eid}")
    _tasks.add(task)
    _by_episode[eid] = task
    task.add_done_callback(_tasks.discard)
    task.add_done_callback(lambda t: _by_episode.pop(eid, None) if _by_episode.get(eid) is t else None)


async def cancel(eid: str) -> None:
    """Stop an episode's running job (its finished work is already saved, so it can be resumed)."""
    task = _by_episode.get(eid)
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(asyncio.shield(task), timeout=5)


def running_for(eid: str) -> float | None:
    """Seconds the episode's current job has been running (None when idle or still queued)."""
    return round(time.monotonic() - _started[eid], 1) if eid in _started else None


def active_jobs() -> dict[str, str]:
    return dict(_active)


def startup_recover() -> None:
    n = store.recover_interrupted()
    if n:
        log.warning("marked %d interrupted episode(s) after restart", n)
