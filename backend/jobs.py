"""Background job runner.

Generation takes minutes, so HTTP requests only *start* work and return an id;
the UI polls status. A global semaphore caps concurrent episodes (several TAs at
once share the LLM/TTS rate limits); extra jobs wait in a visible queue.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from .config import get_settings
from .store import store

log = logging.getLogger("podcast.jobs")

_tasks: set[asyncio.Task] = set()
_sem: asyncio.Semaphore | None = None
_waiting: list[str] = []
_active: dict[str, str] = {}  # episode id -> kind of job running


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
                await fn()
        finally:
            if eid in _waiting:
                _waiting.remove(eid)
            _active.pop(eid, None)

    task = asyncio.create_task(runner(), name=f"{kind}:{eid}")
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def active_jobs() -> dict[str, str]:
    return dict(_active)


def startup_recover() -> None:
    n = store.recover_interrupted()
    if n:
        log.warning("marked %d interrupted episode(s) after restart", n)
