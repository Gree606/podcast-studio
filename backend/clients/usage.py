"""Per-episode cost/latency accounting via a context variable.

The orchestrator sets ``current_metrics`` at the start of a job; every client call
made inside that asyncio task (and its children) adds to the same Metrics object
without threading it through every function signature.
"""
from __future__ import annotations

from contextvars import ContextVar

from ..schemas import Metrics

current_metrics: ContextVar[Metrics | None] = ContextVar("current_metrics", default=None)


def record(**deltas: int) -> None:
    m = current_metrics.get()
    if m is None:
        return
    for k, v in deltas.items():
        setattr(m, k, getattr(m, k) + v)
