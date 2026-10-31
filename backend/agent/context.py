"""Shared state for one agent run: the source, its retrieval index, the episode
being built, and a trace emitter that the UI renders live."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable

from ..config import get_settings
from ..retrieval import BM25Index
from ..schemas import Chunk, Episode, Source, TraceEvent

log = logging.getLogger("podcast.agent")

LENGTHS = {
    # segments, words per segment  (~150 spoken words per minute)
    "short": (3, 230),
    "medium": (4, 280),
    "long": (6, 300),
}


@dataclass
class AgentContext:
    episode: Episode
    source: Source
    chunks: list[Chunk]  # the chunks in scope (selected sections)
    on_change: Callable[[], None] = lambda: None
    index: BM25Index = field(init=False)
    full_index: BM25Index = field(init=False)
    # (speaker, spoken text) -> task recording that line; lets recording start before the fact-check ends
    tts_cache: dict = field(default_factory=dict)
    # listen-back self-check health for this episode (agent/voice.py): first takes checked / badly misheard
    asr_checked: int = 0
    asr_bad: int = 0
    asr_off: bool = False
    # Deep agent LLM budget (see can_spend): calls already made when this job started, parts fact-checked
    llm_start: int = field(init=False, default=0)
    claims_checked: set = field(default_factory=set)

    def __post_init__(self) -> None:
        self.llm_start = self.episode.metrics.llm_calls
        self.index = BM25Index(self.chunks)
        self.full_index = BM25Index(self.source.chunks)
        self.by_id = {c.id: c for c in self.source.chunks}

    def can_spend(self, n: int) -> bool:
        """True if ``n`` more optional LLM calls fit in this job's budget while still leaving 2 calls
        (claim extract + judge) for every part that is being made and hasn't been fact-checked yet."""
        cap = get_settings().deep_max_llm_calls
        if cap <= 0:
            return True
        pending = sum(1 for seg in self.episode.segments
                      if seg.status in ("pending", "writing") and seg.id not in self.claims_checked)
        return self.episode.metrics.llm_calls - self.llm_start + n + 2 * pending <= cap

    @property
    def n_segments(self) -> int:
        return LENGTHS[self.episode.settings.length][0]

    @property
    def words_per_segment(self) -> int:
        return LENGTHS[self.episode.settings.length][1]

    def emit(self, stage: str, kind: str, text: str, segment_id: str | None = None, data: dict | None = None) -> None:
        ev = TraceEvent(stage=stage, kind=kind, text=text, segment_id=segment_id, data=data)
        self.episode.trace.append(ev)
        log.info("[%s] %s/%s %s: %s", self.episode.id, stage, kind, segment_id or "-", text[:200])
        self.on_change()

    def status(self, stage: str, message: str, progress: float | None = None) -> None:
        self.episode.stage = stage
        self.episode.message = message
        if progress is not None:
            self.episode.progress = max(self.episode.progress, min(1.0, progress))
        self.on_change()

    @contextlib.asynccontextmanager
    async def creeping(self, start: float, end: float, expected_s: float):
        """Move the progress bar from ``start`` towards ``end`` while one long call runs, so the
        screen never looks frozen: it covers ~63% of the way in ``expected_s``, then slows down."""
        t0 = time.monotonic()

        async def tick():
            while True:
                await asyncio.sleep(1.5)
                p = start + (end - start) * (1 - math.exp(-(time.monotonic() - t0) / expected_s))
                self.episode.progress = max(self.episode.progress, min(end, p))
                self.on_change()

        task = asyncio.create_task(tick())
        try:
            yield
        finally:
            task.cancel()

    def passages_text(self, chunk_ids: list[str], limit_words: int = 1400) -> str:
        out, words = [], 0
        for cid in chunk_ids:
            c = self.by_id.get(cid)
            if not c:
                continue
            w = len(c.text.split())
            if out and words + w > limit_words:
                break
            out.append(f"[{c.id}] ({c.section}) {c.text}")
            words += w
        return "\n\n".join(out)
