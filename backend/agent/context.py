"""Shared state for one agent run: the source, its retrieval index, the episode
being built, and a trace emitter that the UI renders live."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

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

    def __post_init__(self) -> None:
        self.index = BM25Index(self.chunks)
        self.full_index = BM25Index(self.source.chunks)
        self.by_id = {c.id: c for c in self.source.chunks}

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
