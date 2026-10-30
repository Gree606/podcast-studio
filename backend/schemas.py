"""Data model for sources, episodes, segments and the agent trace.

Everything is a pydantic model so an episode can be persisted as one JSON file and
re-loaded after a restart, and so the API returns a validated shape to the UI.
"""
from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

Speaker = Literal["host", "expert"]
EpisodeStatus = Literal["queued", "running", "done", "failed", "interrupted"]
SegmentStatus = Literal["pending", "writing", "synthesizing", "checking", "done", "failed"]


def new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:10]


class Chunk(BaseModel):
    """A retrievable passage of the source chapter (~120-200 words)."""

    id: str
    section: str
    text: str
    page: int | None = None


class Section(BaseModel):
    id: str
    title: str
    word_count: int
    chunk_ids: list[str]


class Source(BaseModel):
    id: str
    title: str
    origin: Literal["upload", "paste", "sample", "photo"]
    sections: list[Section]
    chunks: list[Chunk]
    word_count: int
    ocr_pages: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    page_texts: list[str] = Field(default_factory=list)  # photo uploads: what OCR read on each page, for the user to check


class EpisodeSettings(BaseModel):
    length: Literal["short", "medium", "long"] = "medium"
    tone: Literal["casual", "balanced", "serious"] = "balanced"
    # Which compute allocation policy to use; "adaptive" is the default and the
    # others exist for the ablation in eval/.
    policy: Literal["adaptive", "uniform", "single", "no_critic"] = "adaptive"
    host_name: str = "Error606"  # the curious alien
    expert_name: str = "Gree"  # the astronaut who explains
    section_ids: list[str] = Field(default_factory=list)  # empty = whole chapter
    language: str = "en"  # a key of backend.languages.LANGUAGES: the episode's spoken language
    # "fast" = 2 LLM calls (write, then fact-check); "agent" = planner + best-of-N + critic loop;
    # "" = the server's PIPELINE_MODE
    mode: Literal["", "fast", "agent"] = ""


class Turn(BaseModel):
    speaker: Speaker
    text: str
    # Filled after synthesis/stitching:
    tts_text: str | None = None  # normalised text actually sent to TTS
    seg_start: float | None = None  # seconds into the segment's audio
    seg_end: float | None = None
    start: float | None = None  # seconds into the full episode (set when assembled)
    end: float | None = None
    wer: float | None = None  # WER of the kept take
    first_wer: float | None = None  # WER of the first take (before any self-check retry)
    asr_text: str | None = None
    tts_attempts: int = 0


class KeyFact(BaseModel):
    fact: str
    chunk_id: str | None = None
    verified: bool | None = None
    verify_note: str | None = None


class Claim(BaseModel):
    claim: str
    supported: bool
    evidence_chunk_id: str | None = None
    note: str | None = None
    line: int | None = None  # 1-based spoken line it was checked on (quick mode checks every line); None = agent claim check
    corrected: bool = False  # the line was rewritten (or dropped) by the fact-check, so it is no longer said


class Scores(BaseModel):
    """Multi-axis rubric (ToolRL-style fine-grained reward, used for selection only)."""

    faithfulness: float = 0
    coverage: float = 0
    naturalness: float = 0
    engagement: float = 0
    clarity: float = 0
    overall: float = 0
    deterministic_penalty: float = 0


class Candidate(BaseModel):
    index: int
    round: int
    turns: list[Turn]
    scores: Scores | None = None
    feedback: str | None = None
    issues: list[str] = Field(default_factory=list)


class SegmentVersion(BaseModel):
    version: int
    created_at: float = Field(default_factory=time.time)
    instruction: str | None = None
    turns: list[Turn]
    scores: Scores | None = None
    claims: list[Claim] = Field(default_factory=list)
    audio_file: str | None = None
    duration: float | None = None


class Segment(BaseModel):
    id: str
    index: int
    title: str
    goal: str
    kind: Literal["intro", "body", "recap"] = "body"
    key_facts: list[KeyFact] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    difficulty: int = 3  # 1 (easy) .. 5 (hard), estimated by the planner
    budget: int = 1  # writer candidates allotted by the allocator
    status: SegmentStatus = "pending"
    error: str | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    reflections: list[str] = Field(default_factory=list)  # Reflexion episodic memory
    versions: list[SegmentVersion] = Field(default_factory=list)
    active_version: int = 0

    @property
    def current(self) -> SegmentVersion | None:
        for v in self.versions:
            if v.version == self.active_version:
                return v
        return None


class TraceEvent(BaseModel):
    t: float = Field(default_factory=time.time)
    stage: str
    kind: Literal["thought", "action", "observation", "info", "warning", "error", "decision"]
    text: str
    segment_id: str | None = None
    data: dict | None = None


class Metrics(BaseModel):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tts_calls: int = 0
    tts_chars: int = 0
    asr_calls: int = 0
    stage_seconds: dict[str, float] = Field(default_factory=dict)


class Episode(BaseModel):
    id: str
    created_at: float = Field(default_factory=time.time)
    source_id: str
    title: str
    settings: EpisodeSettings
    status: EpisodeStatus = "queued"
    stage: str = "queued"
    progress: float = 0.0  # 0..1, for the progress bar
    message: str = "Waiting to start"
    error: str | None = None
    segments: list[Segment] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
    audio_file: str | None = None
    duration: float | None = None
    learn_line: str | None = None  # one-line "what you'll learn"
    # sha256 of the creator's browser cookie: only they can regenerate, switch versions or restart.
    # None = made before ownership existed (editable, as before). Never sent to the browser.
    owner: str | None = None
