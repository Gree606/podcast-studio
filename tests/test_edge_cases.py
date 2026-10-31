"""Acceptance tests for edge cases.

These tests use the existing mock-mode fixture from tests/conftest.py and make
no network calls.
"""
from __future__ import annotations

import re

from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.main import app
from backend.quiz import build_quiz
from backend.schemas import (
    Chunk,
    Episode,
    EpisodeSettings,
    KeyFact,
    Section,
    Segment,
    Source,
)


def test_config_never_exposes_runtime_key_values(monkeypatch):
    """Acceptance criterion 3: a loaded sentinel key is absent from API responses."""
    sentinel = "test-secret-value-must-not-leak-12345"
    monkeypatch.setenv("LLM_API_KEY", sentinel)
    monkeypatch.setenv("TTS_API_KEY", sentinel)
    monkeypatch.setenv("ASR_API_KEY", sentinel)
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            combined = "\n".join(
                response.text
                for response in (
                    client.get("/api/health"),
                    client.get("/api/config"),
                    client.get("/api/episodes/does-not-exist"),
                )
            )
        assert sentinel not in combined
    finally:
        get_settings.cache_clear()


def test_quiz_rejects_case_variant_distractors_from_cited_passage():
    """Acceptance criterion 10: distractors cannot occur in the cited passage."""
    chunks = [
        Chunk(
            id="c1",
            section="Plants",
            text="Plants make food by photosynthesis using sunlight, water and carbon dioxide.",
        ),
        Chunk(
            id="c2",
            section="Plants",
            text="Roots absorb water from the soil and carry it to the plant.",
        ),
        Chunk(
            id="c3",
            section="Plants",
            text="Leaves contain chlorophyll, which traps energy from sunlight.",
        ),
        Chunk(
            id="c4",
            section="Plants",
            text="Stomata allow gases to move in and out of a leaf.",
        ),
    ]
    source = Source(
        id="src",
        title="Plants",
        origin="paste",
        word_count=80,
        chunks=chunks,
        sections=[Section(id="s", title="Plants", word_count=80, chunk_ids=[c.id for c in chunks])],
    )
    facts = [
        "Plants make food by photosynthesis using sunlight, water and carbon dioxide.",
        "Roots absorb water from the soil and carry it to the plant.",
        "Leaves contain chlorophyll, which traps energy from sunlight.",
        "Stomata allow gases to move in and out of a leaf.",
    ]
    episode = Episode(
        id="ep",
        source_id=source.id,
        title=source.title,
        settings=EpisodeSettings(),
        segments=[
            Segment(
                id=f"seg{i}",
                index=i,
                title=f"Part {i}",
                goal="test",
                key_facts=[KeyFact(fact=fact, chunk_id=f"c{i + 1}", verified=True)],
            )
            for i, fact in enumerate(facts)
        ],
    )

    for question in build_quiz(episode, source):
        passage = next(c.text.lower() for c in chunks if c.id == question["chunk_id"])
        for option in question["options"]:
            if option.lower() != question["answer"].lower():
                assert option.lower() not in passage
        assert question["question"].count("_____") == 1
        assert not re.search(rf"\b{re.escape(question['answer'])}\b", question["question"], re.IGNORECASE)


def test_kid_preset_values_remain_valid_episode_settings():
    """Acceptance criterion 6: every kid-facing preset maps to the existing schema."""
    with TestClient(app) as client:
        presets = client.get("/api/config").json()["presets"]

    assert [item["label"] for item in presets["length"]] == [
        "Quick Bite",
        "Story Time",
        "Deep Dive",
    ]
    assert [item["label"] for item in presets["tone"]] == [
        "Fun",
        "Balanced",
        "Exam Hero",
    ]
    for item in presets["length"]:
        assert EpisodeSettings(length=item["value"]).length == item["value"]
    for item in presets["tone"]:
        assert EpisodeSettings(tone=item["value"]).tone == item["value"]
