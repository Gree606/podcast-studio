"""Ask Gree: doubts are answered from retrieved chapter passages, with citations and Gree's voice."""
from fastapi.testclient import TestClient

from backend import ask
from backend.main import app
from backend.schemas import (
    Chunk,
    Episode,
    EpisodeSettings,
    Segment,
    SegmentVersion,
    Source,
    Turn,
)
from tests.test_fast import _wait


def _episode_and_source():
    chunks = [Chunk(id="c1", section="Respiration", text="Respiration releases energy from food in the mitochondria.", page=3),
              Chunk(id="c2", section="Photosynthesis", text="Leaves make food using sunlight, water and carbon dioxide.", page=5),
              Chunk(id="c3", section="Transport", text="Xylem carries water from the roots to the leaves.", page=8)]
    src = Source(id="src_ask", title="Life processes", origin="paste", sections=[], chunks=chunks, word_count=30)
    seg = Segment(id="s1", index=0, title="Food factories", goal="g", chunk_ids=["c2"], active_version=1,
                  versions=[SegmentVersion(version=1, turns=[Turn(speaker="host", text="How do plants eat?"),
                                                             Turn(speaker="expert", text="They make food in leaves.")])])
    ep = Episode(id="ep_ask", source_id=src.id, title=src.title, settings=EpisodeSettings(), segments=[seg])
    return ep, src


def test_retrieval_finds_the_passage_and_adds_the_current_part():
    ep, src = _episode_and_source()
    got = [c.id for c in ask.retrieve(ep, src, "What carries water up from the roots?", segment_id="s1")]
    assert got[0] == "c3" and "c2" in got


def test_follow_up_and_cross_language_questions_still_get_passages():
    ep, src = _episode_and_source()
    follow = ask.retrieve(ep, src, "why?", history=[{"q": "Where is energy released by respiration?", "a": "..."}])
    assert follow and follow[0].id == "c1"
    hindi = "पानी जड़ों से पत्तियों तक कैसे जाता है?"
    assert ask.needs_translation(src, hindi) and not ask.needs_translation(src, "How does water move up?")
    # no words in common: fall back to what the episode taught...
    assert [c.id for c in ask.retrieve(ep, src, hindi)] == ["c2"]
    # ...unless the question was turned into English search words first
    assert ask.retrieve(ep, src, hindi, extra_query="water roots xylem leaves")[0].id == "c3"


def test_ask_endpoint_answers_with_citations_and_voice():
    with TestClient(app) as client:
        sample = client.get("/api/samples").json()[0]["id"]
        src = client.post("/api/sources", data={"sample_id": sample}).json()
        eid = client.post("/api/episodes", json={"source_id": src["id"], "settings": {"length": "short", "mode": "fast"}}).json()["id"]
        ep = _wait(client, eid, lambda e: e["status"] in ("done", "failed") and not e["busy"])
        assert ep["status"] == "done"
        topic = ep["segments"][0]["title"]
        r = client.post(f"/api/episodes/{eid}/ask", json={"question": f"Can you explain {topic} again?",
                                                           "segment_id": ep["segments"][0]["id"]})
        assert r.status_code == 200, r.text
        a = r.json()
        assert a["a"] and a["grounded"] and a["sources"] and a["sources"][0]["id"] in a["retrieved"]
        assert a["follow_ups"]
        audio = client.get(f"/api/episodes/{eid}/ask/{a['id']}/audio")
        assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
        again = client.get(f"/api/episodes/{eid}/ask/{a['id']}/audio")  # cached
        assert again.content == audio.content
        off = client.post(f"/api/episodes/{eid}/ask", json={"question": "Who won the cricket match?",
                                                             "history": [{"q": a["q"], "a": a["a"]}]}).json()
        assert off["grounded"] is False and off["sources"] == []
        assert client.post(f"/api/episodes/{eid}/ask", json={"question": ""}).status_code == 422
        assert client.post(f"/api/episodes/{eid}/ask", json={"question": "x" * 600}).status_code == 422
        assert client.get(f"/api/episodes/{eid}/ask/a_nope/audio").status_code == 404
        assert client.post("/api/episodes/ep_missing/ask", json={"question": "hi there"}).status_code == 404


def test_a_hindi_doubt_gets_search_words_first():
    import asyncio
    ep, src = _episode_and_source()
    words = asyncio.run(ask.search_words(src, "पौधे खाना कैसे बनाते हैं?"))
    assert "food" in words
