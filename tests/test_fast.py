"""The default FAST pipeline: exactly 2 LLM calls per episode and per single-part regeneration."""
import asyncio

from fastapi.testclient import TestClient
from test_agent import CHAPTER

from backend.agent import fast
from backend.agent.context import AgentContext
from backend.clients.usage import current_metrics
from backend.ingest import ingest_text, select_chunks
from backend.main import app
from backend.schemas import Episode, EpisodeSettings, Metrics


def _ctx():
    src = ingest_text(CHAPTER, 9000)
    ep = Episode(id="ep_fast", source_id=src.id, title=src.title, settings=EpisodeSettings(length="short"))
    chunks, _ = select_chunks(src, [], 9000)
    return AgentContext(episode=ep, source=src, chunks=chunks)


def test_write_and_verify_use_two_calls_and_apply_fixes():
    ctx, m = _ctx(), Metrics()
    token = current_metrics.set(m)
    try:
        segs = asyncio.run(fast.write_episode(ctx))
        claims = asyncio.run(fast.verify_segments(ctx, segs))
    finally:
        current_metrics.reset(token)
    assert m.llm_calls == 2
    assert [s.kind for s in segs] == ["intro", "body", "recap"]
    assert all(c in ctx.by_id for s in segs for c in s.chunk_ids)
    # the mock verifier flags one claim in part 2 and returns fixed turns: they must be used
    assert len(segs[1].candidates) == 2 and "step by step" in segs[1].candidates[-1].turns[1].text
    assert any(not c.supported for c in claims[segs[1].id])
    assert all(kf.verified and kf.chunk_id for s in segs for kf in s.key_facts)
    v = fast.new_version(segs[1], claims[segs[1].id])
    assert v.version == 1 and v.turns == segs[1].candidates[-1].turns


def test_select_passages_spreads_across_long_chapters():
    ctx = _ctx()
    ids = [c.id for c in ctx.chunks]
    picked = fast.select_passages(ctx, limit_words=150)
    assert picked and set(picked) <= set(ids)
    assert fast.select_passages(ctx, limit_words=10**6) == ids


def test_episode_and_regeneration_llm_call_counts():
    with TestClient(app) as client:
        sample = client.get("/api/samples").json()[0]["id"]
        src = client.post("/api/sources", data={"sample_id": sample}).json()
        eid = client.post("/api/episodes", json={"source_id": src["id"], "settings": {"length": "medium"}}).json()["id"]
        ep = _wait(client, eid, lambda e: e["status"] in ("done", "failed") and not e["busy"])
        assert ep["status"] == "done", ep["error"]
        assert ep["metrics"]["llm_calls"] == 2
        assert len(ep["segments"]) == 3 and ep["duration"] > 0
        seg = ep["segments"][0]
        client.post(f"/api/episodes/{eid}/segments/{seg['id']}/regenerate", json={"instruction": "Make it funnier"})
        ep2 = _wait(client, eid, lambda e: not e["busy"] and e["segments"][0]["active_version"] == 2)
        assert ep2["metrics"]["llm_calls"] == 4
        assert ep2["segments"][1]["active_version"] == 1


def _wait(client, eid, cond, timeout=120.0):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        ep = client.get(f"/api/episodes/{eid}").json()
        if cond(ep):
            return ep
        time.sleep(0.1)
    raise AssertionError("timed out")


def test_verify_matches_a_renumbered_lone_part(monkeypatch):
    ctx = _ctx()
    segs = asyncio.run(fast.write_episode(ctx))

    async def renumbered(*a, **k):
        return {"parts": [{"part": 1, "claims": [{"claim": "x", "supported": True, "evidence": "c1"}], "fixed_turns": None}]}

    monkeypatch.setattr(fast, "chat_json", renumbered)
    claims = asyncio.run(fast.verify_segments(ctx, segs, only=[segs[2]]))
    assert [c.claim for c in claims[segs[2].id]] == ["x"]
