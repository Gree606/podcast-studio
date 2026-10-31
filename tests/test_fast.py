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


def test_lines_recorded_during_the_fact_check_are_reused_not_recorded_twice(tmp_path):
    from backend.agent.voice import prefetch, synthesize_version
    from backend.schemas import Turn

    ctx, m = _ctx(), Metrics()

    async def run():
        segs = await fast.write_episode(ctx)
        seg = segs[0]
        prefetch(ctx, seg, seg.candidates[-1].turns)
        changed = [t.model_copy() for t in seg.candidates[-1].turns]
        changed[0] = Turn(speaker=changed[0].speaker, text="A brand new opening line about reactions, rewritten by the fact-check.")
        v = fast.new_version(seg, [])
        v.turns = changed
        await synthesize_version(ctx, seg, v, tmp_path / "x.wav")
        return len(changed)

    token = current_metrics.set(m)
    try:
        n = asyncio.run(run())
    finally:
        current_metrics.reset(token)
    # every original line recorded once (early), plus only the one line the check changed
    assert m.tts_calls == n + 1
    assert not ctx.tts_cache or all(t.done() for t in ctx.tts_cache.values())


def test_a_timed_out_model_falls_back_to_the_next_one(monkeypatch):
    from backend.clients import llm
    from backend.clients.base import ServiceError

    tried = []

    async def call(messages, *, model, attempts=None, **kw):
        tried.append(model)
        if model == "slow":
            raise ServiceError("Language model", "timed out", "read timeout", status=504)
        return "ok", {"prompt_tokens": 1, "completion_tokens": 1}

    text, _, used = asyncio.run(llm._call_with_fallback(call, [], models=["slow", "fast"], temperature=0, max_tokens=10,
                                                         json_mode=False, base_url="", api_key=""))
    assert (text, used, tried) == ("ok", "fast", ["slow", "fast"])


def test_line_fixes_change_only_the_named_lines():
    from backend.schemas import Turn
    turns = [Turn(speaker="host", text="Hi!"), Turn(speaker="expert", text="Plants use tiny muscles."), Turn(speaker="host", text="Cool.")]
    out = fast.apply_line_fixes(turns, [{"line": 2, "text": "2. Gree: Plants move without muscles."}, {"line": 9, "text": "x"}])
    assert [(t.speaker, t.text) for t in out] == [("host", "Hi!"), ("expert", "Plants move without muscles."), ("host", "Cool.")]
    assert fast.apply_line_fixes(turns, [{"line": 3, "text": ""}])[-1].text == "Plants use tiny muscles."
    assert fast.apply_line_fixes(turns, []) is None


def test_languages_reach_every_stage():
    from backend import languages
    from backend.audio.wer import wer
    from backend.clients.tts import bodhan_body, split_text
    from backend.textnorm import normalize_for_tts

    assert languages.get("ta").tts == "ta-IN" and languages.get("nope").code == "en"
    assert "Tamil" in languages.writing_rule("ta") and "Hinglish" in languages.writing_rule("hinglish")
    assert bodhan_body("नमस्ते", "host", "hi-IN")["instructions"] == '{"lang": "hi"}'
    assert split_text("पहला वाक्य। दूसरा वाक्य।", 12) == ["पहला वाक्य।", "दूसरा वाक्य।"]
    # Indic: a misspelled name is a small miss (not 0% as before the fix, nor 100%); English words written in
    # Devanagari by the recogniser don't count; a cut-off take does
    assert 0 < wer("अरे ग्री आज हम धरती के बारे में", "अरे घ्री आज हम धरती के बारे में", "hi") < 0.1
    assert wer("Earth पर chemical reaction का कैसे पता चलता है?", "अर्थ पर केमिकल रिएक्शन का कैसे पता चलता है", "hinglish") < 0.1
    assert wer("नहीं! पौधे अपना भोजन खुद बनाते हैं जल और खनिजों से।", "नहीं पौधे", "hi") > 0.6
    assert wer("What about burning a magnesium ribbon?", "वॉट अबाउट बर्निंग", "hinglish") is None
    assert normalize_for_tts("Error606 और 1,200", "hi") == "Error six oh six और 1,200"


def test_episode_in_hindi_end_to_end_in_mock_mode():
    with TestClient(app) as client:
        cfg = client.get("/api/config").json()
        assert {"en", "hi", "ta", "hinglish"} <= {x["code"] for x in cfg["languages"]}
        sample = client.get("/api/samples").json()[0]["id"]
        src = client.post("/api/sources", data={"sample_id": sample}).json()
        bad = client.post("/api/episodes", json={"source_id": src["id"], "settings": {"language": "klingon"}})
        assert bad.status_code == 400
        eid = client.post("/api/episodes", json={"source_id": src["id"], "settings": {"language": "hi"}}).json()["id"]
        ep = _wait(client, eid, lambda e: e["status"] in ("done", "failed") and not e["busy"])
        assert ep["status"] == "done" and ep["settings"]["language"] == "hi"


def test_mode_is_chosen_per_episode():
    with TestClient(app) as client:
        sample = client.get("/api/samples").json()[0]["id"]
        src = client.post("/api/sources", data={"sample_id": sample}).json()
        deep = client.post("/api/episodes", json={"source_id": src["id"], "settings": {"length": "short", "mode": "agent"}}).json()["id"]
        quick = client.post("/api/episodes", json={"source_id": src["id"], "settings": {"length": "short", "mode": "fast"}}).json()["id"]
        d = _wait(client, deep, lambda e: e["status"] in ("done", "failed") and not e["busy"])
        q = _wait(client, quick, lambda e: e["status"] in ("done", "failed") and not e["busy"])
        assert d["status"] == "done" and q["status"] == "done"
        assert q["metrics"]["llm_calls"] == 2
        assert d["metrics"]["llm_calls"] > 2  # planner steps, drafts, critic, claim checks
        assert any(t["stage"] == "plan" for t in d["trace"]) and not any(t["stage"] == "plan" for t in q["trace"])
        bad = client.post("/api/episodes", json={"source_id": src["id"], "settings": {"mode": "agent", "language": "hi"}})
        assert bad.status_code == 400


def test_quick_fact_check_is_per_spoken_line():
    from backend.agent.fast import parse_claims
    from backend.schemas import Turn
    turns = [Turn(speaker="host", text="Hi!"), Turn(speaker="expert", text="Mg plus O2 gives MgO is skeletal."),
             Turn(speaker="host", text="So it's balanced?"), Turn(speaker="expert", text="No, we balance it.")]
    claims = parse_claims({"lines": [{"line": 1, "claim": ""}, {"line": 2, "claim": "Mg + O2 -> MgO is skeletal", "supported": True,
                                                                 "evidence": "c1"},
                                     {"line": 9, "claim": "out of range", "supported": False}]}, turns, {"c1": object()})
    assert [(c.line, c.supported, c.evidence_chunk_id) for c in claims] == [(2, True, "c1"), (None, False, None)]
    assert parse_claims({"claims": [{"claim": "old shape", "supported": True}]}, turns, {})[0].line is None
    with TestClient(app) as client:
        sample = client.get("/api/samples").json()[0]["id"]
        src = client.post("/api/sources", data={"sample_id": sample}).json()
        eid = client.post("/api/episodes", json={"source_id": src["id"], "settings": {"length": "short", "mode": "fast"}}).json()["id"]
        ep = _wait(client, eid, lambda e: e["status"] in ("done", "failed") and not e["busy"])
        assert ep["pipeline"] == "fast" and ep["can_edit"] is True
        claims = [c for s in ep["segments"] for v in s["versions"] if v["version"] == s["active_version"] for c in v["claims"]]
        assert claims and all(c["line"] for c in claims)
        assert any(not c["supported"] and c["corrected"] for c in claims)  # the mock flags (and fixes) one line in part 2


def test_writer_accepts_nested_parts_and_retries_a_wrong_shape(monkeypatch):
    """Seen live (Malayalam chapter): the writer's JSON had no top-level "parts"."""
    import asyncio as _asyncio
    from backend.agent import fast as _fast
    part = {"title": "P", "chunk_ids": [], "turns": [{"speaker": "host", "text": "Hi"}]}
    assert _fast.find_parts({"episode": {"parts": [part]}}) == [part]
    assert _fast.find_parts([part]) == [part]
    assert _fast.find_parts({"speaker": "host", "text": "a lone line"}) == []

    replies = iter([{"speaker": "host", "text": "a lone line"}, {"title": "T", "parts": [part]}])
    calls = []

    async def fake_chat_json(msgs, **kw):
        calls.append(len(msgs))
        return next(replies)
    monkeypatch.setattr(_fast, "chat_json", fake_chat_json)

    class Ctx:
        def emit(self, *a, **k):
            pass
    data, parts = _asyncio.run(_fast.write_episode_call(Ctx(), "prompt"))
    assert parts == [part] and calls == [1, 2]  # one corrective retry, with the correction appended
