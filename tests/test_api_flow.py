"""API-level checks in mock mode: presets, static files, the full flow, and failure + resume (criteria 6, 7, 11, 12)."""
import time

import pytest
from fastapi.testclient import TestClient

from backend.main import PRESETS, app
from backend.schemas import EpisodeSettings


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def wait(client, eid, cond, timeout=180.0):
    t0, ep = time.time(), None
    while time.time() - t0 < timeout:
        ep = client.get(f"/api/episodes/{eid}").json()
        if cond(ep):
            return ep
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {eid}: {ep and (ep['status'], ep['message'])}")


def finished(ep):
    return ep["status"] in ("done", "failed") and not ep["busy"]


def new_episode(client, settings=None):
    sample = client.get("/api/samples").json()[0]["id"]
    src = client.post("/api/sources", data={"sample_id": sample}).json()
    r = client.post("/api/episodes", json={"source_id": src["id"], "settings": settings or {"length": "short"}})
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ---------------------------------------------------------------- criterion 6
def test_presets_have_kid_labels_and_valid_values(client):
    cfg = client.get("/api/config").json()
    assert cfg["presets"] == PRESETS
    assert [p["label"] for p in cfg["presets"]["length"]] == ["Quick Bite", "Story Time", "Deep Dive"]
    assert [p["label"] for p in cfg["presets"]["tone"]] == ["Fun", "Balanced", "Exam Hero"]
    for p in cfg["presets"]["length"]:
        assert EpisodeSettings(length=p["value"]).length == p["value"]
        assert p["value"] in cfg["lengths"]
    for p in cfg["presets"]["tone"]:
        assert EpisodeSettings(tone=p["value"]).tone == p["value"]
    assert all(p["emoji"] and p["hint"] for group in cfg["presets"].values() for p in group)


# ---------------------------------------------------------------- criterion 7
def test_new_ui_static_files_and_classic_fallback(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "/app.js" in r.text and "/app.css" in r.text and "Mission 606" in r.text
    css = client.get("/app.css")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css") and "prefers-reduced-motion" in css.text
    js = client.get("/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"] and "renderQuiz" in js.text
    old = client.get("/classic")
    assert old.status_code == 200 and old.headers["content-type"].startswith("text/html")
    assert "Podcast Studio" in old.text and "<script>" in old.text


# ---------------------------------------------------------------- criterion 11
def test_full_flow_sample_generate_regenerate_switch_quiz_transcript(client):
    eid = new_episode(client)
    ep = wait(client, eid, finished)
    assert ep["status"] == "done", ep["error"]
    assert all(s["status"] == "done" for s in ep["segments"])

    a = client.get(f"/api/episodes/{eid}/audio")
    assert a.status_code == 200 and a.headers["content-type"].startswith("audio/") and len(a.content) > 10000

    seg = ep["segments"][1]
    r = client.post(f"/api/episodes/{eid}/segments/{seg['id']}/regenerate",
                    json={"instruction": "Explain this more simply, as if for a younger student. Keep every fact."})
    assert r.status_code == 200, r.text
    ep2 = wait(client, eid, lambda e: not e["busy"] and e["segments"][1]["active_version"] == 2)
    s2 = ep2["segments"][1]
    assert len(s2["versions"]) == 2 and s2["versions"][1]["instruction"].startswith("Explain this more simply")
    assert ep2["segments"][0]["active_version"] == 1, "other segments must not change"
    assert ep2["audio_file"] != ep["audio_file"]

    r = client.post(f"/api/episodes/{eid}/segments/{seg['id']}/activate", json={"version": 1})
    assert r.status_code == 200
    ep3 = wait(client, eid, lambda e: not e["busy"])
    assert ep3["segments"][1]["active_version"] == 1

    quiz = client.get(f"/api/episodes/{eid}/quiz").json()
    assert 3 <= len(quiz) <= 5
    for q in quiz:
        assert q["answer"] in q["options"] and q["question"].count("_____") == 1

    t = client.get(f"/api/episodes/{eid}/transcript").text
    assert t.startswith("# ") and "##" in t and "Sources" in t
    assert client.get(f"/api/episodes/{eid}/sources").json()


def test_quiz_for_unknown_episode_is_a_friendly_404(client):
    r = client.get("/api/episodes/ep_nope/quiz")
    assert r.status_code == 404 and "not found" in r.json()["detail"].lower()


# ---------------------------------------------------------------- criterion 12
def test_failed_episode_shows_friendly_message_and_resume_finishes_it(client, monkeypatch):
    async def broken_plan(ctx):
        raise RuntimeError("internal detail xyzzy\nTraceback (most recent call last): boom")

    monkeypatch.setattr("backend.agent.orchestrator.plan", broken_plan)
    monkeypatch.setattr("backend.agent.fast.write_episode", broken_plan)  # the default 2-call pipeline
    eid = new_episode(client)
    ep = wait(client, eid, finished)
    assert ep["status"] == "failed"
    for text in (ep["error"], ep["message"]):
        assert text and "Traceback" not in text and "xyzzy" not in text and "RuntimeError" not in text
    assert "try again" in ep["message"].lower()

    monkeypatch.undo()
    r = client.post(f"/api/episodes/{eid}/resume")
    assert r.status_code == 200
    ep = wait(client, eid, finished)
    assert ep["status"] == "done" and ep["error"] is None and ep["audio_file"]


def test_only_the_creator_can_change_an_episode_and_there_is_no_public_list():
    from backend.main import OWNER_COOKIE
    with TestClient(app) as owner:
        eid = new_episode(owner)
        ep = wait(owner, eid, finished)
        seg = ep["segments"][0]["id"]
        assert owner.get(f"/api/episodes/{eid}").json()["can_edit"] is True
        assert owner.get("/api/episodes").json() == []  # nobody can list everyone's episodes
        assert [e["id"] for e in owner.get(f"/api/episodes?ids={eid}").json()] == [eid]
        assert "secrets_path" not in owner.get("/api/config").json()
        visitor = TestClient(app)  # opened the shared link in another browser
        view = visitor.get(f"/api/episodes/{eid}").json()
        assert view["can_edit"] is False and "owner" not in view
        for path, body in ((f"/api/episodes/{eid}/segments/{seg}/regenerate", {"instruction": "x"}),
                           (f"/api/episodes/{eid}/segments/{seg}/activate", {"version": 1}),
                           (f"/api/episodes/{eid}/restart", None), (f"/api/episodes/{eid}/resume", None)):
            assert visitor.post(path, json=body).status_code == 403, path
        visitor.cookies.set(OWNER_COOKIE, "guessed")
        assert visitor.post(f"/api/episodes/{eid}/restart").status_code == 403
        assert owner.post(f"/api/episodes/{eid}/restart").status_code == 200  # done: nothing to do, but allowed


def test_deep_agent_stays_within_ten_llm_calls(client):
    """The deep agent plans with tools, writes, critiques, revises the weakest part, and fact-checks,
    all within its 10-call budget (backend/agent/lean.py)."""
    eid = new_episode(client, {"length": "medium", "mode": "agent"})
    ep = wait(client, eid, finished)
    assert ep["status"] == "done", ep["message"]
    assert ep["metrics"]["llm_calls"] <= 10, ep["metrics"]["llm_calls"]
    kinds = {e["kind"] for e in ep["trace"]}
    assert {"thought", "action", "observation", "decision"} <= kinds  # the agent loop is visible
    texts = " ".join(e["text"] for e in ep["trace"])
    assert "Critic (1 call)" in texts and "Fact-check (1 call)" in texts
