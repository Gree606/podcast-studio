"""End-to-end integration test in mock mode: upload -> generate -> regenerate -> switch version.

Run:  MOCK_MODE=1 python scripts/smoke_test.py [path/to/chapter.pdf]
Exits non-zero on failure. Uses a temporary data directory.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOCK_MODE", "1")
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="podcast_smoke_")
os.environ.setdefault("PODCAST_SECRETS_FILE", os.path.join(tempfile.mkdtemp(), "none.env"))  # never touch the real keys

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402


def wait(client: TestClient, eid: str, cond, timeout: float = 120) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        ep = client.get(f"/api/episodes/{eid}").json()
        if cond(ep):
            return ep
        time.sleep(0.2)
    raise SystemExit(f"timeout waiting for episode {eid}: {ep['status']} {ep['message']}")


def main() -> None:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    with TestClient(app) as c:
        assert c.get("/api/health").json()["ok"]
        if pdf:
            r = c.post("/api/sources", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
        else:
            sample = c.get("/api/samples").json()[0]["id"]
            r = c.post("/api/sources", data={"sample_id": sample})
        assert r.status_code == 200, r.text
        src = r.json()
        print(f"source: {src['title']!r}, {src['word_count']} words, {len(src['sections'])} sections")

        r = c.post("/api/episodes", json={"source_id": src["id"], "settings": {"length": "short"}})
        assert r.status_code == 200, r.text
        eid = r.json()["id"]
        ep = wait(c, eid, lambda e: e["status"] in ("done", "failed") and not e["busy"])
        assert ep["status"] == "done", ep["error"]
        kinds = {t["kind"] for t in c.get(f"/api/episodes/{eid}").json()["trace"]}
        assert {"thought", "action", "observation", "decision"} <= kinds, kinds
        assert all(s["status"] == "done" for s in ep["segments"]), [s["status"] for s in ep["segments"]]
        a = c.get(f"/api/episodes/{eid}/audio")
        assert a.status_code == 200 and len(a.content) > 10000
        turns = [t for s in ep["segments"] for t in s["versions"][-1]["turns"]]
        assert all(t["start"] is not None for t in turns)
        assert all(t1["start"] >= t0["start"] for t0, t1 in zip(turns, turns[1:])), "timestamps not monotonic"
        print(f"episode: {ep['duration']}s, {len(ep['segments'])} segments, {ep['metrics']['llm_calls']} llm calls")

        seg = ep["segments"][1]
        r = c.post(f"/api/episodes/{eid}/segments/{seg['id']}/regenerate", json={"instruction": "make it shorter"})
        assert r.status_code == 200, r.text
        ep2 = wait(c, eid, lambda e: not e["busy"] and e["segments"][1]["active_version"] == 2)
        assert len(ep2["segments"][1]["versions"]) == 2
        assert ep2["segments"][0]["active_version"] == 1, "other segments must not change"
        assert ep2["audio_file"] != ep["audio_file"]
        r = c.post(f"/api/episodes/{eid}/segments/{seg['id']}/activate", json={"version": 1})
        assert r.status_code == 200
        ep3 = c.get(f"/api/episodes/{eid}").json()
        assert ep3["segments"][1]["active_version"] == 1
        t = c.get(f"/api/episodes/{eid}/transcript").text
        assert "##" in t and "Sources" in t
        print("regenerate + version switch OK")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
