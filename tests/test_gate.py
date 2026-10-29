"""The entry gate: with ACCESS_CODE set, only people with the secret code can spend API calls."""
from fastapi.testclient import TestClient

from backend import main
from backend.config import get_settings
from backend.main import app


def test_gate_blocks_spending_until_the_code_is_entered(monkeypatch):
    monkeypatch.setattr(get_settings(), "access_code", "Earthling606")
    main._hits.clear()
    with TestClient(app) as client:
        assert client.get("/api/config").json()["locked"] is True
        assert client.get("/api/config").json()["unlocked"] is False
        sample = client.get("/api/samples").json()[0]["id"]  # reading stays open
        r = client.post("/api/sources", data={"sample_id": sample})
        assert r.status_code == 401 and r.json()["locked"] is True
        assert client.post("/api/unlock", json={"code": "nope"}).status_code == 401
        ok = client.post("/api/unlock", json={"code": " earthling606 "})  # case and spaces don't matter
        assert ok.status_code == 200 and main.GATE_COOKIE in ok.cookies
        assert client.get("/api/config").json()["unlocked"] is True
        assert client.post("/api/sources", data={"sample_id": sample}).status_code == 200


def test_gate_slows_down_guessing(monkeypatch):
    monkeypatch.setattr(get_settings(), "access_code", "Earthling606")
    main._hits.clear()
    with TestClient(app) as client:
        codes = [client.post("/api/unlock", json={"code": f"guess{i}"}).status_code for i in range(10)]
        assert codes[:8] == [401] * 8 and codes[8:] == [429, 429]


def test_no_code_configured_means_no_gate():
    assert get_settings().access_code == ""
    with TestClient(app) as client:
        assert client.get("/api/config").json()["locked"] is False
        assert client.post("/api/unlock", json={"code": "anything"}).json() == {"ok": True}
