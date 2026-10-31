"""Settings panel: each episode picks its script AI and voices; keys never leave the server."""
import asyncio

from fastapi.testclient import TestClient

from backend import providers
from backend.config import Settings, get_settings
from backend.main import app


def settings(**kw) -> Settings:
    kw.setdefault("mock_mode", False)
    return Settings(_env_file=None, **kw)  # never read the repo .env or the secrets file here


# Like the live server: Gemini + Sarvam by default, one Bodhan key for ASR/OCR, no OpenRouter key yet.
LIVE = dict(llm_api_key="AIzaSyRealGemini", tts_api_key="sk_real_sarvam", asr_api_key="bodhan_real")


def test_options_offer_only_what_has_a_key():
    o = providers.options(settings(**LIVE))
    assert {x["id"]: (x["ready"], x["default"]) for x in o["llm"]} == {"gemini": (True, True), "openrouter": (False, False)}
    # Bodhan voices reuse the Bodhan ASR key: one key serves TTS, ASR and OCR
    assert {x["id"]: (x["ready"], x["default"]) for x in o["voices"]} == {"sarvam": (True, True), "bodhan": (True, False)}
    assert "sk_real" not in str(o) and "AIza" not in str(o)


def test_default_choice_changes_nothing():
    s = settings(**LIVE)
    assert providers.settings_for("gemini", "sarvam", base=s) is s
    assert providers.settings_for("", "", base=s) is s


def test_bodhan_voices_and_openrouter_override_only_that_episode():
    s = settings(**LIVE, openrouter_api_key="sk-or-v1-real")
    t = providers.settings_for("openrouter", "bodhan", base=s)
    assert (t.llm_provider, t.llm_base_url, t.llm_model) == ("openrouter", providers.OPENROUTER_URL, "google/gemini-3.6-flash")
    assert t.llm_key() == "sk-or-v1-real"
    assert (t.tts_api_style, t.tts_voice_expert, t.tts_api_key, t.tts_requests_per_minute) == ("bodhan", "Deepika", "bodhan_real", 8)
    assert (s.llm_provider, s.tts_api_style, s.tts_api_key) == ("gemini", "sarvam", "sk_real_sarvam")  # server default untouched


def test_sarvam_needs_its_own_key_when_bodhan_is_the_default():
    s = settings(tts_api_style="bodhan", tts_api_key="bodhan_real", llm_api_key="AIzaSyReal")
    assert {x["id"]: x["ready"] for x in providers.options(s)["voices"]} == {"sarvam": False, "bodhan": True}
    t = providers.settings_for("", "sarvam", base=settings(tts_api_style="bodhan", sarvam_api_key="sk_s"))
    assert (t.tts_api_style, t.tts_api_key) == ("sarvam", "sk_s")


def test_override_is_per_task():
    async def main():
        async def inside():
            with providers.using("", "bodhan"):
                await asyncio.sleep(0)
                return get_settings().tts_api_style
        outside = get_settings().tts_api_style
        assert await asyncio.create_task(inside()) == "bodhan"
        assert get_settings().tts_api_style == outside
    asyncio.run(main())


def test_api_rejects_an_option_without_a_key_and_records_the_choice():
    with TestClient(app) as client:
        cfg = client.get("/api/config").json()
        assert {"llm", "voices"} <= set(cfg["providers"])
        sample = client.get("/api/samples").json()[0]["id"]
        src = client.post("/api/sources", data={"sample_id": sample}).json()["id"]
        r = client.post("/api/episodes", json={"source_id": src, "settings": {"voices": "nope"}})
        assert r.status_code == 422
        r = client.post("/api/episodes", json={"source_id": src, "settings": {"llm": "gemini", "voices": "bodhan"}})
        assert r.status_code == 200
        ep = client.get(f"/api/episodes/{r.json()['id']}").json()
        assert ep["settings"]["voices"] == "bodhan" and ep["providers"]["voices"] == "Bodhan indic-speak"
