"""Owner's real keys: each service is real on its own key, and Bodhan Indic-OCR uses the documented wire format."""
import asyncio
import base64

import httpx
import pytest

from backend.clients import ocr
from backend.clients.base import ServiceError
from backend.config import Settings

PNG = b"\x89PNG\r\n\x1a\nfake-page"


def settings(**kw) -> Settings:
    kw.setdefault("mock_mode", False)
    return Settings(_env_file=None, **kw)  # never read the repo .env or the secrets file here


def test_each_service_is_real_as_soon_as_its_own_key_is_set():
    s = settings(llm_api_key="DUMMY_REPLACE_WITH_GEMINI_KEY", tts_api_key="sk_real_sarvam", asr_api_key="bodhan_real",
                 ocr_mode="bodhan", ocr_api_key="bodhan_real")
    assert s.llm_mock() is True
    assert s.tts_mock() is False and s.asr_mock() is False
    assert s.configured() == {"llm": False, "tts": True, "asr": True, "ocr": True}
    assert s.effective_mock() is True  # the UI still says demo: the script writer is a placeholder


def test_mock_mode_forces_every_service_to_the_fake():
    s = settings(mock_mode=True, llm_api_key="AIzaSyReal", tts_api_key="sk_real", asr_api_key="b_real")
    assert s.llm_mock() and s.tts_mock() and s.asr_mock() and s.effective_mock()


def test_asr_follows_tts():
    # Placeholder voices are checked by the placeholder recogniser, never by the real (billed) ASR.
    s = settings(tts_api_key="DUMMY_X", asr_api_key="bodhan_real")
    assert s.tts_mock() and s.asr_mock()


def test_ocr_ready_per_mode():
    assert settings(ocr_mode="bodhan", ocr_api_key="DUMMY_REPLACE_WITH_BODHAN_KEY").ocr_ready() is False
    assert settings(ocr_mode="bodhan", ocr_api_key="real").ocr_ready() is True
    assert settings(ocr_mode="http", ocr_base_url="http://x").ocr_ready() is True
    assert settings(ocr_mode="llm", llm_api_key="AIzaSyReal").ocr_ready() is True
    assert settings(ocr_mode="off", llm_api_key="AIzaSyReal").ocr_ready() is False


def test_bodhan_request_is_one_user_message_with_one_image():
    body = ocr.bodhan_request(PNG)
    assert body["model"] == "indic-ocr"
    (msg,) = body["messages"]
    assert msg["role"] == "user"
    (part,) = msg["content"]
    assert part["type"] == "image_url"
    assert part["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert "language" not in body and "temperature" not in body


@pytest.fixture
def bodhan(monkeypatch):
    calls = []
    s = settings(ocr_mode="bodhan", ocr_api_key="bodhan_test_key")
    monkeypatch.setattr(ocr, "get_settings", lambda: s)

    def respond(payload):
        async def fake(service, method, url, **kw):
            calls.append({"service": service, "method": method, "url": url, **kw})
            return httpx.Response(200, json=payload)
        monkeypatch.setattr(ocr, "request_with_retry", fake)

    return calls, respond


def test_bodhan_ocr_wire_format(bodhan):
    calls, respond = bodhan
    respond({"object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Chapter 1\n\nPlants make food."}}],
             "blocks": [{"order": 0, "label": "Section-title", "text": "Chapter 1"}]})
    assert ocr.ocr_available()
    text = asyncio.run(ocr.ocr_page(PNG))
    assert text == "Chapter 1\n\nPlants make food."
    (c,) = calls
    assert c["method"] == "POST" and c["url"] == "https://api.bodhan.ai/v1/chat/completions"
    assert c["headers"] == {"Authorization": "Bearer bodhan_test_key"}
    assert c["json"]["model"] == "indic-ocr"


def test_bodhan_ocr_unexpected_response_is_a_friendly_error(bodhan):
    _, respond = bodhan
    respond({"error": "nope"})
    with pytest.raises(ServiceError, match="unexpected response"):
        asyncio.run(ocr.ocr_page(PNG))


def test_bodhan_mode_without_a_key_falls_back_to_the_llm_path(monkeypatch):
    s = settings(ocr_mode="bodhan", ocr_api_key="DUMMY_REPLACE_WITH_BODHAN_KEY")
    monkeypatch.setattr(ocr, "get_settings", lambda: s)

    async def no_http(*a, **kw):
        raise AssertionError("must not call Bodhan without a key")

    async def fake_chat(messages, **kw):
        assert kw["task"] == "ocr"
        return "from the llm path"

    monkeypatch.setattr(ocr, "request_with_retry", no_http)
    monkeypatch.setattr(ocr, "chat", fake_chat)
    assert asyncio.run(ocr.ocr_page(PNG)) == "from the llm path"


# ---------------------------------------------------------------- real ASR vs contractions
def test_wer_ignores_contractions_the_real_asr_expands():
    from backend.audio.wer import normalize_for_wer, wer

    said = "Hi, I'm your host. Today we're learning how plants make food using sunlight, water and carbon dioxide."
    heard = "Hi I am your host Today we are learning how plants make food using sunlight water and carbon dioxide"
    assert wer(said, heard) == 0.0  # was 24%, right at the 25% re-record threshold
    assert normalize_for_wer("it’s what we'll see, can't we? Let's go, they've won't") == \
        ["it", "is", "what", "we", "will", "see", "can", "not", "we", "let", "us", "go", "they", "have", "will", "not"]
    assert normalize_for_wer("Riya's book") == ["riya's", "book"]  # possessives are not expanded
    assert wer("plants make food", "plants make fruit") > 0.3  # real errors still count


def test_asr_names_the_language_by_default():
    assert settings().asr_language_code() == "en"  # from TTS_LANGUAGE=en-IN
    assert settings(tts_language="hi-IN").asr_language_code() == "hi"
    assert settings(asr_language="ta").asr_language_code() == "ta"
    assert settings(asr_language="auto").asr_language_code() == ""


def test_asr_sends_language_and_retries_longer(monkeypatch):
    from backend.clients import asr

    s = settings(tts_api_key="sk_real", asr_api_key="bodhan_real")
    monkeypatch.setattr(asr, "get_settings", lambda: s)
    seen = {}

    async def fake(service, method, url, **kw):
        seen.update(kw, url=url)
        return httpx.Response(200, json={"text": "hello"})

    monkeypatch.setattr(asr, "request_with_retry", fake)
    monkeypatch.setattr(asr, "_sem", None)
    assert asyncio.run(asr.transcribe(b"RIFF")) == "hello"
    assert seen["url"] == "https://api.bodhan.ai/v1/audio/transcriptions"
    assert seen["data"] == {"model": "indic-transcribe", "language": "en"}
    assert seen["attempts"] >= 5 and s.asr_max_concurrency <= 2


def test_asr_pacing_keeps_calls_under_the_per_minute_limit(monkeypatch):
    from backend.clients import asr

    s = settings(asr_requests_per_minute=3)
    monkeypatch.setattr(asr, "get_settings", lambda: s)
    monkeypatch.setattr(asr, "_sent", __import__("collections").deque())
    clock = {"t": 1000.0}
    slept = []
    monkeypatch.setattr(asr.time, "monotonic", lambda: clock["t"])

    async def fake_sleep(d):
        slept.append(d)
        clock["t"] += d

    monkeypatch.setattr(asr.asyncio, "sleep", fake_sleep)

    async def run():
        for _ in range(4):
            await asr._pace()

    asyncio.run(run())
    assert len(slept) == 1 and 59 < slept[0] <= 60.1  # the 4th call waits for the window to roll
    s.asr_requests_per_minute = 0
    asyncio.run(asr._pace())  # 0 = no pacing, returns at once


def test_very_short_lines_skip_the_self_check(monkeypatch):
    from backend.agent import voice
    from backend.schemas import Segment, Turn

    calls = []

    async def counting_transcribe(wav, language=None):
        calls.append(1)
        return "plants make their own food from sunlight"

    monkeypatch.setattr(voice.asr, "transcribe", counting_transcribe)
    monkeypatch.setattr(voice.asr, "asr_available", lambda: True)

    class Ctx:
        asr_checked, asr_bad, asr_off = 0, 0, False

        def emit(self, *a, **kw):
            pass

    seg = Segment(id="s", index=0, title="t", goal="g")
    short, long = Turn(speaker="host", text="No way!"), Turn(speaker="expert", text="Plants make their own food from sunlight.")
    asyncio.run(voice._synth_turn(Ctx(), seg, 0, short))
    assert calls == [] and short.wer is None
    asyncio.run(voice._synth_turn(Ctx(), seg, 1, long))
    assert calls == [1] and long.wer == 0.0


def test_asr_without_a_key_never_calls_the_service(monkeypatch):
    from backend.clients import asr

    s = settings(tts_api_key="sk_real", asr_api_key="")
    monkeypatch.setattr(asr, "get_settings", lambda: s)

    async def no_http(*a, **kw):
        raise AssertionError("unauthenticated ASR request")

    monkeypatch.setattr(asr, "request_with_retry", no_http)
    assert asr.asr_available() is False  # the self-check is skipped in the pipeline
    with pytest.raises(ServiceError, match="not configured"):
        asyncio.run(asr.transcribe(b"RIFF"))
