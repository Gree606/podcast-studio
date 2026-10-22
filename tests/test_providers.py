"""Unit tests for provider adapters (Gemini translation, Sarvam text splitting, config, retries)."""
import httpx
import pytest

from backend.agent.voice import script_mismatch
from backend.clients.base import ServiceError, retry_delay
from backend.clients.llm import from_gemini, to_gemini
from backend.clients.tts import split_text
from backend.config import is_set


def test_to_gemini_maps_roles_system_and_images():
    msgs = [
        {"role": "system", "content": "be strict"},
        {"role": "user", "content": "q1"},
        {"role": "user", "content": "q1b"},  # merged with previous user turn
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                                     {"type": "text", "text": "ocr this"}]},
    ]
    system, contents = to_gemini(msgs)
    assert system == {"parts": [{"text": "be strict"}]}
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[0]["parts"] == [{"text": "q1"}, {"text": "q1b"}]
    assert contents[2]["parts"][0] == {"inline_data": {"mime_type": "image/png", "data": "QUJD"}}


def test_from_gemini_skips_thoughts_and_counts_usage():
    data = {"candidates": [{"content": {"parts": [{"text": "hmm", "thought": True}, {"text": '{"a": 1}'}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 20}}
    text, usage = from_gemini(data)
    assert text == '{"a": 1}'
    assert usage == {"prompt_tokens": 10, "completion_tokens": 25}


def test_from_gemini_errors_are_friendly():
    with pytest.raises(ServiceError, match="declined"):
        from_gemini({"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(ServiceError, match="empty answer.*MAX_TOKENS"):
        from_gemini({"candidates": [{"content": {"parts": [{"text": "x", "thought": True}]}, "finishReason": "MAX_TOKENS"}]})


def test_split_text_respects_limit_and_keeps_all_words():
    text = "Short one. " + "A much longer sentence that goes on, and on, and on for quite a while. " * 30
    parts = split_text(text, 200)
    assert all(len(p) <= 200 for p in parts) and len(parts) > 1
    assert " ".join(parts).split() == text.split()
    assert split_text("tiny", 200) == ["tiny"]


@pytest.mark.parametrize("value,expected", [
    ("", False), ("DUMMY_REPLACE_WITH_GEMINI_KEY", False), ("your-key-here", False), ("<key>", False),
    ("AIzaSyRealLookingKey123", True), ("sk_abc", True),
])
def test_is_set_detects_placeholders(value, expected):
    assert is_set(value) is expected


def test_script_mismatch():
    assert script_mismatch("Plants make food", "पौधे भोजन बनाते हैं")
    assert not script_mismatch("Plants make food", "plants make food")
    assert not script_mismatch("Plants make food", "")


def test_retry_delay_parses_header_and_google_retryinfo():
    assert retry_delay(httpx.Response(429, headers={"retry-after": "7"})) == 7.0
    body = '{"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "13s"}]}}'
    assert retry_delay(httpx.Response(429, text=body)) == 13.0
    assert retry_delay(httpx.Response(429, text="{}")) is None


def test_no_fake_ocr_text_outside_mock_mode(monkeypatch):
    """A real deployment without OCR must skip scanned pages, never insert mock OCR text."""
    from backend.clients import ocr
    from backend.config import Settings

    s = Settings(_env_file=None, mock_mode=False, llm_api_key="", ocr_mode="llm")
    monkeypatch.setattr(ocr, "get_settings", lambda: s)
    assert ocr.ocr_available() is False
    s2 = Settings(_env_file=None, mock_mode=True, ocr_mode="llm")
    monkeypatch.setattr(ocr, "get_settings", lambda: s2)
    assert ocr.ocr_available() is True


def test_llm_falls_back_to_next_model_when_overloaded(monkeypatch):
    """503 on the main model -> next model answers; a bad key (401) is never retried on other models."""
    import asyncio

    from backend.clients import llm

    tried = []

    async def fake_call(messages, *, model, attempts, **kw):
        tried.append(model)
        if model == "busy-model":
            raise ServiceError("Language model", "returned HTTP 503", status=503)
        if model == "bad-key-model":
            raise ServiceError("Language model", "rejected the API key", status=401)
        return "ok", {"prompt_tokens": 1, "completion_tokens": 1}

    text, _, used = asyncio.run(llm._call_with_fallback(fake_call, [], models=["busy-model", "good-model"], temperature=0,
                                                        max_tokens=10, json_mode=False, base_url="", api_key=""))
    assert (text, used, tried) == ("ok", "good-model", ["busy-model", "good-model"])
    tried.clear()
    with pytest.raises(ServiceError, match="API key"):
        asyncio.run(llm._call_with_fallback(fake_call, [], models=["bad-key-model", "good-model"], temperature=0,
                                            max_tokens=10, json_mode=False, base_url="", api_key=""))
    assert tried == ["bad-key-model"]
    assert llm.model_chain("judge-model", explicit=True) == ["judge-model"]


def test_exhausted_daily_quota_is_skipped_and_reported(monkeypatch):
    """A 429 with a retry delay of hours marks the model exhausted; when all are, the user gets a clear message."""
    import asyncio

    from backend.clients import llm

    monkeypatch.setattr(llm, "_exhausted", {})
    tried = []

    async def fake_call(messages, *, model, attempts, **kw):
        tried.append(model)
        if model in ("capped-a", "capped-b"):
            e = ServiceError("Language model", "is rate-limiting us", status=429)
            e.retry_after = 45000
            raise e
        return "ok", {"prompt_tokens": 1, "completion_tokens": 1}

    kw = dict(temperature=0, max_tokens=10, json_mode=False, base_url="", api_key="")
    assert asyncio.run(llm._call_with_fallback(fake_call, [], models=["capped-a", "lite"], **kw))[2] == "lite"
    tried.clear()
    assert asyncio.run(llm._call_with_fallback(fake_call, [], models=["capped-a", "lite"], **kw))[2] == "lite"
    assert tried == ["lite"]  # the exhausted model is not even tried again
    with pytest.raises(ServiceError, match="free daily quota"):
        asyncio.run(llm._call_with_fallback(fake_call, [], models=["capped-a", "capped-b"], **kw))


def test_bodhan_tts_body_and_readiness():
    """Bodhan indic-speak wants instructions as a JSON *string* with the language, and needs a real key."""
    import json as _json

    from backend.clients import tts
    from backend.config import Settings

    s = Settings(_env_file=None, tts_api_style="bodhan", tts_model="indic-speak", tts_voice_host="Deepika",
                 tts_voice_expert="Amit", tts_language="en-IN", tts_api_key="DUMMY_REPLACE_ME")
    import backend.clients.tts as tts_mod
    orig = tts_mod.get_settings
    tts_mod.get_settings = lambda: s
    try:
        body = tts.bodhan_body("Hello there.", "expert")
    finally:
        tts_mod.get_settings = orig
    assert body["model"] == "indic-speak" and body["voice"] == "Amit" and body["input"] == "Hello there."
    assert isinstance(body["instructions"], str) and _json.loads(body["instructions"]) == {"lang": "en"}
    assert s.tts_ready() is False  # placeholder key -> not ready
    s.tts_api_key = "real-looking-key"
    assert s.tts_ready() is True
