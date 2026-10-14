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
