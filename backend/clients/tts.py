"""Text-to-speech client.

Two wire formats are supported, selected by TTS_API_STYLE:

* ``openai``  - POST {base}/v1/audio/speech {model, input, voice, response_format:"wav"}
                -> raw audio bytes. Used by OpenAI and most self-hosted servers.
* ``json``    - POST {base}/synthesize {text, speaker, language, style}
                -> {"audio": <base64 wav/mp3>} or raw audio bytes. Matches the thin
                wrapper in scripts/robin_tts_server.py around the in-house Robin model.

Returned bytes may be WAV or MP3; audio/stitch.py decodes either.
"""
from __future__ import annotations

import asyncio
import base64

from ..config import get_settings
from .base import ServiceError, request_with_retry
from .usage import record

_sem: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(get_settings().tts_max_concurrency)
    return _sem


def voice_for(speaker: str) -> str:
    s = get_settings()
    return s.tts_voice_host if speaker == "host" else s.tts_voice_expert


async def synthesize(text: str, speaker: str, *, style: str | None = None) -> bytes:
    s = get_settings()
    record(tts_calls=1, tts_chars=len(text))
    if s.effective_mock():
        from . import mock

        return await mock.mock_tts(text, speaker)

    headers = {}
    if s.tts_api_key:
        headers["Authorization"] = f"Bearer {s.tts_api_key}"
    base = s.tts_base_url.rstrip("/")
    if s.tts_api_style == "openai":
        body = {"model": s.tts_model, "input": text, "voice": voice_for(speaker), "response_format": "wav"}
        resp = await request_with_retry("Voice service", "POST", base + "/v1/audio/speech", json=body, headers=headers,
                                        timeout=s.tts_timeout_s, semaphore=_semaphore())
        return resp.content
    body = {"text": text, "speaker": voice_for(speaker), "language": s.tts_language}
    if style:
        body["style"] = style
    resp = await request_with_retry("Voice service", "POST", base + "/synthesize", json=body, headers=headers,
                                    timeout=s.tts_timeout_s, semaphore=_semaphore())
    if resp.headers.get("content-type", "").startswith("application/json"):
        try:
            return base64.b64decode(resp.json()["audio"])
        except (KeyError, ValueError) as e:
            raise ServiceError("Voice service", "returned an unexpected response", str(e)) from e
    return resp.content
