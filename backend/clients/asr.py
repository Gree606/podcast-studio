"""Speech recognition client, used only for the agent's self-check of its own audio.

Wire format: OpenAI-style multipart POST {base}/v1/audio/transcriptions with
``file`` (+ ``model``, optional ``language``) returning {"text": ...}. This is the Bodhan API
(model "indic-transcribe", ``Authorization: <key>`` with no "Bearer" prefix) and also
the OpenAI/Canary-NIM shape (set ASR_AUTH_PREFIX="Bearer ").
"""
from __future__ import annotations

import asyncio
import time
from collections import deque

from ..config import get_settings
from .base import ServiceError, request_with_retry
from .usage import record

_sem: asyncio.Semaphore | None = None
_sent: deque[float] = deque()  # monotonic times of recent real ASR calls
_pace_lock: asyncio.Lock | None = None
_pace_loop: asyncio.AbstractEventLoop | None = None


async def _pace() -> None:
    """Wait until another call fits in the service's per-minute budget (sliding 60 s window)."""
    global _pace_lock, _pace_loop
    limit = get_settings().asr_requests_per_minute
    if limit <= 0:
        return
    loop = asyncio.get_running_loop()
    if _pace_lock is None or _pace_loop is not loop:
        _pace_lock, _pace_loop = asyncio.Lock(), loop
    async with _pace_lock:
        while True:
            now = time.monotonic()
            while _sent and now - _sent[0] >= 60:
                _sent.popleft()
            if len(_sent) < limit:
                _sent.append(now)
                return
            await asyncio.sleep(60 - (now - _sent[0]) + 0.05)


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(get_settings().asr_max_concurrency)
    return _sem


def asr_available() -> bool:
    s = get_settings()
    return s.asr_mock() or s.asr_ready()


async def transcribe(wav_bytes: bytes) -> str:
    """Transcribe 16 kHz mono WAV bytes."""
    s = get_settings()
    record(asr_calls=1)
    if s.asr_mock():
        from . import mock

        return await mock.mock_asr(wav_bytes)
    await _pace()
    headers = {"Authorization": f"{s.asr_auth_prefix}{s.asr_api_key}"}
    data = {}
    if s.asr_model:
        data["model"] = s.asr_model
    if s.asr_language_code():
        data["language"] = s.asr_language_code()
    resp = await request_with_retry(
        "Speech recogniser", "POST", s.asr_base_url.rstrip("/") + "/v1/audio/transcriptions",
        files={"file": ("turn.wav", wav_bytes, "audio/wav")}, data=data, headers=headers,
        timeout=s.asr_timeout_s, attempts=s.asr_attempts, semaphore=_semaphore(),
    )
    try:
        return resp.json()["text"]
    except (KeyError, ValueError) as e:
        raise ServiceError("Speech recogniser", "returned an unexpected response", str(e)) from e
