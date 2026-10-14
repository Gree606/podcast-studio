"""Speech recognition client, used only for the agent's self-check of its own audio.

Wire format: OpenAI-style multipart POST {base}/v1/audio/transcriptions with
``file`` + ``language`` (+ ``model``) -> {"text": ...}. This is the HTTP shape of
the Canary NIM; the Riva gRPC path would need a small adapter here.
"""
from __future__ import annotations

import asyncio

from ..config import get_settings
from .base import ServiceError, request_with_retry
from .usage import record

_sem: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(get_settings().asr_max_concurrency)
    return _sem


def asr_available() -> bool:
    s = get_settings()
    return s.effective_mock() or bool(s.asr_base_url)


async def transcribe(wav_bytes: bytes) -> str:
    """Transcribe 16 kHz mono WAV bytes."""
    s = get_settings()
    record(asr_calls=1)
    if s.effective_mock():
        from . import mock

        return await mock.mock_asr(wav_bytes)
    headers = {}
    if s.asr_api_key:
        headers["Authorization"] = f"Bearer {s.asr_api_key}"
    data = {"language": s.asr_language}
    if s.asr_model:
        data["model"] = s.asr_model
    resp = await request_with_retry(
        "Speech recogniser", "POST", s.asr_base_url.rstrip("/") + "/v1/audio/transcriptions",
        files={"file": ("turn.wav", wav_bytes, "audio/wav")}, data=data, headers=headers,
        timeout=s.asr_timeout_s, semaphore=_semaphore(),
    )
    try:
        return resp.json()["text"]
    except (KeyError, ValueError) as e:
        raise ServiceError("Speech recogniser", "returned an unexpected response", str(e)) from e
