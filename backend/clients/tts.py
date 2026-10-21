"""Text-to-speech client. Wire format chosen by TTS_API_STYLE:

* ``sarvam``: Sarvam AI, POST {base}/text-to-speech with an ``api-subscription-key`` header.
  Body {text, language_code, speaker, model, pace, speech_sample_rate} returns
  {"audios": [<base64 WAV>]}. Shape taken from the official ``sarvamai`` SDK v0.1.37.
  Note the field is ``language_code``, not ``target_language_code``.
* ``openai``: POST {base}/v1/audio/speech {model, input, voice, response_format} returns raw audio.
* ``json``: POST {base}/synthesize {text, speaker, language} returns {"audio": base64} or raw audio.

Turns longer than TTS_MAX_CHARS are split at sentence boundaries and the pieces joined.
Returned bytes may be WAV or MP3; audio/stitch.py decodes either.
"""
from __future__ import annotations

import asyncio
import base64
import re

import numpy as np

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


def split_text(text: str, max_chars: int) -> list[str]:
    """Split at sentence (then comma) boundaries into pieces of at most ``max_chars``."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    cur = ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        while len(sent) > max_chars:  # a single huge sentence: break at a comma or space
            cut = max(sent.rfind(", ", 0, max_chars), sent.rfind(" ", 0, max_chars))
            cut = cut if cut > 0 else max_chars
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(sent[:cut].strip())
            sent = sent[cut:].strip(" ,")
        if cur and len(cur) + 1 + len(sent) > max_chars:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        pieces.append(cur)
    return [p for p in pieces if p]


async def _sarvam(text: str, speaker: str) -> bytes:
    s = get_settings()
    body = {"text": text, "language_code": s.tts_language, "speaker": voice_for(speaker), "model": s.tts_model,
            "pace": s.tts_pace, "speech_sample_rate": s.sample_rate}
    resp = await request_with_retry(
        "Voice service", "POST", s.tts_base_url.rstrip("/") + "/text-to-speech", json=body,
        headers={"api-subscription-key": s.tts_api_key, "Content-Type": "application/json"},
        timeout=s.tts_timeout_s, attempts=4, semaphore=_semaphore(),
    )
    try:
        audios = [base64.b64decode(a) for a in resp.json()["audios"]]
        if not audios:
            raise ValueError("empty 'audios' list")
        return audios[0] if len(audios) == 1 else _join_wavs(audios)
    except (KeyError, ValueError, TypeError) as e:
        raise ServiceError("Voice service", "returned an unexpected response", str(e)) from e


async def _openai(text: str, speaker: str) -> bytes:
    s = get_settings()
    headers = {"Authorization": f"Bearer {s.tts_api_key}"} if s.tts_api_key else {}
    body = {"model": s.tts_model, "input": text, "voice": voice_for(speaker), "response_format": "wav"}
    resp = await request_with_retry("Voice service", "POST", s.tts_base_url.rstrip("/") + "/v1/audio/speech", json=body,
                                    headers=headers, timeout=s.tts_timeout_s, semaphore=_semaphore())
    return resp.content


async def _json(text: str, speaker: str, style: str | None) -> bytes:
    s = get_settings()
    headers = {"Authorization": f"Bearer {s.tts_api_key}"} if s.tts_api_key else {}
    body = {"text": text, "speaker": voice_for(speaker), "language": s.tts_language}
    if style:
        body["style"] = style
    resp = await request_with_retry("Voice service", "POST", s.tts_base_url.rstrip("/") + "/synthesize", json=body,
                                    headers=headers, timeout=s.tts_timeout_s, semaphore=_semaphore())
    if resp.headers.get("content-type", "").startswith("application/json"):
        try:
            return base64.b64decode(resp.json()["audio"])
        except (KeyError, ValueError) as e:
            raise ServiceError("Voice service", "returned an unexpected response", str(e)) from e
    return resp.content


def _join_wavs(chunks: list[bytes]) -> bytes:
    from ..audio.stitch import decode_audio, to_wav_bytes

    sr = get_settings().sample_rate
    gap = np.zeros(int(0.08 * sr), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, c in enumerate(chunks):
        if i:
            parts.append(gap)
        parts.append(decode_audio(c, sr))
    return to_wav_bytes(np.concatenate(parts), sr)


async def synthesize(text: str, speaker: str, *, style: str | None = None) -> bytes:
    s = get_settings()
    record(tts_calls=1, tts_chars=len(text))
    if s.tts_mock():
        from . import mock

        return await mock.mock_tts(text, speaker)
    pieces = split_text(text, s.tts_max_chars)
    if s.tts_api_style == "sarvam":
        outs = await asyncio.gather(*(_sarvam(p, speaker) for p in pieces))
    elif s.tts_api_style == "openai":
        outs = await asyncio.gather(*(_openai(p, speaker) for p in pieces))
    else:
        outs = await asyncio.gather(*(_json(p, speaker, style) for p in pieces))
    return outs[0] if len(outs) == 1 else _join_wavs(list(outs))
