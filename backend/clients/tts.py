"""Text-to-speech client. Wire format chosen by TTS_API_STYLE:

* ``sarvam``: Sarvam AI, POST {base}/text-to-speech with an ``api-subscription-key`` header.
  Body {text, language_code, speaker, model, pace, speech_sample_rate} returns
  {"audios": [<base64 WAV>]}. Shape taken from the official ``sarvamai`` SDK v0.1.37.
  Note the field is ``language_code``, not ``target_language_code``.
* ``bodhan``: Bodhan indic-speak, POST {base}/v1/audio/speech with ``Authorization: <key>`` (no
  "Bearer") and body {model: "indic-speak", input, voice, instructions: '{"lang": "en"}'};
  returns raw audio bytes (24 kHz). Rate-limited, so calls are paced (TTS_REQUESTS_PER_MINUTE).
* ``openai``: POST {base}/v1/audio/speech {model, input, voice, response_format} returns raw audio.
* ``json``: POST {base}/synthesize {text, speaker, language} returns {"audio": base64} or raw audio.

Turns longer than TTS_MAX_CHARS are split at sentence boundaries and the pieces joined.
Returned bytes may be WAV or MP3; audio/stitch.py decodes either.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import time

import numpy as np

from .. import languages
from ..config import get_settings
from .base import ServiceError, request_with_retry
from .usage import record

_sem: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(get_settings().tts_max_concurrency)
    return _sem


_next_slot = 0.0  # monotonic time at which the next paced TTS call may start
_pace_lock: asyncio.Lock | None = None
_pace_loop: asyncio.AbstractEventLoop | None = None


async def _pace() -> None:
    """Space calls evenly at the provider's per-minute rate (0 = unlimited). A sliding 60 s window
    let the whole minute's budget go out at once, and Bodhan refuses bursts (HTTP 429 after about
    7 quick calls), so half the calls failed and their back-offs grew. Even spacing has the same
    average rate and almost no refusals. Runs before retries too (see request_with_retry)."""
    global _pace_lock, _pace_loop, _next_slot
    limit = get_settings().tts_requests_per_minute
    if limit <= 0:
        return
    loop = asyncio.get_running_loop()
    if _pace_lock is None or _pace_loop is not loop:
        _pace_lock, _pace_loop = asyncio.Lock(), loop
    async with _pace_lock:
        now = time.monotonic()
        wait = _next_slot - now
        _next_slot = max(now, _next_slot) + 60.0 / limit
    if wait > 0:
        await asyncio.sleep(wait)


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
    for sent in re.split(r"(?<=[.!?\u0964\u0965])\s+", text):  # also the Devanagari full stops । ॥
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


async def _sarvam(text: str, speaker: str, lang: str | None = None) -> bytes:
    s = get_settings()
    body = {"text": text, "language_code": lang or s.tts_language, "speaker": voice_for(speaker), "model": s.tts_model,
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


def bodhan_body(text: str, speaker: str, lang_code: str | None = None) -> dict:
    s = get_settings()
    lang = (lang_code or s.tts_language).split("-")[0].lower() or "en"
    return {"model": s.tts_model, "input": text, "voice": voice_for(speaker), "instructions": json.dumps({"lang": lang})}


async def _bodhan(text: str, speaker: str, lang: str | None = None) -> bytes:
    s = get_settings()
    resp = await request_with_retry(
        "Voice service", "POST", s.tts_base_url.rstrip("/") + "/v1/audio/speech", json=bodhan_body(text, speaker, lang),
        headers={"Authorization": f"{s.tts_auth_prefix}{s.tts_api_key}", "Content-Type": "application/json"},
        timeout=s.tts_timeout_s, attempts=5, semaphore=_semaphore(), pace=_pace,
    )
    if not resp.content or resp.headers.get("content-type", "").startswith("application/json"):
        raise ServiceError("Voice service", "returned no audio", resp.text[:200])
    return resp.content


async def _openai(text: str, speaker: str) -> bytes:
    s = get_settings()
    headers = {"Authorization": f"Bearer {s.tts_api_key}"} if s.tts_api_key else {}
    body = {"model": s.tts_model, "input": text, "voice": voice_for(speaker), "response_format": "wav"}
    resp = await request_with_retry("Voice service", "POST", s.tts_base_url.rstrip("/") + "/v1/audio/speech", json=body,
                                    headers=headers, timeout=s.tts_timeout_s, semaphore=_semaphore())
    return resp.content


async def _json(text: str, speaker: str, style: str | None, lang: str | None = None) -> bytes:
    s = get_settings()
    headers = {"Authorization": f"Bearer {s.tts_api_key}"} if s.tts_api_key else {}
    body = {"text": text, "speaker": voice_for(speaker), "language": lang or s.tts_language}
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


async def synthesize(text: str, speaker: str, *, style: str | None = None, language: str | None = None) -> bytes:
    """``language`` is an episode language code (backend.languages); None = the configured TTS_LANGUAGE."""
    s = get_settings()
    lang = languages.get(language).tts if language else None
    record(tts_calls=1, tts_chars=len(text))
    if s.tts_mock():
        from . import mock

        return await mock.mock_tts(text, speaker)
    pieces = split_text(text, s.tts_max_chars)
    if s.tts_api_style == "sarvam":
        outs = await asyncio.gather(*(_sarvam(p, speaker, lang) for p in pieces))
    elif s.tts_api_style == "bodhan":
        outs = await asyncio.gather(*(_bodhan(p, speaker, lang) for p in pieces))
    elif s.tts_api_style == "openai":
        outs = await asyncio.gather(*(_openai(p, speaker) for p in pieces))
    else:
        outs = await asyncio.gather(*(_json(p, speaker, style, lang) for p in pieces))
    return outs[0] if len(outs) == 1 else _join_wavs(list(outs))
