"""Synthesis + ASR self-check (the agent observing its own output).

Each turn is normalised for speech, synthesised in the speaker's voice, then
transcribed by ASR. If the transcript's WER against the text we asked for is
above threshold (garbled, skipped or hallucinated audio), the turn is
re-synthesised and the best take is kept. This closes the
Perceive -> Act -> Observe loop on the audio itself.
"""
from __future__ import annotations

import asyncio
import zlib

import numpy as np

from ..audio.stitch import prepare_clip, resample, stitch_turns, to_wav_bytes
from ..audio.wer import wer
from ..clients import asr, tts
from ..clients.base import ServiceError
from ..config import get_settings
from ..schemas import Segment, SegmentVersion, Turn
from ..textnorm import normalize_for_tts
from .context import AgentContext

STAGE = "voice"
STYLE = {"host": "CONV", "expert": "CONV"}


def script_mismatch(expected: str, heard: str) -> bool:
    """True if we asked for Latin-script speech but the ASR replied mostly in another script
    (e.g. Devanagari). Comparing WER across scripts would trigger pointless re-synthesis."""
    letters = [c for c in heard if c.isalpha()]
    if not letters:
        return False
    non_latin = sum(1 for c in letters if ord(c) > 0x24F) / len(letters)
    exp_letters = [c for c in expected if c.isalpha()]
    exp_latin = sum(1 for c in exp_letters if ord(c) <= 0x24F) / max(1, len(exp_letters))
    return non_latin > 0.5 and exp_latin > 0.8


def _lang(ctx: AgentContext) -> str:
    ep = getattr(ctx, "episode", None)
    return getattr(getattr(ep, "settings", None), "language", None) or "en"


def _cache_key(ctx: AgentContext, turn: Turn) -> tuple[str, str]:
    """Recordings are shared by speaker and spoken (normalised) text."""
    return turn.speaker, normalize_for_tts(turn.text, _lang(ctx))


def prefetch(ctx: AgentContext, seg: Segment, turns: list[Turn]) -> None:
    """Start recording (TTS + listen-back) these lines now, in the background. A later
    synthesize_version reuses a finished take for any line whose text is unchanged; lines the
    fact-check rewrote simply miss the cache and are recorded fresh."""
    for i, t in enumerate(turns):
        key = _cache_key(ctx, t)
        if key not in ctx.tts_cache:
            task = asyncio.create_task(_record_line(ctx, seg, i, t.model_copy()))
            task.add_done_callback(lambda tk: tk.cancelled() or tk.exception())  # failures are retried later, not logged twice
            ctx.tts_cache[key] = task


async def _synth_turn(ctx: AgentContext, seg: Segment, i: int, turn: Turn) -> np.ndarray:
    task = getattr(ctx, "tts_cache", {}).pop(_cache_key(ctx, turn), None)  # (test doubles may not carry a cache)
    if task is not None:
        try:
            clip, done = await task
            turn.tts_text, turn.tts_attempts = done.tts_text, done.tts_attempts
            turn.wer, turn.first_wer, turn.asr_text = done.wer, done.first_wer, done.asr_text
            return clip
        except Exception:  # noqa: BLE001 - a failed early take is just recorded again below
            pass
    clip, _ = await _record_line(ctx, seg, i, turn)
    return clip


async def _record_line(ctx: AgentContext, seg: Segment, i: int, turn: Turn) -> tuple[np.ndarray, Turn]:
    s = get_settings()
    sr = s.sample_rate
    lang = _lang(ctx)
    turn.tts_text = normalize_for_tts(turn.text, lang)
    best: tuple[float, np.ndarray, str] | None = None
    check = asr.asr_available() and len(turn.tts_text.split()) >= s.asr_min_words
    for attempt in range(1, s.max_tts_retries + 2):
        turn.tts_attempts = attempt
        raw = await tts.synthesize(turn.tts_text, turn.speaker, style=STYLE.get(turn.speaker), language=lang)
        clip = prepare_clip(raw, sr)
        if len(clip) < sr * 0.2:
            ctx.emit(STAGE, "warning", f"Turn {i + 1}: voice service returned (near) silent audio; retrying.", seg.id)
            continue
        if not check:
            return clip, turn
        try:
            heard = await asr.transcribe(to_wav_bytes(resample(clip, sr, 16000), 16000), language=lang)
        except ServiceError as e:
            ctx.emit(STAGE, "warning", f"Self-check skipped for turn {i + 1}: speech recogniser {e.user_message}.", seg.id)
            return clip, turn
        if script_mismatch(turn.tts_text, heard):
            ctx.emit(STAGE, "warning", f"Turn {i + 1}: the speech recogniser answered in a different script "
                     f"(\"{heard[:40]}\"), so WER is meaningless; keeping this take. Set ASR_LANGUAGE to fix.", seg.id)
            return clip, turn
        err = wer(turn.tts_text, heard, lang)
        if err is None:  # e.g. an all-English line in a Hinglish episode: nothing to compare in this script
            return clip, turn
        if attempt == 1:
            turn.first_wer = round(err, 3)
        if best is None or err < best[0]:
            best = (err, clip, heard)
        if err <= s.wer_threshold:
            break
        ctx.emit(STAGE, "observation", f"Turn {i + 1} self-check: WER {err:.0%} > {s.wer_threshold:.0%} "
                 f"(heard: \"{heard[:80]}\"); re-synthesising.", seg.id)
    if best is None:
        raise ServiceError("Voice service", "returned no usable audio")
    turn.wer, turn.asr_text = round(best[0], 3), best[2]
    return best[1], turn


async def synthesize_version(ctx: AgentContext, seg: Segment, version: SegmentVersion, out_path) -> None:
    """Synthesise all turns of ``version`` (in parallel, capped by the TTS semaphore) and write the segment WAV."""
    sr = get_settings().sample_rate
    clips = await asyncio.gather(*(_synth_turn(ctx, seg, i, t) for i, t in enumerate(version.turns)))
    res = stitch_turns(list(clips), [t.text for t in version.turns], sr, seed=zlib.crc32(seg.id.encode()))
    for t, (a, b) in zip(version.turns, res.spans):
        t.seg_start, t.seg_end = round(a, 3), round(b, 3)
    out_path.write_bytes(to_wav_bytes(res.audio, sr))
    version.audio_file = out_path.name
    version.duration = round(res.duration, 2)
    wers = [t.wer for t in version.turns if t.wer is not None]
    retried = sum(1 for t in version.turns if t.tts_attempts > 1)
    if wers:
        ctx.emit(STAGE, "observation", f"Audio for '{seg.title}': {res.duration:.0f}s, mean self-check WER "
                 f"{sum(wers) / len(wers):.0%}, {retried} turn(s) re-synthesised.", seg.id)
