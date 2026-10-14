"""Synthesis + ASR self-check (the agent observing its own output).

Each turn is normalised for speech, synthesised in the speaker's voice, then
transcribed by ASR. If the transcript's WER against the text we asked for is
above threshold (garbled, skipped or hallucinated audio), the turn is
re-synthesised and the best take is kept. This closes the
Perceive -> Act -> Observe loop on the audio itself.
"""
from __future__ import annotations

import asyncio

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


async def _synth_turn(ctx: AgentContext, seg: Segment, i: int, turn: Turn) -> np.ndarray:
    s = get_settings()
    sr = s.sample_rate
    turn.tts_text = normalize_for_tts(turn.text)
    best: tuple[float, np.ndarray, str] | None = None
    check = asr.asr_available()
    for attempt in range(1, s.max_tts_retries + 2):
        turn.tts_attempts = attempt
        raw = await tts.synthesize(turn.tts_text, turn.speaker, style=STYLE.get(turn.speaker))
        clip = prepare_clip(raw, sr)
        if len(clip) < sr * 0.2:
            ctx.emit(STAGE, "warning", f"Turn {i + 1}: voice service returned (near) silent audio; retrying.", seg.id)
            continue
        if not check:
            return clip
        try:
            heard = await asr.transcribe(to_wav_bytes(resample(clip, sr, 16000), 16000))
        except ServiceError as e:
            ctx.emit(STAGE, "warning", f"Self-check skipped for turn {i + 1}: speech recogniser {e.user_message}.", seg.id)
            return clip
        err = wer(turn.tts_text, heard)
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
    return best[1]


async def synthesize_version(ctx: AgentContext, seg: Segment, version: SegmentVersion, out_path) -> None:
    """Synthesise all turns of ``version`` (in parallel, capped by the TTS semaphore) and write the segment WAV."""
    sr = get_settings().sample_rate
    clips = await asyncio.gather(*(_synth_turn(ctx, seg, i, t) for i, t in enumerate(version.turns)))
    res = stitch_turns(list(clips), [t.text for t in version.turns], sr, seed=hash(seg.id) & 0xFFFF)
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
