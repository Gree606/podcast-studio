"""Rate-limited voices: calls are spaced evenly (retries too), and a recogniser that can't follow
the voice stops triggering re-recordings."""
import asyncio
import time

import numpy as np

from backend.agent import voice
from backend.clients import tts
from backend.config import get_settings
from backend.schemas import Segment, Turn


def test_tts_calls_are_spaced_evenly_not_burst(monkeypatch):
    monkeypatch.setattr(get_settings(), "tts_requests_per_minute", 600)  # one call every 0.1 s
    monkeypatch.setattr(tts, "_next_slot", 0.0)
    monkeypatch.setattr(tts, "_pace_lock", None)

    async def main():
        starts = []

        async def call():
            await tts._pace()
            starts.append(time.monotonic())
        await asyncio.gather(*(call() for _ in range(5)))
        return sorted(starts)
    starts = asyncio.run(main())
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert min(gaps) >= 0.08, gaps  # no burst: each call waits for its slot


def test_retries_wait_for_the_pace_too(monkeypatch):
    from backend.clients import base
    paced = []

    class Resp:
        def __init__(self, code):
            self.status_code, self.text, self.headers = code, "", {}

    codes = iter([429, 200])

    class Client:
        async def request(self, *a, **kw):
            return Resp(next(codes))
    monkeypatch.setattr(base, "http_client", lambda: Client())
    monkeypatch.setattr(base, "retry_delay", lambda r: 0.0)

    async def pace():
        paced.append(1)
    r = asyncio.run(base.request_with_retry("Voice service", "POST", "http://x", timeout=1, attempts=3, pace=pace))
    assert r.status_code == 200 and len(paced) == 2  # the retry was paced as well


def test_self_check_stands_down_when_most_first_takes_are_garbled(monkeypatch):
    sr = get_settings().sample_rate
    tts_calls = []

    async def fake_tts(text, speaker, **kw):
        tts_calls.append(text)
        return b"wav"
    monkeypatch.setattr(voice.tts, "synthesize", fake_tts)
    monkeypatch.setattr(voice, "prepare_clip", lambda raw, sr_: np.ones(sr_, dtype=np.float32))
    monkeypatch.setattr(voice.asr, "asr_available", lambda: True)

    async def garbled(*a, **kw):
        return "nadu nadu nadu nadu nadu nadu"
    monkeypatch.setattr(voice.asr, "transcribe", garbled)
    monkeypatch.setattr(get_settings(), "max_tts_retries", 2)

    class Ctx:
        asr_checked, asr_bad, asr_off = 0, 0, False
        notes = []

        def emit(self, stage, kind, text, *a):
            self.notes.append((kind, text))
    ctx, seg = Ctx(), Segment(id="s", index=0, title="t", goal="g")
    for i in range(6):
        asyncio.run(voice._record_line(ctx, seg, i, Turn(speaker="expert", text=f"Line {i}: plants make their own food from sunlight.")))
    assert ctx.asr_off and any("Keeping first takes" in t for k, t in ctx.notes if k == "warning")
    # lines 1-3: 3 takes each; line 4 trips the switch after its first take; lines 5-6: one take, no check
    assert len(tts_calls) == 3 * 3 + 1 + 2
