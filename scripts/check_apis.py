"""Check the real API keys in .env with one tiny call per service (costs well under a cent).

Run after putting keys in .env:   python scripts/check_apis.py
Checks:
  1. Gemini: a JSON-mode reply, plus how many "thinking" tokens it used
  2. Sarvam: one short clip per configured voice (also saved to data/api_check/ so you can listen)
  3. Bodhan: transcribes the Sarvam clip; reports WER and whether it answered in Latin script
Exit code 0 only if LLM and TTS work (ASR is optional for the app).
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.agent.voice import script_mismatch  # noqa: E402
from backend.audio.stitch import decode_audio, resample, to_wav_bytes  # noqa: E402
from backend.audio.wer import wer  # noqa: E402
from backend.clients import asr, llm, tts  # noqa: E402
from backend.clients.base import ServiceError  # noqa: E402
from backend.clients.usage import current_metrics  # noqa: E402
from backend.config import get_settings  # noqa: E402
from backend.schemas import Metrics  # noqa: E402

SAMPLE = "Hi, I'm your host. Today we're learning how plants make food using sunlight, water and carbon dioxide."


def line(ok: bool | None, msg: str) -> None:
    print({True: "  [ OK ]", False: "  [FAIL]", None: "  [SKIP]"}[ok], msg)


async def main() -> int:
    s = get_settings()
    print(f"Provider config: LLM={s.llm_provider}:{s.llm_model}  TTS={s.tts_api_style}:{s.tts_model}  ASR={s.asr_model}")
    if s.mock_mode:
        print("MOCK_MODE=1 is set; unset it to check real APIs.")
        return 1
    out = ROOT / "data" / "api_check"
    out.mkdir(parents=True, exist_ok=True)
    ok_llm = ok_tts = False
    m = Metrics()
    current_metrics.set(m)

    print("1. LLM")
    if not s.llm_ready():
        line(False, "LLM_API_KEY is missing or still a placeholder in .env")
    else:
        try:
            t0 = time.time()
            r = await llm.chat_json([{"role": "user", "content": 'Reply with JSON {"answer": <the capital of India>}'}],
                                    task="check", max_tokens=100)
            ok_llm = isinstance(r, dict) and "delhi" in str(r.get("answer", "")).lower()
            line(ok_llm, f"{s.llm_model} answered {r} in {time.time() - t0:.1f}s "
                         f"(prompt {m.prompt_tokens} tok, output+thinking {m.completion_tokens} tok)")
        except ServiceError as e:
            line(False, f"{e.service} {e.user_message}  {e.detail[:300]}")

    print("2. TTS")
    clip_path = None
    if not s.tts_ready():
        line(False, "TTS_API_KEY is missing or still a placeholder in .env")
    else:
        for role, voice in (("host", s.tts_voice_host), ("expert", s.tts_voice_expert)):
            try:
                t0 = time.time()
                wav = await tts.synthesize(SAMPLE, role)
                audio = decode_audio(wav, s.sample_rate)
                p = out / f"{role}_{voice}.wav"
                p.write_bytes(to_wav_bytes(audio, s.sample_rate))
                ok = len(audio) > s.sample_rate
                ok_tts = ok_tts or ok
                clip_path = clip_path or p
                line(ok, f"voice '{voice}' ({role}): {len(audio) / s.sample_rate:.1f}s of audio in {time.time() - t0:.1f}s -> {p.relative_to(ROOT)}")
            except ServiceError as e:
                line(False, f"voice '{voice}' ({role}): {e.user_message}  {e.detail[:300]}")

    print("3. ASR (self-check)")
    if not s.asr_ready():
        line(None, "ASR_API_KEY not set: the app will skip the audio self-check")
    elif clip_path is None:
        line(None, "no TTS clip to transcribe")
    else:
        try:
            audio = decode_audio(clip_path.read_bytes(), s.sample_rate)
            heard = await asr.transcribe(to_wav_bytes(resample(audio, s.sample_rate, 16000), 16000))
            if script_mismatch(SAMPLE, heard):
                line(False, f"ASR replied in a non-Latin script: \"{heard[:80]}\". Try ASR_LANGUAGE=en in .env.")
            else:
                e = wer(SAMPLE, heard)
                line(e < 0.35, f"heard \"{heard[:100]}\" (WER {e:.0%})")
        except ServiceError as e:
            line(False, f"{e.service} {e.user_message}  {e.detail[:300]}")

    print("\nReady for real episodes." if ok_llm and ok_tts else "\nFix the failures above, then re-run.")
    return 0 if ok_llm and ok_tts else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
