"""Contract test: run the REAL provider clients (MOCK_MODE off) against local fakes
of Gemini, Sarvam and Bodhan (scripts/fake_providers.py), end to end.

This checks request/response wiring, auth headers, retries on 429, friendly
errors on bad keys, and the full upload -> episode -> regenerate flow, all with
zero API credits. Real-API behaviour (voices, quality, rate limits) is checked
separately by scripts/check_apis.py once keys are in the secrets file.

Run:  python scripts/contract_test.py
"""
from __future__ import annotations

import asyncio
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


PORT = free_port()
BASE = f"http://127.0.0.1:{PORT}"
os.environ.update({
    "MOCK_MODE": "0", "DATA_DIR": tempfile.mkdtemp(prefix="podcast_contract_"),
    "PODCAST_SECRETS_FILE": os.path.join(tempfile.mkdtemp(), "none.env"),  # never touch the real keys
    "LLM_PROVIDER": "gemini", "LLM_BASE_URL": BASE, "LLM_API_KEY": "test-gemini-key", "LLM_MODEL": "gemini-flash-latest",
    "TTS_API_STYLE": "sarvam", "TTS_BASE_URL": BASE, "TTS_API_KEY": "test-sarvam-key",
    "ASR_BASE_URL": BASE, "ASR_API_KEY": "test-bodhan-key", "ASR_AUTH_PREFIX": "", "ASR_MODEL": "indic-transcribe",
    "OCR_MODE": "bodhan", "OCR_BASE_URL": BASE, "OCR_API_KEY": "test-bodhan-key",
})

import uvicorn
from fastapi.testclient import TestClient

from backend.clients import llm, ocr, tts
from backend.clients.base import ServiceError
from backend.config import get_settings
from backend.main import app
from scripts import fake_providers


def start_fake() -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(fake_providers.app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            return server
        time.sleep(0.05)
    raise SystemExit("fake provider server did not start")


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def wait(client: TestClient, eid: str, cond, timeout: float = 180) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        ep = client.get(f"/api/episodes/{eid}").json()
        if cond(ep):
            return ep
        time.sleep(0.3)
    raise SystemExit(f"timeout: {ep['status']} {ep['message']}")


async def error_paths() -> None:
    s = get_settings()
    print("error handling:")
    # 429 with Google RetryInfo -> retried and succeeds
    s.llm_api_key = "RATELIMIT"
    t0 = time.time()
    out = await llm.chat([{"role": "user", "content": "List every factual claim in: hi"}], task="x")
    check(bool(out) and time.time() - t0 >= 0.9, "Gemini 429 + retryDelay is honoured, then succeeds")
    # bad key -> friendly, non-retried error
    s.llm_api_key = "BADKEY"
    try:
        await llm.chat([{"role": "user", "content": "hi"}], task="x")
        check(False, "bad Gemini key should raise")
    except ServiceError as e:
        check("API key" in e.user_message, f"bad Gemini key -> '{e.service} {e.user_message}'")
    s.llm_api_key = "test-gemini-key"
    s.tts_api_key = "BADKEY"
    try:
        await tts.synthesize("Hello there.", "host")
        check(False, "bad Sarvam key should raise")
    except ServiceError as e:
        check("403" in e.user_message, f"bad Sarvam key -> '{e.service} {e.user_message}'")
    s.tts_api_key = "test-sarvam-key"
    # long turn split into <= TTS_MAX_CHARS pieces
    fake_providers.reset()
    long = " ".join(["This sentence is about forty characters long."] * 40)
    wav = await tts.synthesize(long, "expert")
    pieces = [r for r in fake_providers.LOG if r["api"] == "sarvam"]
    check(len(pieces) >= 2 and all(p["chars"] <= s.tts_max_chars for p in pieces) and len(wav) > 1000,
          f"{len(long)}-char turn split into {len(pieces)} Sarvam requests and re-joined")


def main() -> None:
    start_fake()
    s = get_settings()
    check(not s.effective_mock(), "running with REAL clients (mock mode off)")
    asyncio.run(error_paths())
    text = asyncio.run(ocr.ocr_page(b"\x89PNG\r\n\x1a\nfake"))
    check("photosynthesis" in text and fake_providers.LOG[-1]["api"] == "bodhan-ocr", "Bodhan Indic-OCR wire format (scanned pages)")
    fake_providers.reset()

    # Bodhan indic-speak TTS (alternative voice provider): wire format, both voices, pacing hook
    s.tts_api_style, s.tts_model, s.tts_api_key = "bodhan", "indic-speak", "test-bodhan-key"
    s.tts_voice_host, s.tts_voice_expert = "Deepika", "Amit"
    wavs = [asyncio.run(tts.synthesize("Plants make their own food.", sp)) for sp in ("host", "expert")]
    calls = [r for r in fake_providers.LOG if r["api"] == "bodhan-tts"]
    check(len(calls) == 2 and {c["speaker"] for c in calls} == {"Deepika", "Amit"} and all(c["lang"] == "en" for c in calls)
          and all(len(w) > 1000 for w in wavs), "Bodhan indic-speak TTS wire format (raw key, JSON-string instructions, two voices)")
    s.tts_api_style, s.tts_model, s.tts_api_key = "sarvam", "bulbul:v4-flash", "test-sarvam-key"
    s.tts_voice_host, s.tts_voice_expert = "simran_en_conversation", "dev_en_conversational"
    fake_providers.reset()

    print("end-to-end (Gemini + Sarvam + Bodhan wire formats):")
    with TestClient(app) as c:
        check(c.get("/api/health").json()["mock"] is False, "/api/health reports mock=false")
        src = c.post("/api/sources", data={"sample_id": "chem-reactions"}).json()
        eid = c.post("/api/episodes", json={"source_id": src["id"], "settings": {"length": "short", "mode": "agent"}}).json()["id"]
        ep = wait(c, eid, lambda e: e["status"] in ("done", "failed") and not e["busy"])
        check(ep["status"] == "done", f"episode finished: {ep['status']} {ep.get('error') or ''}")
        check(all(x["status"] == "done" for x in ep["segments"]), "every segment done")
        apis = {r["api"] for r in fake_providers.LOG}
        check(apis == {"gemini", "sarvam", "bodhan"}, f"all three APIs used: {sorted(apis)}")
        g = [r for r in fake_providers.LOG if r["api"] == "gemini"]
        check({"planner_step", "write_episode", "critic_all", "verify_episode"} <= {r["task"] for r in g},
              f"Gemini handled planner, writer, critic and claim checks: {sorted({r['task'] for r in g})}")
        check(all(r["json"] for r in g if r["task"] in ("planner_step", "write_episode", "critic_all")), "JSON mode requested for structured calls")
        voices = {r["speaker"] for r in fake_providers.LOG if r["api"] == "sarvam"}
        check(voices == {s.tts_voice_host, s.tts_voice_expert}, f"two distinct Sarvam voices: {sorted(voices)}")
        m = ep["metrics"]
        check(m["prompt_tokens"] > 0 and m["asr_calls"] > 0, f"usage tracked: {m['llm_calls']} LLM calls, "
              f"{m['prompt_tokens'] + m['completion_tokens']} tokens, {m['tts_calls']} TTS, {m['asr_calls']} ASR")
        audio = c.get(f"/api/episodes/{eid}/audio")
        check(audio.status_code == 200 and len(audio.content) > 50000, f"episode audio served ({len(audio.content) // 1024} KB)")

        seg = ep["segments"][1]["id"]
        fake_providers.reset()
        c.post(f"/api/episodes/{eid}/segments/{seg}/regenerate", json={"instruction": "Add an everyday example"})
        ep2 = wait(c, eid, lambda e: not e["busy"] and e["segments"][1]["active_version"] == 2)
        n_tts = sum(1 for r in fake_providers.LOG if r["api"] == "sarvam")
        turns2 = len(ep2["segments"][1]["versions"][-1]["turns"])
        check(n_tts <= turns2 * (s.max_tts_retries + 1), f"regeneration only re-voiced that segment ({n_tts} TTS calls for {turns2} turns)")
    print("CONTRACT TEST PASSED")


if __name__ == "__main__":
    main()
