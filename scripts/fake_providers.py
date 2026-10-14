"""Local stand-ins for the three real provider APIs, used by scripts/contract_test.py.

Each endpoint copies the wire format of the real service and REJECTS requests
whose headers or fields don't match it. That lets us verify the real (non-mock)
client code without spending credits:

* Gemini  POST /v1beta/models/{model}:generateContent  (X-goog-api-key)
* Sarvam  POST /text-to-speech                         (api-subscription-key)
* Bodhan  POST /v1/audio/transcriptions                (Authorization: <key>, model=indic-transcribe)

Content comes from backend/clients/mock.py, so outputs are placeholder-quality.
Special keys trigger failure modes: "BADKEY" returns 401, and the first call with
"RATELIMIT" returns 429 with a Google-style retryDelay.
"""
from __future__ import annotations

import base64
import io
import json
import threading

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.clients import mock

app = FastAPI()
LOG: list[dict] = []  # every accepted request, for assertions
_tts_durations: list[tuple[float, str]] = []
_lock = threading.Lock()
_ratelimited: set[str] = set()

TASK_MARKERS = [
    ("planning agent of a podcast studio", "planner_step"),
    ("checking a podcast outline", "fact_verify"),
    ("NOT supported by the textbook passages", "claim_fix"),
    ("Here is your previous draft", "revise"),
    ("You write scripts for", "write"),
    ("strict editor", "critic"),
    ("improving a podcast writer", "reflect"),
    ("List every factual claim", "claim_extract"),
    ("For each numbered claim", "claim_judge"),
    ("evaluating an educational podcast script", "judge"),
    ("Transcribe all text on this textbook page", "ocr"),
]


def _err(code: int, msg: str, status: str = "INVALID_ARGUMENT") -> JSONResponse:
    return JSONResponse(status_code=code, content={"error": {"code": code, "message": msg, "status": status}})


@app.post("/v1beta/models/{model_action}")
async def gemini(model_action: str, request: Request, x_goog_api_key: str | None = Header(None)):
    if not model_action.endswith(":generateContent"):
        return _err(404, "unknown method")
    if not x_goog_api_key or x_goog_api_key == "BADKEY":
        return _err(401 if x_goog_api_key else 403, "API key not valid", "UNAUTHENTICATED")
    if x_goog_api_key == "RATELIMIT" and "gemini" not in _ratelimited:
        _ratelimited.add("gemini")
        return JSONResponse(status_code=429, content={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1s"}]}})
    body = await request.json()
    contents = body.get("contents")
    if not isinstance(contents, list) or not contents:
        return _err(400, "contents is required")
    for c in contents:
        if c.get("role") not in ("user", "model") or not c.get("parts"):
            return _err(400, f"bad content entry: {str(c)[:100]}")
    if contents[0]["role"] != "user":
        return _err(400, "first content must be from user")
    gen = body.get("generationConfig", {})
    if not isinstance(gen.get("maxOutputTokens"), int):
        return _err(400, "maxOutputTokens must be an int")
    # rebuild OpenAI-style messages for the shared mock brain
    msgs = []
    if body.get("systemInstruction"):
        msgs.append({"role": "system", "content": "".join(p.get("text", "") for p in body["systemInstruction"]["parts"])})
    for c in contents:
        msgs.append({"role": "assistant" if c["role"] == "model" else "user",
                     "content": "".join(p.get("text", "") for p in c["parts"])})
    alltext = " ".join(m["content"] for m in msgs)
    has_image = any("inline_data" in p for c in contents for p in c["parts"])
    task = "ocr" if has_image else next((t for marker, t in TASK_MARKERS if marker in alltext), "other")
    text = await mock.mock_chat(msgs, task=task, temperature=gen.get("temperature", 0.7))
    LOG.append({"api": "gemini", "task": task, "json": gen.get("responseMimeType") == "application/json"})
    return {"candidates": [{"content": {"role": "model", "parts": [
        {"text": "thinking...", "thought": True}, {"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": len(alltext) // 4, "candidatesTokenCount": len(text) // 4, "thoughtsTokenCount": 7}}


@app.post("/text-to-speech")
async def sarvam(request: Request, api_subscription_key: str | None = Header(None)):
    if not api_subscription_key or api_subscription_key == "BADKEY":
        return JSONResponse(status_code=403, content={"error": {"message": "Invalid API key"}})
    body = await request.json()
    for field in ("text", "language_code", "speaker", "model"):
        if not body.get(field):
            return JSONResponse(status_code=422, content={"error": {"message": f"missing {field}"}})
    if "target_language_code" in body:
        return JSONResponse(status_code=422, content={"error": {"message": "unknown field target_language_code"}})
    if len(body["text"]) > 2500:
        return JSONResponse(status_code=422, content={"error": {"message": "text too long"}})
    if body["speaker"] != body["speaker"].lower():
        return JSONResponse(status_code=422, content={"error": {"message": "speaker must be lowercase"}})
    sr = int(body.get("speech_sample_rate", 22050))
    wav = await mock.mock_tts(body["text"], "host" if "simran" in body["speaker"] else "expert")
    audio, in_sr = sf.read(io.BytesIO(wav), dtype="float32")
    with _lock:
        _tts_durations.append((len(audio) / in_sr, body["text"]))
    buf = io.BytesIO()
    sf.write(buf, audio, in_sr if sr == in_sr else in_sr, format="WAV")
    LOG.append({"api": "sarvam", "speaker": body["speaker"], "model": body["model"], "chars": len(body["text"])})
    return {"request_id": "fake", "audios": [base64.b64encode(buf.getvalue()).decode()]}


@app.post("/v1/audio/transcriptions")
async def bodhan(file: UploadFile = File(...), model: str = Form(None), authorization: str | None = Header(None)):
    if not authorization or authorization.startswith("Bearer ") or authorization == "BADKEY":
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    if model != "indic-transcribe":
        return JSONResponse(status_code=422, content={"error": f"unknown model {model}"})
    data = await file.read()
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    dur = len(audio) / sr
    with _lock:
        # the app trims ~0.2 s of padding before sending; match on the closest duration
        best = min(_tts_durations, key=lambda d: abs(d[0] - 0.22 - dur), default=(0, ""))
    LOG.append({"api": "bodhan", "seconds": round(dur, 2)})
    return {"text": best[1]}


def reset() -> None:
    LOG.clear()
    _tts_durations.clear()
    _ratelimited.clear()


def dump() -> str:
    return json.dumps(LOG[-5:])


_ = np  # numpy is used indirectly via soundfile
