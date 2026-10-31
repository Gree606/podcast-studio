"""Local stand-ins for the three real provider APIs, used by scripts/contract_test.py.

Each endpoint copies the wire format of the real service and REJECTS requests
whose headers or fields don't match it. That lets us verify the real (non-mock)
client code without spending credits:

* Gemini  POST /v1beta/models/{model}:generateContent  (X-goog-api-key)
* Sarvam  POST /text-to-speech                         (api-subscription-key)
* Bodhan  POST /v1/audio/transcriptions                (Authorization: <key>, model=indic-transcribe)
* Bodhan  POST /v1/audio/speech                        (Authorization: <key>, model=indic-speak, TTS)

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
from fastapi.responses import JSONResponse, Response

from backend.clients import mock

app = FastAPI()
LOG: list[dict] = []  # every accepted request, for assertions
_tts_durations: list[tuple[float, str]] = []
_lock = threading.Lock()
_ratelimited: set[str] = set()

TASK_MARKERS = [
    ("Write a complete episode of", "write_episode"),
    ("strict fact-checker for an educational podcast", "verify_episode"),
    ("podcast episode.\nThe listener asked", "rewrite_part"),
    ("planning agent of a podcast studio", "planner_step"),
    ("checking a podcast outline", "fact_verify"),
    ("NOT supported by the textbook passages", "claim_fix"),
    ("Here is your previous draft", "revise"),
    ("You write scripts for", "write"),
    ("strict editor", "critic"),
    ("improving a podcast writer", "reflect"),
    ("fact-checking one part of an educational podcast", "claim_check"),
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


@app.post("/v1/audio/speech")
async def bodhan_tts(request: Request, authorization: str | None = Header(None)):
    if not authorization or authorization.startswith("Bearer ") or authorization == "BADKEY":
        return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    body = await request.json()
    for field in ("model", "input", "voice", "instructions"):
        if not body.get(field):
            return JSONResponse(status_code=422, content={"detail": f"missing {field}"})
    if body["model"] != "indic-speak":
        return JSONResponse(status_code=422, content={"detail": f"unknown model {body['model']}"})
    try:
        lang = json.loads(body["instructions"]).get("lang")  # a JSON *string*, as in the docs example
    except (TypeError, ValueError):
        return JSONResponse(status_code=422, content={"detail": "instructions must be a JSON string"})
    if not lang:
        return JSONResponse(status_code=422, content={"detail": "instructions.lang missing"})
    wav = await mock.mock_tts(body["input"], "host" if body["voice"] != "Amit" else "expert")
    audio, in_sr = sf.read(io.BytesIO(wav), dtype="float32")
    with _lock:
        _tts_durations.append((len(audio) / in_sr, body["input"]))
    LOG.append({"api": "bodhan-tts", "speaker": body["voice"], "model": body["model"], "lang": lang, "chars": len(body["input"])})
    return Response(content=wav, media_type="audio/mpeg")  # the real service labels its audio audio/mpeg


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


@app.post("/v1/chat/completions")
async def bodhan_ocr(request: Request, authorization: str | None = Header(None)):
    """Bodhan Indic-OCR: Bearer auth, model indic-ocr, exactly one user message holding one image data URI."""
    if not authorization or not authorization.startswith("Bearer ") or authorization == "Bearer BADKEY":
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    body = await request.json()
    msgs = body.get("messages") or []
    parts = msgs[0].get("content") if len(msgs) == 1 and msgs[0].get("role") == "user" else None
    if body.get("model") != "indic-ocr" or not isinstance(parts, list) or len(parts) != 1 or parts[0].get("type") != "image_url":
        return JSONResponse(status_code=422, content={"error": "expected one user message with one image_url part"})
    url = parts[0]["image_url"]["url"]
    if not url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")) or any(k in body for k in ("language", "temperature")):
        return JSONResponse(status_code=422, content={"error": "PNG/JPEG data URI only; no language or temperature"})
    LOG.append({"api": "bodhan-ocr", "bytes": len(base64.b64decode(url.split(",", 1)[1]))})
    text = "Chapter 1\n\nPlants make their own food by photosynthesis."
    return {"id": "fake", "object": "chat.completion", "model": "indic-ocr",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "blocks": [{"order": 0, "label": "Section-title", "type": "SectionHeader", "text": "Chapter 1"}]}


def reset() -> None:
    LOG.clear()
    _tts_durations.clear()
    _ratelimited.clear()


def dump() -> str:
    return json.dumps(LOG[-5:])


_ = np  # numpy is used indirectly via soundfile
