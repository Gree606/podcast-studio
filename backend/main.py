"""FastAPI app: REST API + the single-page frontend.

Run: uvicorn backend.main:app --host 0.0.0.0 --port 8000  (single worker; see store.py)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import jobs
from .agent import fast
from .languages import LANGUAGES
from .agent.context import LENGTHS
from .agent.orchestrator import regenerate_segment, run_episode, switch_version
from .clients.base import close_http_client
from .config import ROOT, get_settings, secrets_path
from .ingest import IngestError, ingest_pdf, ingest_text, select_chunks
from .photos import MAX_PAGES, MAX_PHOTO_BYTES, ingest_photos
from .prompts import PROMPT_VERSION
from .quiz import build_quiz
from .schemas import Episode, EpisodeSettings, new_id
from .store import store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("podcast.api")

MAX_UPLOAD = 25 * 1024 * 1024
SAMPLES_DIR = ROOT / "samples"
FRONTEND = ROOT / "frontend"


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    jobs.startup_recover()
    log.info("Podcast Studio up. mock=%s llm=%s tts=%s asr=%s", s.effective_mock(), bool(s.llm_base_url),
             bool(s.tts_base_url), bool(s.asr_base_url))
    yield
    await close_http_client()


app = FastAPI(title="Podcast Studio", version=PROMPT_VERSION, lifespan=lifespan)


# --- errors: never show a stack trace to a user ---------------------------------
@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on our side. Please try again."})


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "Some of the inputs were invalid. Please check and try again."})


# --- simple per-client rate limit for expensive endpoints --------------------------
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request, key: str, limit: int, window_s: int = 3600) -> None:
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    q = _hits[f"{key}:{ip}"]
    now = time.time()
    while q and now - q[0] > window_s:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, f"You've made a lot of {key} requests. Please wait a few minutes and try again.")
    q.append(now)


# --- the entry gate (secret code) ----------------------------------------------------
GATE_COOKIE = "m606"
GATE_OPEN = {"/api/unlock"}  # the only request that may change something without the code


def _gate_token(code: str) -> str:
    return hmac.new(code.encode(), b"mission-606-earth-unknown", hashlib.sha256).hexdigest()


def gate_unlocked(request: Request) -> bool:
    code = get_settings().access_code
    return not code or hmac.compare_digest(request.cookies.get(GATE_COOKIE, ""), _gate_token(code))


@app.middleware("http")
async def access_gate(request: Request, call_next):
    """With ACCESS_CODE set, every state-changing API call needs the gate cookie. Reads (episodes,
    audio, transcripts) stay open so shared links keep working."""
    if (request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD", "OPTIONS")
            and request.url.path not in GATE_OPEN and not gate_unlocked(request)):
        return JSONResponse(status_code=401, content={"detail": "Please enter the secret code first.", "locked": True})
    return await call_next(request)


class UnlockIn(BaseModel):
    code: str = Field(max_length=200)


@app.post("/api/unlock")
async def unlock(body: UnlockIn, request: Request):
    code = get_settings().access_code
    if not code:
        return {"ok": True}
    if not hmac.compare_digest(body.code.strip().lower(), code.strip().lower()):
        rate_limit(request, "secret code", limit=8, window_s=300)  # 8 wrong guesses per 5 minutes per client
        raise HTTPException(401, "That's not the code.")
    resp = JSONResponse({"ok": True})
    https = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    resp.set_cookie(GATE_COOKIE, _gate_token(code), max_age=30 * 24 * 3600, httponly=True, samesite="lax", secure=https)
    return resp


# --- health / config ---------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"ok": True, "mock": get_settings().effective_mock(), "version": PROMPT_VERSION, "active_jobs": len(jobs.active_jobs())}


def estimate_minutes(length: str, words: int) -> dict:
    if get_settings().pipeline_mode != "agent":  # 2 LLM calls, then recording; episodes are capped at ~2 minutes
        audio = fast.WORDS[length] / fast.WPM
        return {"audio_minutes": round(audio, 1), "wait_minutes": round(1.5 + 0.5 * audio, 1)}
    n_seg, wps = LENGTHS[length]
    audio = n_seg * wps / 150 + 0.3
    wait = 1.0 + n_seg * 0.6
    return {"audio_minutes": round(audio, 1), "wait_minutes": round(wait, 1)}


# Kid-facing names for the episode settings; ``value`` is what EpisodeSettings accepts.
PRESETS = {
    "length": [
        {"value": "short", "label": "Quick Bite", "emoji": "🍪", "hint": "A speedy snack of the big ideas"},
        {"value": "medium", "label": "Story Time", "emoji": "📖", "hint": "The whole story, nice and easy"},
        {"value": "long", "label": "Deep Dive", "emoji": "🤿", "hint": "Every corner of the chapter"},
    ],
    "tone": [
        {"value": "casual", "label": "Fun", "emoji": "🎈", "hint": "Jokes, wow-moments and everyday examples"},
        {"value": "balanced", "label": "Balanced", "emoji": "⚖️", "hint": "Friendly and clear"},
        {"value": "serious", "label": "Exam Hero", "emoji": "🏆", "hint": "Key terms and what to remember for tests"},
    ],
}


@app.get("/api/config")
async def config(request: Request):
    s = get_settings()
    return {"mock": s.effective_mock(), "max_input_words": s.max_input_words,
            "lengths": {k: estimate_minutes(k, 0) for k in LENGTHS},
            "presets": PRESETS, "secrets_path": str(secrets_path()), "configured": s.configured(),
            "languages": [{"code": x.code, "name": x.name, "native": x.native, "html": x.html} for x in LANGUAGES.values()],
            "locked": bool(s.access_code), "unlocked": gate_unlocked(request)}


# --- samples -----------------------------------------------------------------------
def _samples() -> list[dict]:
    p = SAMPLES_DIR / "samples.json"
    return json.loads(p.read_text()) if p.exists() else []


@app.get("/api/samples")
async def samples():
    return [{k: v for k, v in x.items() if k != "file"} for x in _samples()]


# --- sources -----------------------------------------------------------------------
def source_summary(src) -> dict:
    return {"id": src.id, "title": src.title, "origin": src.origin, "word_count": src.word_count, "notes": src.notes,
            "ocr_pages": src.ocr_pages, "page_texts": getattr(src, "page_texts", []),
            "sections": [{"id": x.id, "title": x.title, "word_count": x.word_count} for x in src.sections]}


@app.post("/api/sources")
async def create_source(request: Request, file: UploadFile | None = File(None), text: str | None = Form(None),
                        sample_id: str | None = Form(None), title: str | None = Form(None),
                        images: list[UploadFile] | None = File(None)):
    s = get_settings()
    rate_limit(request, "upload", 40)
    try:
        if sample_id:
            meta = next((x for x in _samples() if x["id"] == sample_id), None)
            if meta is None:
                raise HTTPException(404, "Unknown sample chapter.")
            cached = store.get_source(f"sample_{sample_id}")
            if cached:
                return source_summary(cached)
            path = SAMPLES_DIR / meta["file"]
            if path.suffix == ".pdf":
                src = await ingest_pdf(path.read_bytes(), s.max_input_words, origin="sample")
            else:
                src = ingest_text(path.read_text(encoding="utf-8"), s.max_input_words, origin="sample", title=meta["title"])
            src.id = f"sample_{sample_id}"
            src.title = meta["title"]
        elif images:
            # Photos of textbook pages: one page per photo, OCR'd in the order given.
            photos = [(im.filename or "", await im.read(MAX_PHOTO_BYTES + 1)) for im in images[: MAX_PAGES + 1]]
            src = await ingest_photos(photos, s.max_input_words, title=(title or "").strip() or None)
        elif file is not None:
            data = await file.read(MAX_UPLOAD + 1)
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "That file is too large (max 25 MB). Try uploading just one chapter.")
            name = (file.filename or "").lower()
            if name.endswith(".txt") or name.endswith(".md"):
                src = ingest_text(data.decode("utf-8", errors="ignore"), s.max_input_words, origin="upload", title=title)
            else:
                src = await ingest_pdf(data, s.max_input_words)
        elif text:
            src = ingest_text(text, s.max_input_words, title=title)
        else:
            raise HTTPException(400, "Upload a PDF, snap a photo of a page, paste some text, or pick a sample chapter.")
    except IngestError as e:
        raise HTTPException(422, str(e)) from None
    store.save_source(src)
    return source_summary(src)


@app.get("/api/sources/{sid}")
async def get_source(sid: str):
    src = store.get_source(sid)
    if not src:
        raise HTTPException(404, "Chapter not found. Please upload it again.")
    return source_summary(src)


# --- episodes ----------------------------------------------------------------------
class CreateEpisode(BaseModel):
    source_id: str
    settings: EpisodeSettings = Field(default_factory=EpisodeSettings)


@app.post("/api/episodes")
async def create_episode(body: CreateEpisode, request: Request):
    src = store.get_source(body.source_id)
    if not src:
        raise HTTPException(404, "Chapter not found. Please upload it again.")
    rate_limit(request, "episode", 12)
    st = body.settings
    if st.language not in LANGUAGES:
        raise HTTPException(400, "That language isn't supported yet.")
    st.host_name = (st.host_name or "Error606").strip()[:20] or "Error606"
    st.expert_name = (st.expert_name or "Gree").strip()[:20] or "Gree"
    valid = {x.id for x in src.sections}
    st.section_ids = [x for x in st.section_ids if x in valid]
    _, titles = select_chunks(src, st.section_ids, get_settings().max_input_words)
    title = src.title if not st.section_ids or len(st.section_ids) == len(src.sections) else f"{src.title}: {', '.join(titles[:2])}{'...' if len(titles) > 2 else ''}"
    ep = Episode(id=new_id("ep_"), source_id=src.id, title=title, settings=st)
    store.save_episode(ep)
    jobs.submit(ep.id, "generate", lambda: run_episode(ep.id))
    return {"id": ep.id}


@app.get("/api/episodes")
async def list_episodes():
    return [{"id": e.id, "title": e.title, "status": e.status, "created_at": e.created_at, "duration": e.duration}
            for e in store.list_episodes()]


def episode_view(ep: Episode, since: int) -> dict:
    d = ep.model_dump(exclude={"trace"})
    for seg in d["segments"]:
        seg["candidates"] = [{"index": c["index"], "round": c["round"], "scores": c["scores"], "feedback": c["feedback"],
                              "issues": c["issues"], "words": sum(len(t["text"].split()) for t in c["turns"])}
                             for c in seg["candidates"]]
    d["trace"] = [t.model_dump() for t in ep.trace[since:]]
    d["trace_len"] = len(ep.trace)
    d["queue_position"] = jobs.queue_position(ep.id)
    d["busy"] = jobs.is_busy(ep.id)
    return d


def _get_ep(eid: str) -> Episode:
    ep = store.get_episode(eid)
    if not ep:
        raise HTTPException(404, "Episode not found. It may have been created on an older version of the app.")
    return ep


@app.get("/api/episodes/{eid}")
async def get_episode(eid: str, since: int = 0):
    return episode_view(_get_ep(eid), max(0, since))


@app.post("/api/episodes/{eid}/resume")
async def resume(eid: str):
    ep = _get_ep(eid)
    if jobs.is_busy(eid):
        return {"ok": True}
    if ep.status not in ("failed", "interrupted"):
        raise HTTPException(409, "This episode isn't stopped.")
    ep.status, ep.error, ep.message = "queued", None, "Resuming"
    store.save_episode(ep)
    jobs.submit(eid, "generate", lambda: run_episode(eid))
    return {"ok": True}


class RegenBody(BaseModel):
    instruction: str | None = Field(default=None, max_length=300)


@app.post("/api/episodes/{eid}/segments/{sid}/regenerate")
async def regenerate(eid: str, sid: str, body: RegenBody, request: Request):
    ep = _get_ep(eid)
    if not any(x.id == sid for x in ep.segments):
        raise HTTPException(404, "Segment not found.")
    if jobs.is_busy(eid):
        raise HTTPException(409, "This episode is still being worked on. Please wait for it to finish.")
    rate_limit(request, "regenerate", 30)
    seg = next(x for x in ep.segments if x.id == sid)
    seg.status = "writing"
    store.save_episode(ep)
    instr = (body.instruction or "").strip() or None
    jobs.submit(eid, "regenerate", lambda: regenerate_segment(eid, sid, instr))
    return {"ok": True}


class VersionBody(BaseModel):
    version: int


@app.post("/api/episodes/{eid}/segments/{sid}/activate")
async def activate(eid: str, sid: str, body: VersionBody):
    _get_ep(eid)
    if jobs.is_busy(eid):
        raise HTTPException(409, "Please wait for the current job to finish.")
    if not await switch_version(eid, sid, body.version):
        raise HTTPException(404, "Version not found.")
    return {"ok": True}


def _safe_file(eid: str, name: str | None) -> Path:
    if not name or "/" in name or ".." in name:
        raise HTTPException(404, "Audio not ready yet.")
    p = store.episode_dir(eid) / name
    if not p.exists():
        raise HTTPException(404, "Audio not ready yet.")
    return p


@app.get("/api/episodes/{eid}/audio")
async def episode_audio(eid: str, download: bool = False):
    ep = _get_ep(eid)
    p = _safe_file(eid, ep.audio_file)
    fname = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in ep.title)[:60].strip() or "episode"
    ext = p.suffix.lstrip(".")
    return FileResponse(p, media_type="audio/mpeg" if ext == "mp3" else "audio/wav", filename=f"{fname}.{ext}" if download else None)


@app.get("/api/episodes/{eid}/segments/{sid}/audio")
async def segment_audio(eid: str, sid: str, version: int | None = None):
    ep = _get_ep(eid)
    seg = next((x for x in ep.segments if x.id == sid), None)
    if not seg:
        raise HTTPException(404, "Segment not found.")
    v = next((x for x in seg.versions if x.version == (version or seg.active_version)), None)
    return FileResponse(_safe_file(eid, v.audio_file if v else None), media_type="audio/wav")


@app.get("/api/episodes/{eid}/transcript", response_class=PlainTextResponse)
async def transcript(eid: str):
    ep = _get_ep(eid)
    st = ep.settings
    src = store.get_source(ep.source_id)
    by_id = {c.id: c for c in src.chunks} if src else {}
    lines = [f"# {ep.title}", "", f"A Mission 606: Earth Unknown episode with {st.host_name} (host) and {st.expert_name} (explainer).", ""]
    for seg in ep.segments:
        v = seg.current
        lines.append(f"## {seg.index + 1}. {seg.title}")
        if not v:
            lines += ["(not available)", ""]
            continue
        for t in v.turns:
            ts = f"[{int(t.start // 60):02d}:{int(t.start % 60):02d}] " if t.start is not None else ""
            lines.append(f"{ts}**{st.host_name if t.speaker == 'host' else st.expert_name}:** {t.text}")
        pages = sorted({by_id[c].page for c in seg.chunk_ids if c in by_id and by_id[c].page})
        secs = list(dict.fromkeys(by_id[c].section for c in seg.chunk_ids if c in by_id))
        lines += ["", f"_Sources: {', '.join(secs)}" + (f" (pages {', '.join(map(str, pages))})" if pages else "") + "_", ""]
    return PlainTextResponse("\n".join(lines), headers={"Content-Disposition": f'attachment; filename="{ep.id}_transcript.md"'})


@app.get("/api/episodes/{eid}/sources")
async def episode_sources(eid: str):
    """Passages each segment drew on, for the Sources panel."""
    ep = _get_ep(eid)
    src = store.get_source(ep.source_id)
    if not src:
        return {}
    by_id = {c.id: c for c in src.chunks}
    return {seg.id: [{"id": c, "section": by_id[c].section, "page": by_id[c].page, "text": by_id[c].text}
                     for c in seg.chunk_ids if c in by_id] for seg in ep.segments}


@app.get("/api/episodes/{eid}/quiz")
async def episode_quiz(eid: str):
    """Fill-in-the-blank questions built only from the episode's verified key facts ([] = no quiz)."""
    ep = _get_ep(eid)
    return build_quiz(ep, store.get_source(ep.source_id))


# --- frontend ----------------------------------------------------------------------
NO_CACHE = {"Cache-Control": "no-cache"}


@app.get("/")
async def index():
    return FileResponse(FRONTEND / "index.html", media_type="text/html", headers=NO_CACHE)


@app.get("/app.css")
async def app_css():
    return FileResponse(FRONTEND / "app.css", media_type="text/css", headers=NO_CACHE)


@app.get("/app.js")
async def app_js():
    return FileResponse(FRONTEND / "app.js", media_type="text/javascript", headers=NO_CACHE)


@app.get("/classic")
async def classic():
    """The previous single-file UI, kept as a fallback."""
    return FileResponse(FRONTEND / "classic.html", media_type="text/html", headers=NO_CACHE)


@app.get("/fonts/{name}")
async def font(name: str):
    p = FRONTEND / "fonts" / name
    if not name.endswith(".woff2") or "/" in name or ".." in name or not p.is_file():
        raise HTTPException(404, "Not found.")
    return FileResponse(p, media_type="font/woff2", headers={"Cache-Control": "public, max-age=31536000"})


@app.get("/sprites/{name}")
async def sprite(name: str):
    p = FRONTEND / "sprites" / name
    if not name.endswith(".png") or "/" in name or ".." in name or not p.is_file():
        raise HTTPException(404, "Not found.")
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/sounds/{name}")
async def sound(name: str):
    p = FRONTEND / "sounds" / name
    if not name.endswith((".wav", ".mp3")) or "/" in name or ".." in name or not p.is_file():
        raise HTTPException(404, "Not found.")
    kind = "audio/mpeg" if name.endswith(".mp3") else "audio/wav"
    return FileResponse(p, media_type=kind, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(FRONTEND / "sprites" / "logo.png", media_type="image/png")
