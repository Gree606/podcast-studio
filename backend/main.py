"""FastAPI app: REST API + the single-page frontend.

Run: uvicorn backend.main:app --host 0.0.0.0 --port 8000  (single worker; see store.py)
"""
from __future__ import annotations

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
from .agent.orchestrator import regenerate_segment, run_episode, switch_version
from .clients.base import close_http_client
from .config import ROOT, get_settings
from .ingest import IngestError, ingest_pdf, ingest_text, select_chunks
from .agent.context import LENGTHS
from .prompts import PROMPT_VERSION
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


# --- health / config ---------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"ok": True, "mock": get_settings().effective_mock(), "version": PROMPT_VERSION, "active_jobs": len(jobs.active_jobs())}


def estimate_minutes(length: str, words: int) -> dict:
    n_seg, wps = LENGTHS[length]
    audio = n_seg * wps / 150 + 0.3
    wait = 1.0 + n_seg * 0.6
    return {"audio_minutes": round(audio, 1), "wait_minutes": round(wait, 1)}


@app.get("/api/config")
async def config():
    s = get_settings()
    return {"mock": s.effective_mock(), "max_input_words": s.max_input_words,
            "lengths": {k: estimate_minutes(k, 0) for k in LENGTHS}}


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
            "ocr_pages": src.ocr_pages,
            "sections": [{"id": x.id, "title": x.title, "word_count": x.word_count} for x in src.sections]}


@app.post("/api/sources")
async def create_source(request: Request, file: UploadFile | None = File(None), text: str | None = Form(None),
                        sample_id: str | None = Form(None), title: str | None = Form(None)):
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
            raise HTTPException(400, "Upload a PDF, paste some text, or pick a sample chapter.")
    except IngestError as e:
        raise HTTPException(422, str(e))
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
    st.host_name = (st.host_name or "Riya").strip()[:20] or "Riya"
    st.expert_name = (st.expert_name or "Arjun").strip()[:20] or "Arjun"
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
    return FileResponse(p, media_type="audio/mpeg", filename=f"{fname}.mp3" if download else None)


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
    lines = [f"# {ep.title}", "", f"A Chapter Chat episode with {st.host_name} (host) and {st.expert_name} (explainer).", ""]
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


# --- frontend ----------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(FRONTEND / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(FRONTEND / "favicon.svg", media_type="image/svg+xml")
