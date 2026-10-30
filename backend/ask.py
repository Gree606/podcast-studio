"""Ask Gree: a student's doubt about an episode, answered from the chapter (retrieval-augmented).

1. Retrieve: BM25 over the whole chapter for the question (plus the previous question, so
   "why?" follow-ups still find the topic) and the passages of the part the student was
   listening to. A question in another script than the chapter (a Hindi doubt about an
   English chapter) is first turned into search words in the chapter's language by a short
   LLM call; if still nothing matches, the passages the episode itself cited are used.
2. Answer: one LLM call, in Gree's voice, from those passages only. It cites the passage ids
   it used and says plainly when the chapter doesn't cover the doubt (grounded = false).
3. Voice: on request, the answer is spoken in Gree's TTS voice and cached next to the episode.

Answers are saved in data/episodes/<id>/asks.json so their audio can be fetched later.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from . import languages
from .agent.writer import SHOW
from .audio.stitch import prepare_clip, to_wav_bytes
from .clients import llm, tts
from .config import get_settings
from .prompts import ASK_GREE, ASK_KEYWORDS
from .retrieval import BM25Index
from .schemas import Chunk, Episode, Source, new_id
from .store import atomic_write, store
from .textnorm import normalize_for_tts

log = logging.getLogger("podcast.ask")

MAX_PASSAGES = 6
MAX_HISTORY = 3
_locks: dict[str, asyncio.Lock] = {}


class AskError(Exception):
    """Safe to show to the student."""


def _lock(eid: str) -> asyncio.Lock:
    return _locks.setdefault(eid, asyncio.Lock())


def _non_latin(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if ord(c) > 0x24F) / len(letters) if letters else 0.0


def needs_translation(src: Source, question: str) -> bool:
    """The question is written in a different script from the chapter, so BM25 can't match it."""
    sample = " ".join(c.text for c in src.chunks[:5])
    return abs(_non_latin(question) - _non_latin(sample)) > 0.5


async def search_words(src: Source, question: str) -> str:
    """A few search words for the question, in the chapter's own language (one short LLM call)."""
    lang = "the language and script of this text: " + src.chunks[0].text[:120] if src.chunks else "English"
    out = await llm.chat_json([{"role": "user", "content": ASK_KEYWORDS.format(question=question[:500], language=lang)}],
                              task="ask_keywords", temperature=0.0, max_tokens=200,
                              thinking=get_settings().llm_check_thinking or None)
    return " ".join(str(out.get("keywords", "")).split())[:300]


def retrieve(ep: Episode, src: Source, question: str, *, segment_id: str | None = None,
             history: list[dict] | None = None, extra_query: str = "") -> list[Chunk]:
    by_id = {c.id: c for c in src.chunks}
    picked: list[str] = []

    def add(ids):
        for cid in ids:
            if cid in by_id and cid not in picked and len(picked) < MAX_PASSAGES:
                picked.append(cid)

    index = BM25Index(src.chunks)
    hits = [c.id for c, _ in index.search(f"{question} {extra_query}", k=4)]
    if len(hits) < 2 and history:  # a short follow-up ("why?"): search with the previous question too
        hits += [c.id for c, _ in index.search(f"{history[-1].get('q', '')} {question}", k=3)]
    add(hits)
    seg = next((s for s in ep.segments if s.id == segment_id), None)
    if seg:
        add(seg.chunk_ids[:2])
    if not hits:  # nothing lexical in common: lean on what the episode itself taught
        add(c for s in ep.segments for c in s.chunk_ids[:2])
    return [by_id[c] for c in picked]


def _outline(ep: Episode) -> str:
    return " | ".join(f"Part {s.index + 1}: {s.title}" for s in ep.segments)


def _now(ep: Episode, segment_id: str | None) -> str:
    seg = next((s for s in ep.segments if s.id == segment_id), None)
    v = seg.current if seg else None
    if not v:
        return ""
    st = ep.settings
    lines = "\n".join(f"{st.host_name if t.speaker == 'host' else st.expert_name}: {t.text}" for t in v.turns[-8:])
    return f'The student was listening to part {seg.index + 1}, "{seg.title}":\n{lines}\n'


def _history(history: list[dict]) -> str:
    if not history:
        return "(nothing yet)"
    return "\n".join(f"Student: {h.get('q', '')[:300]}\nYou: {h.get('a', '')[:600]}" for h in history[-MAX_HISTORY:])


def load_asks(eid: str) -> list[dict]:
    p = store.episode_dir(eid) / "asks.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    except (OSError, ValueError):
        return []


async def answer(ep: Episode, src: Source, question: str, *, segment_id: str | None = None,
                 history: list[dict] | None = None) -> dict:
    question = " ".join(question.split())
    if len(question) < 2:
        raise AskError("Type your doubt first.")
    if not any(s.current for s in ep.segments):
        raise AskError("Gree can take doubts once the first part of the episode is ready.")
    history = (history or [])[-MAX_HISTORY:]
    extra = ""
    if needs_translation(src, question):
        try:
            extra = await search_words(src, question)
        except Exception as e:  # noqa: BLE001 - retrieval still falls back to the episode's own passages
            log.warning("ep=%s ask: search words failed: %s", ep.id, e)
    passages = retrieve(ep, src, question, segment_id=segment_id, history=history, extra_query=extra)
    st = ep.settings
    prompt = ASK_GREE.format(
        expert=st.expert_name, show=SHOW, title=ep.title, episode_language=languages.get(st.language).name,
        outline=_outline(ep), now=_now(ep, segment_id), history=_history(history),
        passages="\n\n".join(f"[{c.id}] ({c.section}{f', page {c.page}' if c.page else ''}) {c.text}" for c in passages) or "(none found)",
        question=question[:500], codes=", ".join(languages.LANGUAGES))
    t0 = time.time()
    out = await llm.chat_json([{"role": "user", "content": prompt}], task="ask_gree", temperature=0.4, max_tokens=900,
                              thinking=get_settings().llm_check_thinking or None)
    text = " ".join(str(out.get("answer", "")).split())
    if not text:
        raise AskError("Gree lost the signal for a moment. Please ask again.")
    given = {c.id: c for c in passages}
    cited = [given[c] for c in out.get("chunk_ids") or [] if isinstance(c, str) and c in given]
    lang = out.get("language") if out.get("language") in languages.LANGUAGES else st.language
    follow = [" ".join(str(f).split())[:120] for f in (out.get("follow_ups") or []) if str(f).strip()][:2]
    item = {"id": new_id("a"), "q": question[:500], "a": text, "language": lang,
            "grounded": bool(out.get("grounded", True)) and bool(cited), "follow_ups": follow,
            "sources": [{"id": c.id, "section": c.section, "page": c.page, "text": c.text} for c in cited],
            "retrieved": [c.id for c in passages], "search_words": extra, "segment_id": segment_id, "t": time.time(),
            "seconds": round(time.time() - t0, 2)}
    async with _lock(ep.id):
        asks = load_asks(ep.id)
        asks.append(item)
        atomic_write(store.episode_dir(ep.id) / "asks.json", json.dumps(asks[-200:], ensure_ascii=False))
    log.info("ep=%s ask %s: %d passages, cited %s, grounded=%s, %.1fs", ep.id, item["id"], len(passages),
             [c["id"] for c in item["sources"]], item["grounded"], item["seconds"])
    return item


async def answer_audio(ep: Episode, aid: str) -> Path | None:
    """Gree speaking the answer (WAV), synthesised once and cached. None if there's no such answer."""
    if not aid.replace("_", "").isalnum():
        return None
    path = store.episode_dir(ep.id) / "asks" / f"{aid}.wav"
    if path.exists():
        return path
    item = next((a for a in load_asks(ep.id) if a["id"] == aid), None)
    if not item:
        return None
    async with _lock(f"{ep.id}:{aid}"):
        if not path.exists():
            sr = get_settings().sample_rate
            wav = await tts.synthesize(normalize_for_tts(item["a"], item["language"]), "expert", language=item["language"])
            path.parent.mkdir(exist_ok=True)
            atomic_write(path, to_wav_bytes(prepare_clip(wav, sr), sr))
    return path
