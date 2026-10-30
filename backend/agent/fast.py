"""FAST pipeline: exactly 2 LLM calls per episode (and 2 per single-part regeneration).

    Perceive : retrieve the chapter passages in scope (no LLM)
    Act      : call 1 writes the whole ~2-minute episode in one go, as 3 parts that
               cite the passages they used (the model "thinks" internally)
    Observe  : call 2 fact-checks every claim against those passages and returns
               corrected turns for any part with an unsupported claim
    Act      : apply the corrections, then the unchanged tool steps: TTS per line,
               ASR listen-back with re-recording, stitching

Why: the multi-step agent (planner loop, per-segment critic, reflections, best-of-N)
made 25-40 LLM calls per episode, which free-tier quotas cannot sustain and which
added little over a single thinking-model call for a 2-minute script. The parts
that a single call cannot do (checking against the book, hearing its own audio,
OCR, per-part regeneration) are kept as tool steps. The old pipeline remains
available for comparison ("Deep agent" on the launch screen, or PIPELINE_MODE=agent).
"""
from __future__ import annotations

import logging
import re

from .. import languages
from ..clients.base import ServiceError
from ..clients.llm import chat_json
from ..config import get_settings
from ..prompts import FAST_REWRITE, FAST_VERIFY, FAST_WRITE
from ..schemas import Candidate, Claim, KeyFact, Segment, SegmentVersion, Turn, new_id
from .context import AgentContext
from .planner import assign_kinds
from .writer import SHOW, TONES, WriterError, parse_turns, script_text

log = logging.getLogger("podcast.fast")

STAGE_WRITE, STAGE_VERIFY = "write", "verify"
PASSAGE_WORDS = 2400  # enough for a 2-minute episode; keeps the single prompt small and fast
WPM = 120  # measured: the Bodhan voices speak ~120 words per minute (348 words -> 175 s)
# target words per episode, capped at ~2 minutes (quick back-and-forth runs longer per word than monologue)
WORDS = {"short": 160, "medium": 220, "long": 250}
# "1. Arjun: ..." / "Riya: ..." - a number and/or speaker label the checker sometimes puts in front of a fixed line
_LINE_LABEL = re.compile(r"^(?:\d+[.)]\s*)?[A-Za-z0-9 ]{1,20}:\s+")


def target_words(ctx: AgentContext) -> int:
    return WORDS.get(ctx.episode.settings.length, 240)


def select_passages(ctx: AgentContext, limit_words: int = PASSAGE_WORDS) -> list[str]:
    """Chunk ids to show the writer: all of them if they fit, else evenly spread across the
    chapter so a 2-minute episode still covers its beginning, middle and end."""
    ids = [c.id for c in ctx.chunks]
    words = {c.id: len(c.text.split()) for c in ctx.chunks}
    if sum(words.values()) <= limit_words:
        return ids
    keep, total, n = [], 0, len(ids)
    avg = max(1, sum(words.values()) // n)
    want = max(3, limit_words // avg)
    for k in range(want):
        cid = ids[min(n - 1, round(k * (n - 1) / max(1, want - 1)))]
        if cid not in keep and total + words[cid] <= limit_words:
            keep.append(cid)
            total += words[cid]
    return keep


def _tidy(turns: list[Turn]) -> list[Turn]:
    """Every episode is the listener's first: Gemini still says "welcome back" despite the prompt."""
    for x in turns:
        x.text = re.sub(r"\b[Ww]elcome back\b", "Welcome", x.text)
    return turns


def _part_turns(raw: dict | list, names: tuple[str, str]) -> list[Turn]:
    # parts of a 2-minute episode are short; a 2-line exchange is still a usable part
    return _tidy(parse_turns(raw if isinstance(raw, dict) else {"turns": raw}, *names, min_turns=2))


# ---------------------------------------------------------------- call 1: write
async def write_episode(ctx: AgentContext) -> list[Segment]:
    st = ctx.episode.settings
    names = (st.host_name, st.expert_name)
    pids = select_passages(ctx)
    ctx.emit(STAGE_WRITE, "info", f"Perceive: {len(pids)} passages from the chapter in scope "
             f"({sum(len(ctx.by_id[c].text.split()) for c in pids)} words).")
    words = target_words(ctx)
    prompt = FAST_WRITE.format(show=SHOW, host=st.host_name, expert=st.expert_name, words=words,
                               minutes=round(words / WPM, 1), tone=TONES[st.tone], title=ctx.source.title,
                               passages=ctx.passages_text(pids, limit_words=PASSAGE_WORDS + 200), extra="",
                               language_rule=languages.writing_rule(st.language))
    ctx.status(STAGE_WRITE, "Writing the whole episode (1 model call)", 0.08)
    async with ctx.creeping(0.08, 0.38, expected_s=25):
        data = await chat_json([{"role": "user", "content": prompt}], task="write_episode", expect=dict,
                               temperature=0.8, max_tokens=3500, thinking=get_settings().llm_write_thinking or None)
    parts = data.get("parts") if isinstance(data, dict) else None
    if not isinstance(parts, list) or not parts:
        raise ServiceError("Language model", "did not return the episode in the expected format")
    ctx.episode.learn_line = str(data.get("learn_line") or "").strip() or None
    if data.get("title") and ctx.episode.title == ctx.source.title:
        ctx.episode.title = str(data["title"])[:120]
    segments: list[Segment] = []
    for i, p in enumerate(parts[:4]):
        try:
            turns = _part_turns(p, names)
        except WriterError as e:
            log.warning("part %d unusable: %s", i + 1, e)
            continue
        cids = [c for c in (p.get("chunk_ids") or []) if c in ctx.by_id] or pids[:3]
        seg = Segment(id=new_id("seg_"), index=len(segments), title=str(p.get("title") or f"Part {i + 1}")[:80],
                      goal="", chunk_ids=cids, status="checking")
        seg.candidates = [Candidate(index=0, round=0, turns=turns)]
        segments.append(seg)
    if not segments:
        raise ServiceError("Language model", "returned no usable dialogue")
    assign_kinds(segments)
    total = sum(len(t.text.split()) for s in segments for t in s.candidates[0].turns)
    ctx.emit(STAGE_WRITE, "action", f"Wrote the whole episode in one call: {len(segments)} parts, "
             f"{sum(len(s.candidates[0].turns) for s in segments)} lines, {total} words (~{total / WPM:.1f} min).",
             data={"parts": [s.title for s in segments]})
    return segments


# ---------------------------------------------------------------- call 2: verify (+ fix)
async def verify_segments(ctx: AgentContext, segments: list[Segment], only: list[Segment] | None = None) -> dict[str, list[Claim]]:
    """One call: fact-check the given parts against their passages and apply the returned fixes.
    The final turns end up in seg.candidates[-1]; returns the claims per segment id."""
    st = ctx.episode.settings
    names = (st.host_name, st.expert_name)
    targets = only or segments
    cids: list[str] = list(dict.fromkeys(c for s in targets for c in s.chunk_ids))
    script = "\n\n".join(f"PART {s.index + 1}: {s.title}\n" + script_text(s.candidates[-1].turns, names) for s in targets)
    prompt = FAST_VERIFY.format(passages=ctx.passages_text(cids, limit_words=PASSAGE_WORDS + 200), script=script)
    ctx.status(STAGE_VERIFY, "Fact-checking against the chapter (1 model call)", 0.4)
    try:
        async with ctx.creeping(0.4, 0.6, expected_s=15):
            data = await chat_json([{"role": "user", "content": prompt}], task="verify_episode", expect=dict,
                                   temperature=0.0, max_tokens=6000, thinking=get_settings().llm_check_thinking or None)
    except ServiceError as e:
        ctx.emit(STAGE_VERIFY, "warning", f"Fact-check unavailable ({e.user_message}); the script is kept as written.")
        return {s.id: [] for s in targets}
    returned = [p for p in (data.get("parts") or []) if isinstance(p, dict)]
    by_part = {}
    for p in returned:
        try:
            by_part[int(p.get("part", 0))] = p
        except (TypeError, ValueError):
            pass
    out: dict[str, list[Claim]] = {}
    for k, s in enumerate(targets):
        # match by part number; fall back to position (a lone part is often renumbered "1")
        p = by_part.get(s.index + 1) or (returned[k] if len(returned) == len(targets) else {})
        claims = parse_claims(p, s.candidates[-1].turns, ctx.by_id)
        bad = [c for c in claims if not c.supported]
        # per-line fixes; else the older answer shape: a whole corrected part
        fixed = (apply_line_fixes(s.candidates[-1].turns, p.get("fixes")) or p.get("fixed_turns")) if bad else None
        if fixed:
            try:
                turns = fixed if isinstance(fixed, list) and fixed and isinstance(fixed[0], Turn) else _part_turns(fixed, names)
                s.candidates.append(Candidate(index=len(s.candidates), round=1, turns=turns))
                ctx.emit(STAGE_VERIFY, "action", f"Part {s.index + 1}: corrected {len(bad)} unsupported claim(s): "
                         + " | ".join(c.claim[:80] for c in bad[:3]), s.id)
                for c in bad:  # the corrected version no longer makes these claims
                    c.corrected = True
                    c.note = (c.note or "") + " (removed/corrected in the final script)"
            except WriterError:
                ctx.emit(STAGE_VERIFY, "warning", f"Part {s.index + 1}: fix was unusable; flagging the claims instead.", s.id)
        out[s.id] = claims
        # supported, sourced claims double as the part's key facts (shown in the UI, used by the quiz)
        s.key_facts = [KeyFact(fact=c.claim, chunk_id=c.evidence_chunk_id, verified=True)
                       for c in claims if c.supported and c.evidence_chunk_id]
    every = [c for cl in out.values() for c in cl]
    n_lines = sum(len(s.candidates[0].turns) for s in targets)
    ctx.emit(STAGE_VERIFY, "observation", f"Fact-check (1 call): checked all {n_lines} spoken lines; "
             f"{len(every)} make a claim about the subject, {sum(c.supported for c in every)} supported by the chapter, "
             f"{sum(c.corrected for c in every)} corrected.", data={"unsupported": [c.claim for c in every if not c.supported]})
    return out


def parse_claims(p: dict, turns: list[Turn], by_id: dict) -> list[Claim]:
    """The checker's verdicts for one part: one per spoken line that makes a claim ("lines"), or the
    older free-form "claims" list. Lines with no claim (greetings, questions, jokes) are skipped."""
    claims: list[Claim] = []
    for c in p.get("lines") or p.get("claims") or []:
        if not isinstance(c, dict) or not str(c.get("claim", "")).strip():
            continue
        try:
            line = int(c["line"]) if "line" in c else None
        except (TypeError, ValueError):
            line = None
        if line is not None and not 1 <= line <= len(turns):
            line = None
        ev = str(c.get("evidence") or "")
        claims.append(Claim(claim=str(c["claim"]).strip(), supported=bool(c.get("supported", True)), line=line,
                            evidence_chunk_id=ev if ev in by_id else None, note=c.get("note") or None))
    return claims


def apply_line_fixes(turns: list[Turn], fixes) -> list[Turn] | None:
    """Apply the fact-checker's per-line corrections ({"line": n, "text": ...}, 1-based) to a copy of the
    part. Only those lines change; nothing can be reordered or pulled in from another part."""
    if not isinstance(fixes, list) or not fixes:
        return None
    out: list[Turn | None] = [t.model_copy() for t in turns]
    changed = False
    for fx in fixes:
        try:
            i = int(fx.get("line")) - 1
        except (AttributeError, TypeError, ValueError):
            continue
        if not 0 <= i < len(out) or out[i] is None:
            continue
        text = _LINE_LABEL.sub("", str(fx.get("text") or "").strip(), count=1)
        out[i] = Turn(speaker=out[i].speaker, text=text) if text else None
        changed = True
    kept = [t for t in out if t is not None]
    return kept if changed and kept else None


def next_version_number(seg: Segment) -> int:
    return max((v.version for v in seg.versions), default=0) + 1


def new_version(seg: Segment, claims: list[Claim], instruction: str | None = None) -> SegmentVersion:
    """The segment's final (fact-checked) turns as its next version, ready to record."""
    return SegmentVersion(version=next_version_number(seg), instruction=instruction,
                          turns=[t.model_copy() for t in seg.candidates[-1].turns], claims=claims)


# ---------------------------------------------------------------- regenerate one part (2 calls)
async def rewrite_part(ctx: AgentContext, seg: Segment, segments: list[Segment], instruction: str | None) -> list[Claim]:
    st = ctx.episode.settings
    names = (st.host_name, st.expert_name)
    prev = segments[seg.index - 1] if seg.index > 0 else None
    nxt = segments[seg.index + 1] if seg.index + 1 < len(segments) else None
    cur = seg.current.turns if seg.current else []
    prompt = FAST_REWRITE.format(
        index=seg.index + 1, total=len(segments), title=seg.title, show=SHOW, host=st.host_name, expert=st.expert_name,
        instruction=instruction or "Make this part better: more natural and engaging, same facts.",
        words=max(60, sum(len(t.text.split()) for t in cur) or target_words(ctx) // 3),
        prev_last=(prev.current.turns[-1].text[:200] if prev and prev.current else "(this is the opening)"),
        next_first=(nxt.current.turns[0].text[:200] if nxt and nxt.current else "(this is the last part)"),
        role=("This is the opening: it still starts with the alien greeting the astronaut and asking what they will learn about Earth today, then the welcome to the show." if seg.index == 0 else
              "This is the last part: keep the wrap-up where the alien says back what it learned and the astronaut fixes and completes it, then the goodbye."
              if nxt is None else ""),
        language_rule=languages.writing_rule(st.language),
        current=script_text(cur, names), passages=ctx.passages_text(seg.chunk_ids or [c.id for c in ctx.chunks[:4]]))
    data = await chat_json([{"role": "user", "content": prompt}], task="rewrite_part", expect=dict,
                           temperature=0.7, max_tokens=2000, thinking=get_settings().llm_write_thinking or None)
    turns = _part_turns(data, names)
    seg.candidates = [Candidate(index=0, round=0, turns=turns)]
    seg.chunk_ids = [c for c in (data.get("chunk_ids") or []) if c in ctx.by_id] or seg.chunk_ids
    ctx.emit("regenerate", "action", f"Rewrote part {seg.index + 1} in one call ({len(turns)} lines).", seg.id)
    return (await verify_segments(ctx, segments, only=[seg])).get(seg.id, [])
