"""The deep agent within a fixed budget of about 10 LLM calls per episode.

    Plan     ReAct planner with tools and hybrid RAG search (<= 3 calls), then the
             key facts' dual-layer check: rules, then one model judge (1 call)      planner.py
    Act      write all parts in one call, following the plan and the memory (1)  write_from_plan
    Observe  the critic grades every part on the rubric in one call (1)          critique_all
    Act      revise only the parts below the bar, weakest first, from the        refine
             critic's note stored as a Reflexion lesson (<= 2)
    Observe  re-grade the revisions in one call and keep the better draft (1)
    Observe  fact-check every spoken line with fixes in one call (1)             fast.verify_segments
             then TTS, ASR listen-back and mixing (no LLM)

The per-part pipeline (best-of-N drafts per part, per-part critic and claim checks) made 25-48
calls per episode; it stays in search.py/claims.py for the eval ablation.
"""
from __future__ import annotations

import asyncio
import logging

from .. import languages
from ..clients.base import ServiceError
from ..clients.llm import chat_json
from ..config import get_settings
from ..prompts import CRITIC_ALL, FAST_WRITE
from ..schemas import Candidate, Scores, Segment, Turn
from .context import AgentContext
from .critic import WEIGHTS, _num, deterministic_issues, lesson_from
from .fast import PASSAGE_WORDS, WPM, _part_turns, select_passages, target_words
from .writer import SHOW, TONES, WriterError, revise, script_text

log = logging.getLogger("podcast.lean")
STAGE = "write"
PARTS = 3          # the one-call writer's episode shape (hook, core, wrap-up)
MAX_REVISIONS = 2


def calls_used(ctx: AgentContext) -> int:
    return ctx.episode.metrics.llm_calls - ctx.llm_start


def calls_left(ctx: AgentContext) -> int:
    cap = get_settings().deep_max_llm_calls
    return 10 ** 6 if cap <= 0 else cap - calls_used(ctx)


def _plan_text(segments: list[Segment]) -> str:
    out = []
    for s in segments:
        facts = "\n".join(f"     - {f.fact} [{f.chunk_id}]" for f in s.key_facts) or "     - (the main ideas of its passages)"
        out.append(f"  Part {s.index + 1}: {s.title} (difficulty {s.difficulty}/5). Goal: {s.goal}\n   Key facts (verified):\n{facts}")
    return "\n".join(out)


async def write_from_plan(ctx: AgentContext, segments: list[Segment]) -> None:
    """One call writes every part, following the planner's outline and verified key facts."""
    st = ctx.episode.settings
    names = (st.host_name, st.expert_name)
    pids = list(dict.fromkeys([c for s in segments for c in s.chunk_ids] + select_passages(ctx)))
    words = target_words(ctx)
    memory = [r for s in segments for r in s.reflections[-2:]]
    extra = ("\nFollow this plan from the planning agent: one part per planned part, in order, teaching its key facts.\n"
             + _plan_text(segments)
             + ("\nLessons from earlier drafts (memory): " + " | ".join(memory) if memory else ""))
    prompt = FAST_WRITE.format(show=SHOW, host=st.host_name, expert=st.expert_name, words=words,
                               minutes=round(words / WPM, 1), tone=TONES[st.tone], title=ctx.source.title,
                               passages=ctx.passages_text(pids, limit_words=PASSAGE_WORDS + 400), extra=extra,
                               language_rule=languages.writing_rule(st.language))
    ctx.status(STAGE, "Writing every part from the plan (1 model call)", 0.2)
    async with ctx.creeping(0.2, 0.4, expected_s=25):
        data = await chat_json([{"role": "user", "content": prompt}], task="write_episode", expect=dict,
                               temperature=0.8, max_tokens=3500, thinking=get_settings().llm_write_thinking or None)
    parts = [p for p in (data.get("parts") or []) if isinstance(p, dict)] if isinstance(data, dict) else []
    if not parts:
        raise ServiceError("Language model", "did not return the episode in the expected format")
    ctx.episode.learn_line = str(data.get("learn_line") or "").strip() or ctx.episode.learn_line
    for seg, p in zip(segments, parts):
        try:
            turns = _part_turns(p, names)
        except WriterError as e:
            ctx.emit(STAGE, "warning", f"Part {seg.index + 1} came back unusable ({e}).", seg.id)
            continue
        seg.chunk_ids = list(dict.fromkeys(seg.chunk_ids + [c for c in (p.get("chunk_ids") or []) if c in ctx.by_id]))
        seg.candidates = [Candidate(index=0, round=0, turns=turns)]
        seg.status = "writing"
    total = sum(len(t.text.split()) for s in segments for c in s.candidates[:1] for t in c.turns)
    ctx.emit(STAGE, "action", f"Wrote all {len(parts)} parts in one call following the plan: {total} words "
             f"(~{total / WPM:.1f} min).", data={"parts": [s.title for s in segments]})


async def critique_all(ctx: AgentContext, segments: list[Segment], cands: dict[str, Candidate]) -> None:
    """One call grades every given part on the 5-axis rubric; rule checks add penalties (no LLM)."""
    st = ctx.episode.settings
    names = (st.host_name, st.expert_name)
    per_part = max(60, target_words(ctx) // max(1, len(segments)))
    rule = {s.id: deterministic_issues(s, cands[s.id].turns, per_part) for s in segments}
    script = "\n\n".join(f"PART {s.index + 1}: {s.title}\n" + script_text(cands[s.id].turns, names) for s in segments)
    auto = "; ".join(f"part {s.index + 1}: " + "; ".join(rule[s.id][0]) for s in segments if rule[s.id][0]) or "none"
    cids = list(dict.fromkeys(c for s in segments for c in s.chunk_ids))
    by_part: dict[int, dict] = {}
    try:
        data = await chat_json([{"role": "system", "content": "You are a strict editor for an educational podcast and an experienced NCERT teacher."},
                                {"role": "user", "content": CRITIC_ALL.format(plan=_plan_text(segments), script=script, auto_issues=auto,
                                                                              passages=ctx.passages_text(cids, limit_words=1600))}],
                               task="critic_all", temperature=0.1, max_tokens=1800)
        for p in data.get("parts", []) if isinstance(data, dict) else []:
            if isinstance(p, dict):
                try:
                    by_part[int(p.get("part", 0))] = p
                except (TypeError, ValueError):
                    pass
    except Exception as e:  # noqa: BLE001 - the critic is advisory: rule checks still score the drafts
        ctx.emit(STAGE, "warning", f"Critic unavailable ({type(e).__name__}); scoring with the rule checks only.")
    for s in segments:
        issues, penalty = rule[s.id]
        p = by_part.get(s.index + 1, {})
        sc = p.get("scores") if isinstance(p.get("scores"), dict) else {}
        axes = {k: _num(sc.get(k)) for k in WEIGHTS}
        unsupported = [str(u) for u in p.get("unsupported", []) if str(u).strip()][:5]
        if unsupported:
            issues = issues + ["judge flagged unsupported: " + " | ".join(u[:100] for u in unsupported)]
            axes["faithfulness"] = min(axes["faithfulness"], 3.0)
        missing = [str(u) for u in p.get("missing_facts", []) if str(u).strip()][:5]
        if missing:
            issues = issues + ["judge says missing: " + " | ".join(m[:80] for m in missing)]
        overall = min(sum(WEIGHTS[k] * axes[k] for k in WEIGHTS), axes["faithfulness"] + 1.0) - penalty
        feedback = str(p.get("feedback") or "").strip()
        if issues:
            feedback = (feedback + "\nAlso fix: " + "; ".join(issues)).strip()
        c = cands[s.id]
        c.scores = Scores(**axes, overall=round(max(1.0, overall), 2), deterministic_penalty=round(penalty, 2))
        c.feedback, c.issues = feedback or "No major issues.", issues


async def refine(ctx: AgentContext, segments: list[Segment]) -> None:
    """Critic on every part (1 call); revise the weakest parts below the bar from the critic's note
    (<= 2 calls, as many as the budget allows); re-grade the revisions (1 call) and keep the better."""
    s = get_settings()
    ready = [x for x in segments if x.candidates]
    if not ready:
        return
    await critique_all(ctx, ready, {x.id: x.candidates[0] for x in ready})
    ctx.emit(STAGE, "observation", "Critic (1 call): " + ", ".join(
        f"part {x.index + 1} {x.candidates[0].scores.overall:.2f}" for x in ready),
        data={x.id: x.candidates[0].scores.model_dump() for x in ready})
    weak = sorted((x for x in ready if x.candidates[0].scores.overall < s.accept_score),
                  key=lambda x: (x.candidates[0].scores.overall, -x.difficulty))
    room = calls_left(ctx) - 2  # keep 1 call to re-grade revisions and 1 for the fact-check
    n = max(0, min(MAX_REVISIONS, len(weak), room))
    if not weak:
        ctx.emit(STAGE, "decision", f"Every part meets the bar ({s.accept_score}); no revision needed.")
        return
    if n == 0:
        ctx.emit(STAGE, "decision", "LLM budget for this episode is used up: keeping the first drafts.")
        return
    chosen = weak[:n]
    ctx.emit(STAGE, "decision", f"Spending {n} revision(s) where they help most (after Snell et al.): "
             + ", ".join(f"part {x.index + 1} ({x.candidates[0].scores.overall:.2f}, difficulty {x.difficulty})" for x in chosen)
             + (f"; {len(weak) - n} other part(s) below the bar kept for budget" if len(weak) > n else ""))

    async def one(x: Segment) -> Candidate | None:
        first = x.candidates[0]
        lesson = lesson_from(first.feedback or "", first.issues)
        x.reflections.append(lesson)
        ctx.emit(STAGE, "thought", f"Reflection stored in memory: {lesson}", x.id)
        try:
            turns = await revise(ctx, x, segments, first.turns, first.feedback or "Make it more natural and complete.")
        except (WriterError, ServiceError) as e:
            ctx.emit(STAGE, "warning", f"Revision of part {x.index + 1} failed: {getattr(e, 'user_message', e)}", x.id)
            return None
        cand = Candidate(index=len(x.candidates), round=1, turns=turns)
        x.candidates.append(cand)
        ctx.emit(STAGE, "action", f"Revised part {x.index + 1} using the critic's feedback", x.id)
        return cand

    revised = {x.id: c for x, c in zip(chosen, await asyncio.gather(*(one(x) for x in chosen))) if c}
    if not revised:
        return
    if calls_left(ctx) >= 2:
        regrade = [x for x in chosen if x.id in revised]
        await critique_all(ctx, regrade, revised)
        for x in regrade:
            old, new = x.candidates[0], revised[x.id]
            keep = new if new.scores.overall >= old.scores.overall else old
            if keep is old:  # fact-check and recording use the last candidate
                x.candidates.append(old.model_copy(update={"index": len(x.candidates)}))
            ctx.emit(STAGE, "decision", f"Part {x.index + 1}: revision {new.scores.overall:.2f} vs first draft "
                     f"{old.scores.overall:.2f}; keeping the {'revision' if keep is new else 'first draft'}.", x.id)
    else:
        ctx.emit(STAGE, "decision", "No budget left to re-grade; keeping the revisions (they follow the critic's notes).")


async def write_and_refine(ctx: AgentContext, segments: list[Segment]) -> None:
    await write_from_plan(ctx, segments)
    ctx.status(STAGE, "The critic is grading the parts", 0.42)
    await refine(ctx, segments)
    for x in segments:
        if x.candidates:
            x.status = "checking"
        else:
            x.status, x.error = "failed", "The writer returned nothing usable for this part."
