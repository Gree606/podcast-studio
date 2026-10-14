"""ReAct planner (Yao et al.): interleaved Thought -> Action -> Observation steps.

The model never sees the whole chapter. It must explore it with tools
(list_sections / search / read / define), and every key fact in the final outline
must cite a passage. After `finish`, facts go through dual-layer verification
(tools.rule_check_fact + tools.model_check_facts, after ToolACE); unsupported facts
are dropped and logged in the trace.
"""
from __future__ import annotations

import json
import re

from ..clients.base import ServiceError
from ..clients.llm import chat
from ..config import get_settings
from ..jsonutil import JSONExtractError, extract_json
from ..prompts import PLANNER_START, PLANNER_SYSTEM
from ..schemas import KeyFact, Segment, new_id
from .context import AgentContext
from .tools import TOOLS, model_check_facts, rule_check_fact

STAGE = "plan"


def heuristic_difficulty(text: str) -> float:
    """Cheap, model-free difficulty prior (1-5) from symbol/number/term density."""
    words = text.split() or ["x"]
    nums = len(re.findall(r"\d", text)) / len(words)
    symbols = len(re.findall(r"[=+→⟶×÷^²³°]|\b[A-Z][a-z]?\d", text)) / len(words)
    longw = sum(1 for w in words if len(w) >= 11) / len(words)
    score = 1.5 + 12 * symbols + 3 * nums + 10 * longw
    return max(1.0, min(5.0, score))


async def run_planner(ctx: AgentContext) -> list[Segment]:
    s = get_settings()
    st = ctx.episode.settings
    n_seg = ctx.n_segments
    section_titles: list[str] = []
    in_scope = {c.id for c in ctx.chunks}
    for sec in ctx.source.sections:
        ids = [c for c in sec.chunk_ids if c in in_scope]
        if ids:
            section_titles.append(f"- {sec.title} ({len(ids)} passages: {ids[0]}..{ids[-1]})")
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM.replace("{n_segments}", str(n_seg)).replace("{max_steps}", str(s.planner_max_steps))},
        {"role": "user", "content": PLANNER_START.format(
            title=ctx.source.title, n_chunks=len(ctx.chunks), n_words=sum(len(c.text.split()) for c in ctx.chunks),
            section_list="\n".join(section_titles), length=st.length, n_segments=n_seg)},
    ]
    outline: dict | None = None
    read_ids: set[str] = set()
    for step in range(1, s.planner_max_steps + 3):
        force = step > s.planner_max_steps
        if force:
            messages.append({"role": "user", "content": "You are out of tool calls. Call finish now with the complete outline."})
        ctx.status(STAGE, f"Planning: step {step}", 0.02 + 0.10 * min(step, s.planner_max_steps) / s.planner_max_steps)
        raw = await chat(messages, task="planner_step", temperature=0.3, max_tokens=2500)
        try:
            act = extract_json(raw, expect=dict)
        except JSONExtractError:
            messages += [{"role": "assistant", "content": raw[:2000]},
                         {"role": "user", "content": "Observation: invalid format. Reply with ONLY a JSON object with thought, action, input."}]
            ctx.emit(STAGE, "warning", "Planner produced malformed output; asked it to retry.")
            continue
        thought = str(act.get("thought", "")).strip()
        action = str(act.get("action", "")).strip().lower()
        inp = act.get("input") if isinstance(act.get("input"), dict) else {}
        if thought:
            ctx.emit(STAGE, "thought", thought)
        messages.append({"role": "assistant", "content": json.dumps(act, ensure_ascii=False)[:3000]})
        if action == "finish":
            outline = inp.get("outline") if isinstance(inp.get("outline"), dict) else inp
            ctx.emit(STAGE, "action", "finish(outline)")
            if outline and outline.get("segments"):
                break
            messages.append({"role": "user", "content": "Observation: the outline was empty or missing 'segments'. Call finish again with the full outline."})
            outline = None
            continue
        tool = TOOLS.get(action)
        if tool is None:
            obs = f"Error: unknown tool '{action}'. Available: {', '.join(TOOLS)} or finish."
        else:
            obs = tool(ctx, inp)
            read_ids.update(re.findall(r"\[(c\d+)\]", obs))
        ctx.emit(STAGE, "action", f"{action}({json.dumps(inp, ensure_ascii=False)[:120]})")
        ctx.emit(STAGE, "observation", obs[:600] + ("..." if len(obs) > 600 else ""))
        messages.append({"role": "user", "content": f"Observation:\n{obs}"})

    if not outline:
        ctx.emit(STAGE, "warning", "Planner did not finish; using a section-based outline instead.")
        return fallback_outline(ctx)
    return await _outline_to_segments(ctx, outline, read_ids)


async def _outline_to_segments(ctx: AgentContext, outline: dict, read_ids: set[str]) -> list[Segment]:
    ctx.episode.learn_line = str(outline.get("learn_line") or "").strip() or None
    raw_segs = [x for x in outline.get("segments", []) if isinstance(x, dict)][: ctx.n_segments + 1]
    segments: list[Segment] = []
    all_facts: list[KeyFact] = []
    owners: list[int] = []
    for i, rs in enumerate(raw_segs):
        facts = []
        for f in rs.get("key_facts", []) or []:
            if isinstance(f, dict) and f.get("fact"):
                facts.append(KeyFact(fact=str(f["fact"]), chunk_id=str(f.get("chunk_id") or "") or None))
            elif isinstance(f, str):
                facts.append(KeyFact(fact=f))
        chunk_ids = [c for c in (rs.get("chunk_ids") or []) if c in ctx.by_id]
        try:
            llm_diff = float(rs.get("difficulty", 3))
        except (TypeError, ValueError):
            llm_diff = 3.0
        seg = Segment(id=new_id("seg_"), index=i, title=str(rs.get("title") or f"Part {i + 1}")[:80],
                      goal=str(rs.get("goal") or ""), key_facts=facts, chunk_ids=chunk_ids,
                      difficulty=int(round(max(1, min(5, llm_diff)))))
        seg_reason = rs.get("difficulty_reason")
        if seg_reason:
            ctx.emit(STAGE, "info", f"Difficulty {seg.difficulty}/5 for '{seg.title}': {seg_reason}", seg.id)
        segments.append(seg)
        for f in facts:
            all_facts.append(f)
            owners.append(i)

    # --- dual-layer verification -------------------------------------------
    ctx.status(STAGE, "Verifying the outline against the chapter", 0.13)
    checked = [rule_check_fact(ctx, f) for f in all_facts]
    n_rule_fail = sum(1 for f in checked if f.verified is False)
    checked = await model_check_facts(ctx, checked)
    n_model_fail = sum(1 for f in checked if f.verified is False) - n_rule_fail
    for seg in segments:
        seg.key_facts = []
    dropped = []
    for f, i in zip(checked, owners):
        if f.verified is False:
            dropped.append(f.fact)
        else:
            segments[i].key_facts.append(f)
    ctx.emit(STAGE, "decision",
             f"Verified {len(checked)} key facts: {n_rule_fail} failed the rule check, {n_model_fail} failed the model check; "
             f"{len(dropped)} dropped.", data={"dropped": dropped})

    for seg in segments:
        # make sure each segment's passages include every cited fact's passage
        for f in seg.key_facts:
            if f.chunk_id and f.chunk_id not in seg.chunk_ids:
                seg.chunk_ids.append(f.chunk_id)
        if not seg.chunk_ids:
            seg.chunk_ids = [c.id for c, _ in ctx.index.search(seg.title + " " + seg.goal, k=3)]
        # blend the planner's estimate with a model-free prior
        prior = heuristic_difficulty(" ".join(ctx.by_id[c].text for c in seg.chunk_ids if c in ctx.by_id))
        seg.difficulty = int(round(max(1, min(5, 0.65 * seg.difficulty + 0.35 * prior))))
    segments = [s for s in segments if s.key_facts or s.chunk_ids]
    if not segments:
        return fallback_outline(ctx)
    _assign_kinds(segments)
    return segments


def _assign_kinds(segments: list[Segment]) -> None:
    for i, seg in enumerate(segments):
        seg.index = i
        seg.kind = "intro" if i == 0 else ("recap" if i == len(segments) - 1 and len(segments) > 1 else "body")


def fallback_outline(ctx: AgentContext) -> list[Segment]:
    """Deterministic outline from section structure, used if the LLM planner fails."""
    n = ctx.n_segments
    chunks = ctx.chunks
    per = max(1, len(chunks) // n)
    segments = []
    for i in range(n):
        group = chunks[i * per: (i + 1) * per] if i < n - 1 else chunks[i * per:]
        if not group:
            break
        sec_title = group[0].section
        facts = []
        for c in group[:4]:
            first = re.split(r"(?<=[.!?])\s+", c.text)[0]
            facts.append(KeyFact(fact=first[:240], chunk_id=c.id, verified=True, verify_note="taken verbatim"))
        text = " ".join(c.text for c in group)
        segments.append(Segment(id=new_id("seg_"), index=i, title=sec_title[:80], goal=f"Explain {sec_title}.",
                                key_facts=facts, chunk_ids=[c.id for c in group],
                                difficulty=int(round(heuristic_difficulty(text)))))
    _assign_kinds(segments)
    return segments


async def plan(ctx: AgentContext) -> list[Segment]:
    try:
        return await run_planner(ctx)
    except ServiceError as e:
        ctx.emit(STAGE, "warning", f"Planner unavailable ({e.user_message}); using a section-based outline.")
        return fallback_outline(ctx)
