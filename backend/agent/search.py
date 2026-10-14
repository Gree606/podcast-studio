"""Per-segment generate -> verify -> reflect -> revise loop.

Combines Reflexion (verbal feedback distilled into an episodic memory that
conditions later attempts) with best-of-N selection using the critic as the
verifier. The number and shape of attempts come from agent/allocation.py.
"""
from __future__ import annotations

import asyncio
import random

from ..clients.base import ServiceError
from ..schemas import Candidate, Segment, Turn
from .allocation import search_shape
from .context import AgentContext
from .critic import critique, reflect
from .writer import WriterError, revise, write

STAGE = "write"


def best_candidate(seg: Segment) -> Candidate | None:
    scored = [c for c in seg.candidates if c.turns]
    if not scored:
        return None
    return max(scored, key=lambda c: (c.scores.overall if c.scores else 0.0, -c.index))


async def _score(ctx: AgentContext, seg: Segment, cand: Candidate, use_critic: bool, accept: float) -> None:
    if not use_critic:
        return
    scores, feedback, issues = await critique(ctx, seg, cand.turns)
    cand.scores, cand.feedback, cand.issues = scores, feedback, issues
    ctx.emit(STAGE, "observation",
             f"Critic scored draft {cand.index + 1}: {scores.overall:.2f}/5 (faith {scores.faithfulness:.0f}, "
             f"cover {scores.coverage:.0f}, natural {scores.naturalness:.0f}, engage {scores.engagement:.0f}, "
             f"clarity {scores.clarity:.0f}; rule penalty {scores.deterministic_penalty})", seg.id,
             data={"scores": scores.model_dump(), "feedback": feedback})
    if scores.overall < accept:
        lesson = await reflect(feedback, issues)
        seg.reflections.append(lesson)
        ctx.emit(STAGE, "thought", f"Reflection stored in memory: {lesson}", seg.id)


async def run_search(ctx: AgentContext, seg: Segment, segments: list[Segment], budget: int, *, policy: str,
                     accept: float, instruction: str | None = None, current: list[Turn] | None = None) -> int:
    """Spend up to ``budget`` writer calls on ``seg``. Returns calls actually spent (early stop saves budget)."""
    use_critic = policy != "no_critic"
    early_stop = policy == "adaptive"
    n_par, n_rev = search_shape(budget, seg.difficulty, policy)
    if seg.candidates:  # second-round reallocation: these are extra attempts on an existing pool
        n_par, n_rev = (0, budget) if seg.difficulty <= 3 else (budget - budget // 2, budget // 2)
    ctx.emit(STAGE, "decision", f"Budget {budget} for '{seg.title}' (difficulty {seg.difficulty}/5): "
             f"{n_par} parallel draft(s) + {n_rev} revision(s)", seg.id)
    spent = 0

    async def one_draft(k: int) -> None:
        nonlocal spent
        spent += 1
        temp = 0.8 + 0.15 * random.random() if k else 0.8
        try:
            turns = await write(ctx, seg, segments, temperature=temp, instruction=instruction, current=current)
        except (WriterError, ServiceError) as e:
            ctx.emit(STAGE, "warning", f"Draft failed: {getattr(e, 'user_message', e)}", seg.id)
            return
        cand = Candidate(index=len(seg.candidates), round=0, turns=turns)
        seg.candidates.append(cand)
        ctx.emit(STAGE, "action", f"Writer produced draft {cand.index + 1} ({sum(len(t.text.split()) for t in turns)} words, {len(turns)} turns)", seg.id)
        await _score(ctx, seg, cand, use_critic, accept)

    if n_par:
        await asyncio.gather(*(one_draft(k) for k in range(n_par)))
    for r in range(n_rev):
        best = best_candidate(seg)
        if best is None:
            await one_draft(r + n_par)
            continue
        if early_stop and best.scores and best.scores.overall >= accept:
            ctx.emit(STAGE, "decision", f"Best draft scores {best.scores.overall:.2f} >= {accept}; stopping early "
                     f"and saving {n_rev - r} call(s) of budget.", seg.id)
            break
        spent += 1
        try:
            turns = await revise(ctx, seg, segments, best.turns, best.feedback or "Make it more natural and complete.")
        except (WriterError, ServiceError) as e:
            ctx.emit(STAGE, "warning", f"Revision failed: {getattr(e, 'user_message', e)}", seg.id)
            continue
        cand = Candidate(index=len(seg.candidates), round=r + 1, turns=turns)
        seg.candidates.append(cand)
        ctx.emit(STAGE, "action", f"Revised draft {best.index + 1} using critic feedback -> draft {cand.index + 1}", seg.id)
        await _score(ctx, seg, cand, use_critic, accept)
    if best_candidate(seg) is None and spent < budget + 1:
        # last resort: one low-temperature attempt so a single flaky call doesn't lose the segment
        spent += 1
        try:
            turns = await write(ctx, seg, segments, temperature=0.4, instruction=instruction, current=current)
            cand = Candidate(index=len(seg.candidates), round=0, turns=turns)
            seg.candidates.append(cand)
            await _score(ctx, seg, cand, use_critic, accept)
        except (WriterError, ServiceError) as e:
            ctx.emit(STAGE, "error", f"Could not write this segment: {getattr(e, 'user_message', e)}", seg.id)
    return spent
