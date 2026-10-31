"""Post-write, claim-level faithfulness check.

Planning-time verification only covers the outline's key facts; the writer can
still add things. Here passages are retrieved for every spoken line (BM25 + embeddings),
and one judge call lists the script's factual claims and says whether each is
supported. Unsupported claims trigger one targeted rewrite; anything still
unsupported is surfaced in the UI rather than hidden.
"""
from __future__ import annotations

import asyncio
import json

from ..clients.llm import chat_json
from ..prompts import CLAIM_CHECK, CLAIM_FIX
from ..schemas import Claim, Segment, Turn
from .context import AgentContext
from .writer import build_messages, parse_turns

STAGE = "verify"


MAX_PASSAGES = 8


async def check_claims(ctx: AgentContext, seg: Segment, turns: list[Turn]) -> list[Claim]:
    """One LLM call per part: the model lists the script's factual claims and judges each against
    passages retrieved beforehand for every spoken line (hybrid search, no LLM). Extracting the
    claims first and judging them in a second call cost twice as much for the same verdicts."""
    st = ctx.episode.settings
    names = {"host": st.host_name, "expert": st.expert_name}
    lines = [f"{n}. {names.get(t.speaker, t.speaker)}: {t.text}" for n, t in enumerate(turns, 1)]
    spoken = [t.text for t in turns if len(t.text.split()) >= 6]
    found = await asyncio.gather(*(ctx.full_index.asearch(text, k=2) for text in spoken))
    votes: dict[str, int] = {}
    for cid in seg.chunk_ids:
        votes[cid] = votes.get(cid, 0) + 2  # the passages this part was written from
    for hits in found:
        for c, _ in hits:
            votes[c.id] = votes.get(c.id, 0) + 1
    ids = [cid for cid, _ in sorted(votes.items(), key=lambda x: -x[1]) if cid in ctx.by_id][:MAX_PASSAGES]
    if not ids:
        return []
    passages = "\n".join(f"[{cid}] {ctx.by_id[cid].text[:550]}" for cid in ids)
    try:
        data = await chat_json([{"role": "user", "content": CLAIM_CHECK.format(script="\n".join(lines), passages=passages)}],
                               task="claim_check", temperature=0.0, max_tokens=2400)
    except Exception:
        return []
    out = []
    for r in (data.get("claims") or [])[:14]:
        if not isinstance(r, dict) or not str(r.get("claim") or "").strip():
            continue
        ev = str(r.get("evidence") or "")
        out.append(Claim(claim=str(r["claim"]).strip(), supported=bool(r.get("supported", True)),
                         evidence_chunk_id=ev if ev in ctx.by_id else ids[0], note=r.get("note") or None))
    return out


async def verify_and_fix(ctx: AgentContext, seg: Segment, segments: list[Segment], turns: list[Turn]) -> tuple[list[Turn], list[Claim]]:
    claims = await check_claims(ctx, seg, turns)
    if hasattr(ctx, "claims_checked"):
        ctx.claims_checked.add(seg.id)
    bad = [c for c in claims if not c.supported]
    ctx.emit(STAGE, "observation", f"Claim check: {len(claims) - len(bad)}/{len(claims)} claims supported by the chapter.",
             seg.id, data={"unsupported": [c.claim for c in bad]})
    if not bad:
        return turns, claims
    if hasattr(ctx, "can_spend") and not ctx.can_spend(2):  # fix + re-check
        ctx.emit(STAGE, "decision", f"LLM budget for this episode is used up: flagging {len(bad)} unsupported "
                 "claim(s) for the listener instead of rewriting.", seg.id)
        return turns, claims
    ctx.emit(STAGE, "action", "Rewriting to remove/correct: " + " | ".join(c.claim[:90] for c in bad[:4]), seg.id)
    st = ctx.episode.settings
    msgs = build_messages(ctx, seg, segments)
    msgs.append({"role": "assistant", "content": '{"turns": ' + _turns_json(turns) + "}"})
    msgs.append({"role": "user", "content": CLAIM_FIX.format(claims="; ".join(f'"{c.claim}"' for c in bad))})
    try:
        data = await chat_json(msgs, task="claim_fix", expect=None, temperature=0.3, max_tokens=2600)
        fixed = parse_turns(data, st.host_name, st.expert_name)
    except Exception as e:  # includes WriterError / ServiceError
        ctx.emit(STAGE, "warning", f"Fix attempt failed ({type(e).__name__}); keeping the draft and flagging claims.", seg.id)
        return turns, claims
    claims2 = await check_claims(ctx, seg, fixed)
    bad2 = [c for c in claims2 if not c.supported]
    if claims2 and len(bad2) < len(bad):
        ctx.emit(STAGE, "decision", f"Fixed version has {len(bad2)} unsupported claim(s) (was {len(bad)}); keeping it.", seg.id)
        return fixed, claims2
    ctx.emit(STAGE, "decision", "Fix did not help; keeping the original and flagging the claims for the listener.", seg.id)
    return turns, claims


def _turns_json(turns: list[Turn]) -> str:
    return json.dumps([{"speaker": t.speaker, "text": t.text} for t in turns], ensure_ascii=False)
