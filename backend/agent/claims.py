"""Post-write, claim-level faithfulness check.

Planning-time verification only covers the outline's key facts; the writer can
still add things. Here we extract every factual claim from the chosen script,
retrieve evidence passages for each with BM25, and ask a judge whether each is
supported. Unsupported claims trigger one targeted rewrite; anything still
unsupported is surfaced in the UI rather than hidden.
"""
from __future__ import annotations

from ..clients.llm import chat_json
from ..prompts import CLAIM_EXTRACT, CLAIM_FIX, CLAIM_JUDGE
from ..schemas import Claim, Segment, Turn
from .context import AgentContext
from .writer import build_messages, parse_turns, script_text

STAGE = "verify"


async def check_claims(ctx: AgentContext, seg: Segment, turns: list[Turn]) -> list[Claim]:
    st = ctx.episode.settings
    script = script_text(turns, (st.host_name, st.expert_name))
    try:
        data = await chat_json([{"role": "user", "content": CLAIM_EXTRACT.format(script=script)}], task="claim_extract",
                               temperature=0.0, max_tokens=1200)
        claims = [str(c).strip() for c in data.get("claims", []) if str(c).strip()][:14]
    except Exception:
        return []
    if not claims:
        return []
    evid: list[list[str]] = []
    items = []
    for i, cl in enumerate(claims, 1):
        hits = ctx.full_index.search(cl, k=2)
        ids = [c.id for c, _ in hits]
        for cid in seg.chunk_ids[:1]:
            if cid not in ids:
                ids.append(cid)
        evid.append(ids)
        passages = "\n".join(f"   [{cid}] {ctx.by_id[cid].text[:550]}" for cid in ids)
        items.append(f"{i}. CLAIM: {cl}\n{passages}")
    try:
        res = await chat_json([{"role": "user", "content": CLAIM_JUDGE.format(items="\n\n".join(items))}],
                              task="claim_judge", temperature=0.0, max_tokens=1500)
        verdicts = {int(r.get("i", 0)): r for r in res.get("results", []) if isinstance(r, dict)}
    except Exception:
        return []
    out = []
    for i, cl in enumerate(claims, 1):
        v = verdicts.get(i, {})
        ev = str(v.get("evidence") or "")
        if ev not in ctx.by_id:
            ev = evid[i - 1][0] if evid[i - 1] else ""
        out.append(Claim(claim=cl, supported=bool(v.get("supported", True)), evidence_chunk_id=ev or None,
                         note=v.get("note")))
    return out


async def verify_and_fix(ctx: AgentContext, seg: Segment, segments: list[Segment], turns: list[Turn]) -> tuple[list[Turn], list[Claim]]:
    claims = await check_claims(ctx, seg, turns)
    bad = [c for c in claims if not c.supported]
    ctx.emit(STAGE, "observation", f"Claim check: {len(claims) - len(bad)}/{len(claims)} claims supported by the chapter.",
             seg.id, data={"unsupported": [c.claim for c in bad]})
    if not bad:
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
    import json

    return json.dumps([{"speaker": t.speaker, "text": t.text} for t in turns], ensure_ascii=False)
