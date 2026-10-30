"""Tools the planner can call, plus ToolACE-style dual-layer verification of
their results (a rule-based layer, then a model-based layer).

Tools are plain functions over the chapter, so observations are deterministic
and cheap; the LLM decides *which* tool to call and with what input.
"""
from __future__ import annotations

import re

from ..clients.llm import chat_json
from ..prompts import FACT_VERIFY
from ..retrieval import overlap
from ..schemas import KeyFact
from .context import AgentContext

OBS_CHARS = 1800  # keep observations short: the served context is ~20k tokens


def sections_in_scope(ctx: AgentContext) -> list[tuple[str, list[str]]]:
    """(section title, its in-scope chunk ids) for every section with at least one passage in scope."""
    in_scope = {c.id for c in ctx.chunks}
    out = []
    for s in ctx.source.sections:
        ids = [c for c in s.chunk_ids if c in in_scope]
        if ids:
            out.append((s.title, ids))
    return out


def tool_list_sections(ctx: AgentContext, _: dict) -> str:
    lines = []
    for title, ids in sections_in_scope(ctx):
        first = ctx.by_id[ids[0]].text[:90].replace("\n", " ")
        lines.append(f"- {title} | passages {ids[0]}..{ids[-1]} ({len(ids)}) | starts: \"{first}...\"")
    return "\n".join(lines)


def tool_search(ctx: AgentContext, inp: dict) -> str:
    q = str(inp.get("query") or inp.get("q") or "").strip()
    if not q:
        return "Error: search needs a non-empty 'query'."
    hits = ctx.index.search(q, k=4)
    if not hits:
        return f"No passages match '{q}'. Try other words, or list_sections."
    out = [f"[{c.id}] ({c.section}) {c.text[:380]}{'...' if len(c.text) > 380 else ''}" for c, _ in hits]
    return "\n".join(out)[:OBS_CHARS]


def tool_read(ctx: AgentContext, inp: dict) -> str:
    """Read one passage or several at once ({"chunk_ids": ["c4", "c5"]}). Batching matters with a
    real model: reading a chapter one passage per step took 14+ slow steps."""
    ids = inp.get("chunk_ids") or inp.get("ids")
    if not isinstance(ids, list):
        ids = [inp.get("chunk_id") or inp.get("id") or ""]
    ids = [str(i).strip() for i in ids][:4]
    out, missing, budget = [], [], OBS_CHARS * 2
    for cid in ids:
        c = ctx.by_id.get(cid)
        if not c:
            missing.append(cid)
            continue
        out.append(f"[{c.id}] ({c.section}) {c.text}"[: budget // max(1, len(ids))])
    if missing and not out:
        return f"Error: no passage {', '.join(repr(m) for m in missing)}. Passage ids look like c12; use search or list_sections."
    if missing:
        out.append(f"(not found: {', '.join(missing)})")
    return "\n\n".join(out)


_DEF_PATTERNS = [
    r"{t}\s+(?:is|are)\s+(?:called|known as|defined as|termed)\b",
    r"(?:is|are)\s+(?:called|known as|termed|referred to as)\s+(?:an?\s+|the\s+)?{t}\b",
    r"{t}\s*(?:is|are|refers to|means)\s+",
    r"{t}\s*[:–-]\s+",
]


def tool_define(ctx: AgentContext, inp: dict) -> str:
    term = str(inp.get("term") or "").strip()
    if not term:
        return "Error: define needs a 'term'."
    low = term.lower()
    patterns = [re.compile(p.format(t=re.escape(low))) for p in _DEF_PATTERNS]  # compiled once, not per sentence
    best: list[tuple[int, str, str]] = []
    for c in ctx.source.chunks:
        for sent in re.split(r"(?<=[.!?])\s+", c.text):
            sl = sent.lower()
            if low not in sl:
                continue
            for rank, pat in enumerate(patterns):
                if pat.search(sl):
                    best.append((rank, c.id, sent.strip()))
                    break
    if not best:
        hits = ctx.full_index.search(term, k=2)
        if not hits:
            return f"The chapter does not mention '{term}'."
        return f"No explicit definition of '{term}' in the chapter. Closest passages: " + "; ".join(
            f"[{c.id}] {c.text[:200]}" for c, _ in hits)
    best.sort()
    return "\n".join(f"[{cid}] {s}" for _, cid, s in best[:3])[:OBS_CHARS]


TOOLS = {
    "list_sections": tool_list_sections,
    "search": tool_search,
    "read": tool_read,
    "define": tool_define,
}


# ---------------------------------------------------------------------------
# Dual-layer verification of the planner's key facts
# ---------------------------------------------------------------------------
def rule_check_fact(ctx: AgentContext, kf: KeyFact) -> KeyFact:
    """Layer 1 (rule-based): the cited passage exists and lexically supports the fact.
    If the citation is wrong, re-attach the fact to the best-matching passage."""
    cited = ctx.by_id.get(kf.chunk_id or "")
    if cited and overlap(kf.fact, cited.text) >= 0.5:
        return kf
    for c, _ in ctx.full_index.search(kf.fact, k=3):
        if overlap(kf.fact, c.text) >= 0.5:
            return kf.model_copy(update={"chunk_id": c.id, "verify_note": f"citation corrected {kf.chunk_id}->{c.id}"})
    return kf.model_copy(update={"verified": False, "verify_note": "no passage lexically supports this fact"})


async def model_check_facts(ctx: AgentContext, facts: list[KeyFact]) -> list[KeyFact]:
    """Layer 2 (model-based): an LLM judge reads fact + cited passage and says supported or not."""
    todo = [(i, f) for i, f in enumerate(facts) if f.verified is not False]
    if not todo:
        return facts
    items = "\n\n".join(
        f"{n}. FACT: {f.fact}\n   PASSAGE [{f.chunk_id}]: {ctx.by_id[f.chunk_id].text[:700]}"
        for n, (_, f) in enumerate(todo, 1) if f.chunk_id in ctx.by_id
    )
    try:
        res = await chat_json([{"role": "user", "content": FACT_VERIFY.format(items=items)}], task="fact_verify",
                              temperature=0.0, max_tokens=1200)
        verdicts = {int(r.get("i", 0)): r for r in res.get("results", []) if isinstance(r, dict)}
    except Exception:
        verdicts = {}
    out = list(facts)
    for n, (i, f) in enumerate(todo, 1):
        v = verdicts.get(n)
        if v is None:
            out[i] = f.model_copy(update={"verified": f.chunk_id in ctx.by_id,
                                          "verify_note": f.verify_note or "rule-checked only"})
        else:
            ok = bool(v.get("supported"))
            out[i] = f.model_copy(update={"verified": ok, "verify_note": v.get("note") or f.verify_note})
    return out
