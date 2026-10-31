"""Critic / verifier: a deterministic check layer plus an LLM rubric judge.

The writer and critic are the same base model, so the critic is prone to
self-preference and score inflation. Mitigations here:
  * a different persona, near-zero temperature and anchored 1-5 rubrics;
  * a deterministic layer (monologues, length, missing key-fact terms, banned
    phrasing) that subtracts from the score no matter what the LLM says;
  * the overall score caps at faithfulness+1, so a fluent but wrong draft can't win.
Multi-axis scores follow ToolRL's point that fine-grained rewards beat a coarse
pass/fail signal; here they're used for best-of-N selection, not training.
"""
from __future__ import annotations

import re

from ..clients.llm import chat, chat_json
from ..prompts import CRITIC_SYSTEM, CRITIC_USER, REFLECT
from ..retrieval import overlap
from ..schemas import Scores, Segment, Turn
from .context import AgentContext
from .writer import MAX_TURN_WORDS, script_text

BANNED = [r"\bin this segment\b", r"\bas an ai\b", r"\bin conclusion\b", r"\bdelve\b", r"\btapestry\b",
          r"\blet's dive (right )?in\b", r"[#*`]", r"\bsegment \d\b"]
WEIGHTS = {"faithfulness": 0.32, "coverage": 0.2, "naturalness": 0.2, "engagement": 0.14, "clarity": 0.14}


def deterministic_issues(seg: Segment, turns: list[Turn], target_words: int) -> tuple[list[str], float]:
    """Rule-based checks. Returns human-readable issues and a score penalty (0..2)."""
    issues: list[str] = []
    penalty = 0.0
    words = [len(t.text.split()) for t in turns]
    total = sum(words)
    long_turns = [i + 1 for i, w in enumerate(words) if w > MAX_TURN_WORDS]
    if long_turns:
        issues.append(f"monologue: turns {long_turns} exceed {MAX_TURN_WORDS} words")
        penalty += min(0.8, 0.25 * len(long_turns))
    if total < 0.6 * target_words:
        issues.append(f"too short: {total} words (target ~{target_words})")
        penalty += 0.5
    elif total > 1.5 * target_words:
        issues.append(f"too long: {total} words (target ~{target_words})")
        penalty += 0.4
    short_reactions = sum(1 for w in words if w <= 8)
    if len(turns) >= 6 and short_reactions == 0:
        issues.append("no quick reactions/back-and-forth: every turn is a long speech")
        penalty += 0.3
    host_share = sum(w for w, t in zip(words, turns) if t.speaker == "host") / max(1, total)
    if host_share < 0.15:
        issues.append("the curious host barely speaks; it feels like a lecture")
        penalty += 0.3
    script = " ".join(t.text for t in turns)
    low = script.lower()
    for pat in BANNED:
        if re.search(pat, low):
            issues.append(f"banned phrasing/formatting: /{pat}/")
            penalty += 0.15
    missing = [f.fact for f in seg.key_facts if overlap(f.fact, script) < 0.35]
    if missing:
        issues.append("key facts seemingly not covered: " + " | ".join(m[:80] for m in missing[:4]))
        penalty += min(0.8, 0.25 * len(missing))
    return issues, min(2.0, penalty)


def _num(v, default: float = 3.0) -> float:
    try:
        return max(1.0, min(5.0, float(v)))
    except (TypeError, ValueError):
        return default


async def critique(ctx: AgentContext, seg: Segment, turns: list[Turn]) -> tuple[Scores, str, list[str]]:
    """Return (scores, verbal feedback, issues). Never raises: on judge failure, falls back to rules only."""
    st = ctx.episode.settings
    issues, penalty = deterministic_issues(seg, turns, ctx.words_per_segment)
    facts = "\n".join(f"- {f.fact}" for f in seg.key_facts) or "- (main ideas of the passages)"
    msgs = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": CRITIC_USER.format(
            goal=seg.goal, facts=facts, passages=ctx.passages_text(seg.chunk_ids, 1200),
            script=script_text(turns, (st.host_name, st.expert_name)),
            auto_issues="; ".join(issues) or "none", grade_hint="8-10")},
    ]
    feedback = ""
    try:
        data = await chat_json(msgs, task="critic", temperature=0.1, max_tokens=900)
        sc = data.get("scores", {}) if isinstance(data, dict) else {}
        axes = {k: _num(sc.get(k)) for k in WEIGHTS}
        unsupported = [str(u) for u in data.get("unsupported", []) if str(u).strip()][:5]
        missing = [str(u) for u in data.get("missing_facts", []) if str(u).strip()][:5]
        feedback = str(data.get("feedback", "")).strip()
        if unsupported:
            issues.append("judge flagged unsupported: " + " | ".join(u[:100] for u in unsupported))
            axes["faithfulness"] = min(axes["faithfulness"], 3.0)
        if missing:
            issues.append("judge says missing: " + " | ".join(m[:80] for m in missing))
    except Exception as e:  # judge failed: rules-only scoring keeps the pipeline moving
        axes = {k: 3.0 for k in WEIGHTS}
        issues.append(f"(rubric judge unavailable: {type(e).__name__})")
    weighted = sum(WEIGHTS[k] * axes[k] for k in WEIGHTS)
    overall = min(weighted, axes["faithfulness"] + 1.0) - penalty
    scores = Scores(**axes, overall=round(max(1.0, overall), 2), deterministic_penalty=round(penalty, 2))
    if issues:
        feedback = (feedback + "\nAlso fix: " + "; ".join(issues)).strip()
    return scores, feedback or "No major issues.", issues


def lesson_from(feedback: str, issues: list[str]) -> str:
    """Reflexion memory entry without another model call: the top automatic issue, else the
    first sentence of the critic's feedback."""
    text = issues[0] if issues else (feedback or "").strip().split("\n")[0]
    first = text.split(". ")[0].strip()
    return (first if first.endswith(".") else first + ".")[:220] if first else "Keep it faithful and conversational."


async def reflect(feedback: str, issues: list[str]) -> str:
    """Reflexion: compress feedback into a one-line lesson stored in episodic memory."""
    try:
        text = await chat([{"role": "user", "content": REFLECT.format(feedback=feedback[:1500], auto_issues="; ".join(issues) or "none")}],
                          task="reflect", temperature=0.2, max_tokens=80)
        return text.strip().strip('"')[:220]
    except Exception:
        return (issues[0] if issues else feedback.split("\n")[0])[:220]

