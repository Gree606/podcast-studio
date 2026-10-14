"""Writer: turns one outline segment into a two-host dialogue (JSON list of turns)."""
from __future__ import annotations

import re

from ..clients.llm import chat_json
from ..prompts import (POSITION_FIRST, POSITION_LAST, POSITION_MIDDLE, REGEN_INSTRUCTION, REVISE_USER, WRITER_SYSTEM,
                       WRITER_USER)
from ..schemas import Segment, Turn
from .context import AgentContext

SHOW = "Chapter Chat"
MAX_TURN_WORDS = 75
TONES = {
    "casual": "relaxed and playful, lots of banter, but still accurate",
    "balanced": "friendly and lively, with clear explanations",
    "serious": "calm and focused, fewer jokes, like a good revision session",
}


class WriterError(ValueError):
    pass


def script_text(turns: list[Turn], names: tuple[str, str] | None = None) -> str:
    h, e = names or ("HOST", "EXPERT")
    return "\n".join(f"{i}. {h if t.speaker == 'host' else e}: {t.text}" for i, t in enumerate(turns, 1))


def parse_turns(data: dict, host_name: str, expert_name: str) -> list[Turn]:
    raw = data.get("turns") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise WriterError("no 'turns' list")
    turns: list[Turn] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        sp = str(r.get("speaker", "")).strip().lower()
        text = str(r.get("text", "")).strip()
        text = re.sub(r"\[[^\]]{1,30}\]|\*[^*]{1,30}\*", "", text).strip()  # stage directions
        if not text:
            continue
        if sp in ("host", host_name.lower()):
            speaker = "host"
        elif sp in ("expert", expert_name.lower()):
            speaker = "expert"
        else:
            speaker = "expert" if turns and turns[-1].speaker == "host" else "host"
        if turns and turns[-1].speaker == speaker:  # merge same-speaker consecutive turns
            turns[-1] = Turn(speaker=speaker, text=turns[-1].text + " " + text)
        else:
            turns.append(Turn(speaker=speaker, text=text))
    if len(turns) < 4:
        raise WriterError(f"only {len(turns)} usable turns")
    return turns


def build_messages(ctx: AgentContext, seg: Segment, segments: list[Segment], *, instruction: str | None = None,
                   current: list[Turn] | None = None) -> list[dict]:
    st = ctx.episode.settings
    names = (st.host_name, st.expert_name)
    total = len(segments)
    prev = segments[seg.index - 1] if seg.index > 0 else None
    prev_last = ""
    if prev and prev.current and prev.current.turns:
        prev_last = prev.current.turns[-1].text[:200]
    elif prev:
        prev_last = f"(the previous segment, about {prev.title})"
    if seg.kind == "intro":
        pos = POSITION_FIRST.format(host=st.host_name, expert=st.expert_name, show=SHOW,
                                    learn_line=ctx.episode.learn_line or f"the key ideas of {ctx.source.title}")
    elif seg.kind == "recap":
        pos = POSITION_LAST.format(host=st.host_name, expert=st.expert_name, prev_last=prev_last,
                                   all_titles="; ".join(s.title for s in segments))
    else:
        pos = POSITION_MIDDLE.format(prev_title=prev.title if prev else "", prev_last=prev_last)
    target = ctx.words_per_segment
    facts = "\n".join(f"- {f.fact} [{f.chunk_id}]" for f in seg.key_facts) or "- (cover the passages' main ideas)"
    memory = ""
    if seg.reflections:
        memory = "\nLessons from your earlier attempts at this segment (apply them):\n" + "\n".join(
            f"- {r}" for r in seg.reflections[-4:]) + "\n"
    instr = ""
    if instruction:
        instr = REGEN_INSTRUCTION.format(instruction=instruction, current=script_text(current or [], names))
    user = WRITER_USER.format(
        index=seg.index + 1, total=total, title=seg.title, goal=seg.goal, tone=TONES[st.tone],
        target_words=target, min_turns=max(8, target // 30), max_turns=max(14, target // 16), position_rules=pos,
        facts=facts, passages=ctx.passages_text(seg.chunk_ids), memory=memory, instruction=instr)
    system = WRITER_SYSTEM.format(show=SHOW, host=st.host_name, expert=st.expert_name, max_turn_words=MAX_TURN_WORDS)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def write(ctx: AgentContext, seg: Segment, segments: list[Segment], *, temperature: float = 0.85,
                instruction: str | None = None, current: list[Turn] | None = None) -> list[Turn]:
    msgs = build_messages(ctx, seg, segments, instruction=instruction, current=current)
    st = ctx.episode.settings
    data = await chat_json(msgs, task="write", temperature=temperature, max_tokens=2600)
    return parse_turns(data, st.host_name, st.expert_name)


async def revise(ctx: AgentContext, seg: Segment, segments: list[Segment], draft: list[Turn], feedback: str,
                 *, temperature: float = 0.6) -> list[Turn]:
    st = ctx.episode.settings
    msgs = build_messages(ctx, seg, segments)
    msgs.append({"role": "user", "content": REVISE_USER.format(
        draft=script_text(draft, (st.host_name, st.expert_name)), feedback=feedback)})
    data = await chat_json(msgs, task="revise", temperature=temperature, max_tokens=2600)
    return parse_turns(data, st.host_name, st.expert_name)
