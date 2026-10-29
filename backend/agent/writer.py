"""Writer: turns one outline segment into a two-host dialogue (JSON list of turns)."""
from __future__ import annotations

import re

import logging

from ..clients.llm import chat_json
from ..prompts import (POSITION_FIRST, POSITION_LAST, POSITION_MIDDLE, REGEN_INSTRUCTION, REVISE_USER, WRITER_SYSTEM,
                       WRITER_USER)
from ..schemas import Segment, Turn
from .context import AgentContext

log = logging.getLogger("podcast.writer")
SHOW = "Mission 606: Earth Unknown"
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


TURN_KEYS = ("turns", "dialogue", "dialog", "script", "conversation", "lines", "segment", "exchanges")
SPEAKER_KEYS = ("speaker", "role", "name", "host", "who")
TEXT_KEYS = ("text", "line", "content", "dialogue", "utterance", "says")
_NAMED_LINE = re.compile(r"^\s*([A-Za-z][\w ]{0,20}?)\s*:\s*(.+)$")


def _find_turn_list(data) -> list | None:
    """Real models don't always use our exact schema: accept a bare list, an alternative key,
    or one level of nesting ({"segment": {"dialogue": [...]}})."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return None
    for k in TURN_KEYS:
        v = data.get(k)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            inner = _find_turn_list(v)
            if inner is not None:
                return inner
    lists = [v for v in data.values() if isinstance(v, list) and v and all(isinstance(x, (dict, str)) for x in v)]
    return lists[0] if len(lists) == 1 else None


def _speaker_and_text(r) -> tuple[str, str]:
    if isinstance(r, str):  # "Riya: So what's a reaction?"
        m = _NAMED_LINE.match(r)
        return (m.group(1), m.group(2)) if m else ("", r)
    if not isinstance(r, dict):
        return "", ""
    sp = next((str(r[k]) for k in SPEAKER_KEYS if isinstance(r.get(k), str)), "")
    text = next((str(r[k]) for k in TEXT_KEYS if isinstance(r.get(k), str)), "")
    if not sp and not text and len(r) == 1:  # {"Riya": "So what's a reaction?"}
        sp, text = next(iter(r.items()))
        text = str(text)
    return sp, text


def parse_turns(data, host_name: str, expert_name: str, min_turns: int = 4) -> list[Turn]:
    raw = _find_turn_list(data)
    if not isinstance(raw, list):
        raise WriterError("no 'turns' list")
    turns: list[Turn] = []
    for r in raw:
        sp, text = _speaker_and_text(r)
        sp, text = sp.strip().lower(), text.strip()
        text = re.sub(r"\[[^\]]{1,30}\]|\*[^*]{1,30}\*", "", text).strip()  # stage directions
        # "2. Arjun: You got it!" - a numbered and/or name-prefixed line inside the text (seen from real Gemini)
        m = re.match(rf"^(?:\d+[.)]\s*)?({re.escape(host_name)}|{re.escape(expert_name)}|host|expert)\s*:\s*", text, re.I)
        if m:
            sp, text = m.group(1).lower(), text[m.end():].strip()
        if not text:
            continue
        if sp in ("host", host_name.lower()) or sp.startswith(host_name.lower()):
            speaker = "host"
        elif sp in ("expert", expert_name.lower()) or sp.startswith(expert_name.lower()):
            speaker = "expert"
        else:
            speaker = "expert" if turns and turns[-1].speaker == "host" else "host"
        if turns and turns[-1].speaker == speaker:  # merge same-speaker consecutive turns
            turns[-1] = Turn(speaker=speaker, text=turns[-1].text + " " + text)
        else:
            turns.append(Turn(speaker=speaker, text=text))
    if len(turns) < min_turns:
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
    data = await chat_json(msgs, task="write", expect=None, temperature=temperature, max_tokens=2600)
    try:
        return parse_turns(data, st.host_name, st.expert_name)
    except WriterError:
        log.warning("writer output had no usable dialogue: %r", str(data)[:400])
        raise


async def revise(ctx: AgentContext, seg: Segment, segments: list[Segment], draft: list[Turn], feedback: str,
                 *, temperature: float = 0.6) -> list[Turn]:
    st = ctx.episode.settings
    msgs = build_messages(ctx, seg, segments)
    msgs.append({"role": "user", "content": REVISE_USER.format(
        draft=script_text(draft, (st.host_name, st.expert_name)), feedback=feedback)})
    data = await chat_json(msgs, task="revise", expect=None, temperature=temperature, max_tokens=2600)
    return parse_turns(data, st.host_name, st.expert_name)
