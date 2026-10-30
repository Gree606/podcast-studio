"""Deterministic offline stand-ins for the LLM, TTS and ASR (MOCK_MODE=1).

They let the full pipeline (planner tool loop, best-of-N, claim check, ASR
self-check, stitching, regeneration) run in tests and demos with no network.
Outputs are plausible-shaped, not good content.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
from contextvars import ContextVar

import numpy as np
import soundfile as sf

_last_tts_text: ContextVar[str] = ContextVar("_last_tts_text", default="")
_tts_attempts: dict[str, int] = {}


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def _text_of(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def _first_sentence(t: str) -> str:
    return re.split(r"(?<=[.!?])\s+", t.strip())[0][:220]


def _planner(messages: list[dict]) -> str:
    n_assist = sum(1 for m in messages if m["role"] == "assistant")
    first_user = _text_of(messages[1])
    n_seg = int(re.search(r"with (\d+) segments", first_user).group(1))
    sections = re.findall(r"^- (.+?) \(\d+ passages", first_user, re.M)
    queries = (sections * 3)[: min(n_seg, 6)]
    if n_assist == 0:
        return json.dumps({"thought": "I should see how the chapter is organised first.", "action": "list_sections", "input": {}})
    if n_assist <= len(queries):
        q = queries[n_assist - 1]
        return json.dumps({"thought": f"Next I need the core content of '{q}'.", "action": "search", "input": {"query": q}})
    if n_assist == len(queries) + 1:
        return json.dumps({"thought": "Let me check how the book defines its key term.", "action": "define",
                           "input": {"term": (sections[0].split()[-1] if sections else "energy")}})
    observations = [_text_of(m) for m in messages if m["role"] == "user" and _text_of(m).startswith("Observation:\nError") is False
                    and _text_of(m).startswith("Observation:")]
    search_obs = observations[1: 1 + len(queries)]
    segs = []
    for i in range(n_seg):
        obs = search_obs[i % max(1, len(search_obs))] if search_obs else ""
        hits = re.findall(r"\[(c\d+)\] \([^)]*\) (.+)", obs)
        facts = [{"fact": _first_sentence(txt.rstrip(".")), "chunk_id": cid} for cid, txt in hits[:3]]
        segs.append({"title": (queries[i % len(queries)] if queries else f"Part {i + 1}")[:60], "goal": "Explain the main idea clearly.",
                     "chunk_ids": [cid for cid, _ in hits[:2]], "key_facts": facts, "difficulty": 2 + (i % 3),
                     "difficulty_reason": "mock estimate"})
    return json.dumps({"thought": "I have enough grounded facts to outline the episode.", "action": "finish",
                       "input": {"outline": {"learn_line": "the big ideas of this chapter, step by step", "segments": segs}}})


def _writer(messages: list[dict], task: str, temperature: float) -> str:
    user = "\n".join(_text_of(m) for m in messages if m["role"] == "user")
    facts = re.findall(r"^- (.+?) \[c\d+\]$", user, re.M) or ["the main idea of the passage"]
    title = (re.search(r'segment \d+ of \d+: "(.+?)"', user) or re.search(r"(.+)", user)).group(1)
    better = task in ("revise", "claim_fix") or "listener asked" in user
    reactions = ["Wait, really?", "Oh, that's neat.", "Hmm, okay.", "Right, right.", "No way!"]
    turns = [{"speaker": "host", "text": f"So, today let's talk about {title}. What's the big idea here?"}]
    for k, f in enumerate(facts):
        turns.append({"speaker": "expert", "text": f"Think of it like this: {f}. It's a bit like making chai, every step matters."})
        turns.append({"speaker": "host", "text": reactions[(k + _h(title)) % len(reactions)] if (better or k % 2 == 0) else
                      f"Okay so let me repeat that: {f.lower()}. Did I get that right, and why does it matter so much for us?"})
    turns.append({"speaker": "expert", "text": "Exactly. And that's the key thing to remember from this part."})
    if better:
        turns.insert(1, {"speaker": "expert", "text": "Great question, and honestly it's one of my favourites."})
    return json.dumps({"turns": turns})


def _critic(messages: list[dict]) -> str:
    script = _text_of(messages[-1])
    better = "favourites" in script
    base = 3 + (_h(script) % 2)
    s = {"faithfulness": 5 if better else 4, "coverage": 4, "naturalness": base + (1 if better else 0),
         "engagement": base, "clarity": 4}
    s = {k: min(5, v) for k, v in s.items()}
    return json.dumps({"scores": s, "unsupported": [], "missing_facts": [],
                       "feedback": "1. Add a quick reaction after turn 2. 2. Use one more everyday example."})


def _passages(text: str) -> list[tuple[str, str]]:
    return re.findall(r"^\[(c\d+)\] \([^)]*\) (.+)$", text, re.M)


def _episode(messages: list[dict]) -> str:
    """FAST pipeline call 1: the whole episode as 3 parts."""
    user = _text_of(messages[-1])
    ps = _passages(user) or [("c0", "the main idea of the chapter")]
    title = (re.search(r"^Chapter: (.+)$", user, re.M) or re.search(r"(.+)", user)).group(1)
    names = ["Opening", "The core idea", "Wrapping up"]
    parts = []
    for i in range(3):
        mine = ps[i::3] or ps[:1]
        turns = [{"speaker": "host", "text": f"Welcome to Mission 606! Today: {title}. So what's first?" if i == 0
                  else "Okay, so what comes next?"}]
        for cid, txt in mine[:2]:
            turns.append({"speaker": "expert", "text": f"Think of it like this: {_first_sentence(txt).rstrip('.')}. It's like making chai."})
            turns.append({"speaker": "host", "text": "Oh, that's neat."})
        if i == 2:
            turns.append({"speaker": "expert", "text": "Quick recap: three ideas, step by step. See you next time!"})
        parts.append({"title": names[i], "chunk_ids": [c for c, _ in mine[:2]], "turns": turns})
    return json.dumps({"title": f"{title}: the big ideas", "learn_line": "the big ideas of this chapter, step by step",
                       "parts": parts})


def _verify_episode(messages: list[dict]) -> str:
    """FAST pipeline call 2: all claims supported, except one flagged (and fixed) in part 2 to exercise the fix path."""
    user = _text_of(messages[-1])
    src = _passages(user)

    def evidence(claim: str) -> str:
        return next((cid for cid, txt in src if claim[:40] in txt), src[0][0] if src else "c1")

    out = []
    for m in re.finditer(r"^PART (\d+): .+$", user, re.M):
        n = int(m.group(1))
        body = user[m.end():].split("\nPART ")[0]
        lines, fixes = [], []
        for lm in re.finditer(r"^(\d+)\. [^:]+: (.*)$", body, re.M):
            k, text = int(lm.group(1)), lm.group(2)
            c = re.search(r"Think of it like this: (.+?)\. It's", text)
            if not c:
                lines.append({"line": k, "claim": ""})
                continue
            entry = {"line": k, "claim": c.group(1), "supported": True, "evidence": evidence(c.group(1)), "note": ""}
            if n == 2 and not fixes:  # one unsupported line, corrected
                entry["supported"] = False
                fixes = [{"line": k, "text": "Think of it like this: the book explains it step by step. It's like making chai."}]
            lines.append(entry)
        out.append({"part": n, "lines": lines, "fixes": fixes})
    return json.dumps({"parts": out})


def _ask(messages: list[dict]) -> str:
    """Ask Gree: answer from the first passage found, or admit the chapter doesn't cover it."""
    user = _text_of(messages[-1])
    src = _passages(user)
    q = re.search(r'Student\'s doubt: "(.+)"', user)
    off = q and re.search(r"cricket|pizza|movie", q.group(1), re.I)
    if not src or off:
        return json.dumps({"answer": "This chapter doesn't talk about that, but here is what it does say about our topic.",
                           "chunk_ids": [], "grounded": False, "language": "en", "follow_ups": ["What is this chapter about?"]})
    cid, txt = src[0]
    return json.dumps({"answer": f"Good doubt! {_first_sentence(txt)}", "chunk_ids": [cid], "grounded": True, "language": "en",
                       "follow_ups": ["Can you give an example?", "Why does that happen?"]})


async def mock_chat(messages: list[dict], *, task: str, temperature: float) -> str:
    await asyncio.sleep(0.01)
    if "capital of India" in _text_of(messages[-1]):  # scripts/check_apis.py
        return json.dumps({"answer": "New Delhi"})
    if task == "planner_step":
        return _planner(messages)
    if task == "write_episode":
        return _episode(messages)
    if task == "verify_episode":
        return _verify_episode(messages)
    if task == "ask_keywords":
        return json.dumps({"keywords": "plants food leaves sunlight"})
    if task == "ask_gree":
        return _ask(messages)
    if task == "rewrite_part":
        return json.dumps({"title": "Rewritten part", "chunk_ids": [c for c, _ in _passages(_text_of(messages[-1]))[:2]],
                           "turns": json.loads(_writer(messages, "revise", 0.7))["turns"]})
    if task in ("write", "revise", "claim_fix"):
        return _writer(messages, task, temperature)
    if task == "critic":
        return _critic(messages)
    if task == "reflect":
        return "Break long explanations with a quick question from the host."
    if task == "fact_verify":
        n = len(re.findall(r"^\d+\. FACT:", _text_of(messages[-1]), re.M))
        return json.dumps({"results": [{"i": i, "supported": True} for i in range(1, n + 1)]})
    if task == "claim_extract":
        sents = re.findall(r"Think of it like this: (.+?)\. It's", _text_of(messages[-1]))
        return json.dumps({"claims": sents[:6]})
    if task == "claim_judge":
        n = len(re.findall(r"^\d+\. CLAIM:", _text_of(messages[-1]), re.M))
        return json.dumps({"results": [{"i": i, "supported": True} for i in range(1, n + 1)]})
    if task == "ocr":
        return "Mock OCR text.\n\nThis page was a scanned image. " * 20
    if task == "judge":
        return json.dumps({"faithfulness": 4, "naturalness": 3, "engagement": 3, "pedagogy": 4, "comment": "mock"})
    return "{}"


async def mock_tts(text: str, speaker: str) -> bytes:
    """Speech-like placeholder: voiced 'syllables' with per-speaker pitch, ~0.3 s per word."""
    await asyncio.sleep(0.005)
    _last_tts_text.set(text)
    sr = 24000
    words = max(1, len(text.split()))
    dur = 0.3 * words + 0.2
    t = np.arange(int(sr * dur)) / sr
    f0 = 200.0 if speaker == "host" else 125.0
    f0 = f0 * (1 + 0.06 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 6))
    syll = 0.5 * (1 + np.sin(2 * np.pi * 3.3 * t)) ** 2
    audio = (0.15 * voice * syll).astype(np.float32)
    pad = np.zeros(int(0.15 * sr), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, np.concatenate([pad, audio, pad]), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


async def mock_asr(wav: bytes) -> str:
    """Return what was 'said'. The first take of ~1 in 8 turns comes back garbled, to exercise re-synthesis."""
    text = _last_tts_text.get()
    n = _tts_attempts.get(text, 0) + 1
    _tts_attempts[text] = n
    if n == 1 and _h(text) % 8 == 0:
        return " ".join(text.split()[: max(1, len(text.split()) // 3)]) + " um"
    return text
