"""Robust JSON extraction from LLM output.

Small open models (Gemma) often wrap JSON in prose or ```json fences, leave trailing
commas, or use smart quotes. We repair the common cases deterministically rather
than spending another LLM call.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class JSONExtractError(ValueError):
    pass


def _balanced_span(text: str, start: int) -> str | None:
    """Return the substring from text[start] (a '{' or '[') to its matching close."""
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1] if c == close_ch else None
    return None


def _repair(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas


def _close_truncated(s: str) -> str:
    """Best effort: close brackets of a truncated JSON document (max_tokens cut-off)."""
    stack: list[str] = []
    in_str = False
    esc = False
    for c in s:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]" and stack:
            stack.pop()
    if in_str:
        s += '"'
    s = re.sub(r",\s*$", "", s.rstrip())
    return s + "".join(reversed(stack))


def _jsonl_objects(text: str) -> list:
    """Parse text made of several consecutive top-level JSON objects ({...}\n{...}); [] otherwise."""
    out, i, t = [], 0, text.strip()
    while i < len(t):
        if t[i] in " \n\r\t,":
            i += 1
            continue
        if t[i] != "{":
            return []
        span = _balanced_span(t, i)
        if span is None:
            return []
        try:
            out.append(json.loads(_repair(span)))
        except json.JSONDecodeError:
            return []
        i += len(span)
    return out


def extract_json(text: str, expect: type | None = None) -> Any:
    """Parse the first JSON object/array found in ``text``.

    ``expect`` (dict or list) picks which bracket type to look for first.
    """
    if text is None:
        raise JSONExtractError("empty response")
    candidates: list[str] = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    if expect is dict:
        openers = "{["
    elif expect is list:
        openers = "[{"
    else:
        # No preference: take whichever bracket comes first, so a top-level list of objects
        # isn't mistaken for its own first element.
        first_obj, first_arr = text.find("{"), text.find("[")
        openers = "[{" if first_arr != -1 and (first_obj == -1 or first_arr < first_obj) else "{["
        objs = _jsonl_objects(text)
        if len(objs) > 1:  # several top-level objects in a row (JSON Lines): return them as a list
            return objs
    for cand in candidates:
        cand = cand.strip()
        for opener in openers:
            idx = cand.find(opener)
            while idx != -1:
                span = _balanced_span(cand, idx)
                if span is None:
                    # unbalanced outer object = truncated output; close it rather than
                    # returning one of its inner objects
                    try:
                        val = json.loads(_repair(_close_truncated(cand[idx:])))
                        if expect is None or isinstance(val, expect):
                            return val
                    except json.JSONDecodeError:
                        pass
                else:
                    for attempt in (span, _repair(span)):
                        try:
                            val = json.loads(attempt)
                            if expect is None or isinstance(val, expect):
                                return val
                        except json.JSONDecodeError:
                            pass
                idx = cand.find(opener, idx + 1)
    raise JSONExtractError(f"no JSON found in model output: {text[:200]!r}")
