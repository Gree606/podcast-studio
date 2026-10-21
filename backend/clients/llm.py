"""LLM client. Two wire formats, chosen by LLM_PROVIDER:

* ``gemini``: Google Generative Language API,
  POST {base}/v1beta/models/{model}:generateContent with an X-goog-api-key header.
* ``openai``: any OpenAI-compatible /v1/chat/completions server (e.g. vLLM serving Gemma).

The rest of the app always speaks OpenAI-style messages
([{"role": "system"|"user"|"assistant", "content": str | parts}]); this module
translates them. We don't use server-side tool calling: the agent asks for a JSON
action in the prompt and parses it (see agent/planner.py), so it works the same
on every backend.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from ..config import get_settings
from ..jsonutil import JSONExtractError, extract_json
from .base import ServiceError, request_with_retry
from .usage import record

# Gemma 4 emits an (often empty) thought channel before the answer; some servers
# don't strip it. Also strip generic <think> blocks from other reasoning models.
_THOUGHT_RE = re.compile(r"<\|channel\>thought.*?<channel\|>|<think>.*?</think>", re.S)
_DATA_URI = re.compile(r"^data:([\w/+.-]+);base64,(.*)$", re.S)

log = logging.getLogger("podcast.llm")
_sem: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(get_settings().llm_max_concurrency)
    return _sem


def clean_output(text: str) -> str:
    return _THOUGHT_RE.sub("", text or "").strip()


# ---------------------------------------------------------------------------
# Gemini translation
# ---------------------------------------------------------------------------
def to_gemini(messages: list[dict[str, Any]]) -> tuple[dict | None, list[dict]]:
    """OpenAI-style messages -> (systemInstruction, contents). Consecutive same-role
    turns are merged because Gemini expects user/model alternation."""
    system_parts: list[dict] = []
    contents: list[dict] = []
    for m in messages:
        content = m.get("content")
        parts: list[dict] = []
        if isinstance(content, str):
            parts.append({"text": content})
        else:
            for p in content or []:
                if p.get("type") == "text":
                    parts.append({"text": p.get("text", "")})
                elif p.get("type") == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    mm = _DATA_URI.match(url)
                    if mm:
                        parts.append({"inline_data": {"mime_type": mm.group(1), "data": mm.group(2)}})
        if m.get("role") == "system":
            system_parts += parts
            continue
        role = "model" if m.get("role") == "assistant" else "user"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"] += parts
        else:
            contents.append({"role": role, "parts": parts})
    system = {"parts": system_parts} if system_parts else None
    return system, contents


def from_gemini(data: dict) -> tuple[str, dict]:
    """Extract answer text (skipping thought parts) and usage from a generateContent response."""
    cands = data.get("candidates") or []
    if not cands:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise ServiceError("Language model", f"declined to answer ({reason})")
    cand = cands[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    finish = cand.get("finishReason", "")
    if not text.strip():
        raise ServiceError("Language model", f"returned an empty answer ({finish or 'unknown reason'})")
    um = data.get("usageMetadata") or {}
    usage = {
        "prompt_tokens": int(um.get("promptTokenCount", 0)),
        "completion_tokens": int(um.get("candidatesTokenCount", 0)) + int(um.get("thoughtsTokenCount", 0)),
    }
    return text, usage


async def _gemini_call(messages, *, temperature, max_tokens, json_mode, base_url, api_key, model, attempts=None) -> tuple[str, dict]:
    s = get_settings()
    system, contents = to_gemini(messages)
    gen: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens + s.llm_thinking_headroom}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    body: dict[str, Any] = {"contents": contents, "generationConfig": gen}
    if system:
        body["systemInstruction"] = system
    resp = await request_with_retry(
        "Language model", "POST", f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent",
        json=body, headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
        timeout=s.llm_timeout_s, attempts=attempts or s.llm_attempts, semaphore=_semaphore(),
    )
    try:
        data = resp.json()
    except ValueError as e:
        raise ServiceError("Language model", "returned an unexpected response", str(e)) from e
    return from_gemini(data)


async def _openai_call(messages, *, temperature, max_tokens, json_mode, base_url, api_key, model, attempts=None) -> tuple[str, dict]:
    s = get_settings()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    resp = await request_with_retry(
        "Language model", "POST", base_url.rstrip("/") + "/v1/chat/completions", json=body, headers=headers,
        timeout=s.llm_timeout_s, attempts=attempts or s.llm_attempts, semaphore=_semaphore(),
    )
    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError) as e:
        raise ServiceError("Language model", "returned an unexpected response", str(e)) from e
    u = data.get("usage") or {}
    return text, {"prompt_tokens": int(u.get("prompt_tokens", 0)), "completion_tokens": int(u.get("completion_tokens", 0))}


async def chat(
    messages: list[dict[str, Any]],
    *,
    task: str,
    temperature: float = 0.7,
    max_tokens: int = 1500,
    json_mode: bool = False,
    endpoint: tuple[str, str, str] | None = None,
) -> str:
    """Return the assistant text. ``task`` labels the call for logs and the mock backend.

    ``endpoint`` = (base_url, api_key, model) overrides the default LLM; the eval harness
    uses it to grade with a different judge model than the one that wrote the script.
    """
    s = get_settings()
    if s.llm_mock():
        from . import mock

        record(llm_calls=1)
        return await mock.mock_chat(messages, task=task, temperature=temperature)

    base_url, api_key, model = endpoint or (s.llm_base_url, s.llm_api_key, s.llm_model)
    call = _gemini_call if s.llm_provider == "gemini" else _openai_call
    t0 = time.time()
    text, usage, used = await _call_with_fallback(call, messages, temperature=temperature, max_tokens=max_tokens,
                                                  json_mode=json_mode, base_url=base_url, api_key=api_key,
                                                  models=model_chain(model, explicit=endpoint is not None))
    record(llm_calls=1, **usage)
    log.info("llm task=%s model=%s %.1fs tokens=%s", task, used, time.time() - t0, usage)
    return clean_output(text)


FALLBACK_STATUS = {404, 429, 500, 502, 503, 504}


def model_chain(model: str, *, explicit: bool = False) -> list[str]:
    """The main model, then the configured fallbacks (deduplicated). An explicit endpoint
    (the eval judge) never falls back, so its scores always come from the model it names."""
    if explicit:
        return [model]
    extra = [m.strip() for m in get_settings().llm_fallback_models.split(",") if m.strip()]
    return list(dict.fromkeys([model, *extra]))


async def _call_with_fallback(call, messages, *, models: list[str], **kw) -> tuple[str, dict, str]:
    """Try each model in turn when one is overloaded/rate-limited/retired. Other errors
    (bad key, bad request, malformed answer) are raised at once: another model won't fix them."""
    s = get_settings()
    last: ServiceError | None = None
    for i, m in enumerate(models):
        try:
            text, usage = await call(messages, model=m, attempts=s.llm_attempts_per_model if i < len(models) - 1 else s.llm_attempts, **kw)
            return text, usage, m
        except ServiceError as e:
            if e.status not in FALLBACK_STATUS or i == len(models) - 1:
                raise
            last = e
            log.warning("model %s unavailable (%s); falling back to %s", m, e.user_message, models[i + 1])
    raise last or ServiceError("Language model", "is unavailable")


async def chat_json(
    messages: list[dict[str, Any]],
    *,
    task: str,
    expect: type = dict,
    temperature: float = 0.4,
    max_tokens: int = 1500,
    retries: int = 2,
    endpoint: tuple[str, str, str] | None = None,
) -> Any:
    """Chat and parse JSON; on a parse failure, show the model its output and ask it to fix it."""
    msgs = list(messages)
    last_err: Exception | None = None
    for _ in range(retries + 1):
        text = await chat(msgs, task=task, temperature=temperature, max_tokens=max_tokens, json_mode=True, endpoint=endpoint)
        try:
            return extract_json(text, expect=expect)
        except JSONExtractError as e:
            last_err = e
            msgs = msgs + [
                {"role": "assistant", "content": text[:4000]},
                {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON, no prose, no code fences."},
            ]
            temperature = max(0.1, temperature - 0.2)
    raise ServiceError("Language model", "kept returning malformed output", str(last_err))
