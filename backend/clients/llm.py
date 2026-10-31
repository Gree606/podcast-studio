"""LLM client. Two wire formats, chosen by LLM_PROVIDER:

* ``gemini``: Google Generative Language API,
  POST {base}/v1beta/models/{model}:generateContent with an X-goog-api-key header.
* ``openai``: any OpenAI-compatible /v1/chat/completions server (e.g. vLLM serving Gemma).
* ``openrouter``: the same OpenAI wire format against OpenRouter, plus its reasoning/JSON options.

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
_THOUGHT_RE = re.compile(r"<\|channel\>thought.*?<channel\|>|<think>.*?</think>", re.DOTALL)
_DATA_URI = re.compile(r"^data:([\w/+.-]+);base64,(.*)$", re.DOTALL)

log = logging.getLogger("podcast.llm")
_sem: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(get_settings().llm_max_concurrency)
    return _sem


def clean_output(text: str) -> str:
    return _THOUGHT_RE.sub("", text or "").strip()


def _json_body(resp) -> Any:
    try:
        return resp.json()
    except ValueError as e:
        raise ServiceError("Language model", "returned an unexpected response", str(e)) from e


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


async def _gemini_call(messages, *, temperature, max_tokens, json_mode, base_url, api_key, model, attempts=None, thinking=None) -> tuple[str, dict]:
    s = get_settings()
    system, contents = to_gemini(messages)
    gen: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens + s.llm_thinking_headroom}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    if thinking:  # "low" for checking tasks: much faster, and checking doesn't need long deliberation
        gen["thinkingConfig"] = {"thinkingLevel": thinking}
    body: dict[str, Any] = {"contents": contents, "generationConfig": gen}
    if system:
        body["systemInstruction"] = system
    resp = await request_with_retry(
        "Language model", "POST", f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent",
        json=body, headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
        timeout=s.llm_timeout_s, attempts=attempts or s.llm_attempts, semaphore=_semaphore(), retry_timeouts=False,
    )
    return from_gemini(_json_body(resp))


async def _openai_call(messages, *, temperature, max_tokens, json_mode, base_url, api_key, model, attempts=None, thinking=None) -> tuple[str, dict]:
    s = get_settings()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if s.llm_provider == "openrouter":
        headers.update({"HTTP-Referer": "https://github.com/Gree606/podcast-studio", "X-Title": "Mission 606: Earth Unknown"})
        body["max_tokens"] = max_tokens + s.llm_thinking_headroom  # reasoning tokens count against it, as on Gemini
        if thinking:  # "low" for checking tasks, like Gemini's thinkingLevel
            body["reasoning"] = {"effort": thinking}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
    resp = await request_with_retry(
        "Language model", "POST", base_url.rstrip("/") + "/v1/chat/completions", json=body, headers=headers,
        timeout=s.llm_timeout_s, attempts=attempts or s.llm_attempts, semaphore=_semaphore(), retry_timeouts=False,
    )
    data = _json_body(resp)
    if isinstance(data.get("error"), dict):  # OpenRouter can report an upstream failure inside a 200
        err = data["error"]
        code = err.get("code") if isinstance(err.get("code"), int) else 502
        raise ServiceError("Language model", f"returned HTTP {code}", str(err.get("message", ""))[:300], status=code)
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise ServiceError("Language model", "returned an unexpected response", str(e)) from e
    if (data["choices"][0].get("finish_reason") or "") == "length":
        log.warning("model=%s stopped at the output limit (%d tokens): the answer is cut off", model, body["max_tokens"])
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
    thinking: str | None = None,
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

    base_url, api_key, model = endpoint or (s.llm_base_url, s.llm_key(), s.llm_model)
    call = _gemini_call if s.llm_provider == "gemini" else _openai_call
    t0 = time.time()
    text, usage, used = await _call_with_fallback(call, messages, temperature=temperature, max_tokens=max_tokens,
                                                  json_mode=json_mode, base_url=base_url, api_key=api_key, thinking=thinking,
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


# model -> monotonic time until which its quota is known to be used up (e.g. a free-tier daily cap)
_exhausted: dict[str, float] = {}


def _quota_message(until_s: float) -> str:
    hours = max(1, round(until_s / 3600))
    return (f"has used up its free daily quota (the Gemini free tier allows about 20 requests per model per day); "
            f"it resets in about {hours} h, or enable billing on the API key to lift the cap")


async def _call_with_fallback(call, messages, *, models: list[str], **kw) -> tuple[str, dict, str]:
    """Try each model in turn when one is overloaded/rate-limited/retired. A model whose quota is
    used up for hours is skipped immediately for the rest of that window. Other errors (bad key,
    bad request, malformed answer) are raised at once: another model won't fix them."""
    s = get_settings()
    now = time.monotonic()
    live = [m for m in models if _exhausted.get(m, 0) <= now]
    if not live:
        raise ServiceError("Language model", _quota_message(min(_exhausted[m] for m in models) - now), status=429)
    last: ServiceError | None = None
    for i, m in enumerate(live):
        try:
            text, usage = await call(messages, model=m, attempts=s.llm_attempts_per_model if i < len(live) - 1 else s.llm_attempts, **kw)
            return text, usage, m
        except ServiceError as e:
            if e.status not in FALLBACK_STATUS:
                raise
            if e.retry_after:
                _exhausted[m] = time.monotonic() + e.retry_after
            last = e
            if i == len(live) - 1:
                break
            log.warning("model %s unavailable (%s); falling back to %s", m, e.user_message, live[i + 1])
    if last is not None and all(_exhausted.get(m, 0) > time.monotonic() for m in models):
        raise ServiceError("Language model", _quota_message(min(_exhausted[m] for m in models) - time.monotonic()), status=429)
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
    thinking: str | None = None,
) -> Any:
    """Chat and parse JSON; on a parse failure, show the model its output and ask it to fix it."""
    msgs = list(messages)
    last_err: Exception | None = None
    json_mode = True
    for _ in range(retries + 1):
        try:
            text = await chat(msgs, task=task, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode, endpoint=endpoint, thinking=thinking)
        except ServiceError as e:
            # Gemini sometimes fails *inside* its JSON mode (finishReason MALFORMED_RESPONSE). Plain
            # text plus our own JSON repair usually works, so retry once without it.
            if json_mode and "MALFORMED" in e.user_message:
                log.warning("task=%s: JSON mode failed (%s); retrying in plain-text mode", task, e.user_message)
                json_mode, last_err = False, e
                continue
            raise
        try:
            return extract_json(text, expect=expect)
        except JSONExtractError as e:
            log.warning("task=%s: unparseable model output: %r", task, text[:400])
            last_err = e
            msgs = msgs + [
                {"role": "assistant", "content": text[:4000]},
                {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON, no prose, no code fences."},
            ]
            temperature = max(0.1, temperature - 0.2)
    raise ServiceError("Language model", "kept returning malformed output", str(last_err))
