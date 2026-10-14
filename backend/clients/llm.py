"""LLM client for an OpenAI-compatible /v1/chat/completions endpoint (vLLM serving Gemma).

We deliberately do NOT rely on server-side tool calling (`tools=`): the vLLM
deployments are not launched with `--enable-auto-tool-choice`, so the agent asks
for a JSON action in the prompt and we parse it (see agent/planner.py).
"""
from __future__ import annotations

import asyncio
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

_sem: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(get_settings().llm_max_concurrency)
    return _sem


def clean_output(text: str) -> str:
    return _THOUGHT_RE.sub("", text or "").strip()


async def chat(
    messages: list[dict[str, Any]],
    *,
    task: str,
    temperature: float = 0.7,
    max_tokens: int = 1500,
) -> str:
    """Return the assistant text. ``task`` labels the call for logs and the mock backend."""
    s = get_settings()
    if s.effective_mock():
        from . import mock

        record(llm_calls=1)
        return await mock.mock_chat(messages, task=task, temperature=temperature)

    headers = {"Content-Type": "application/json"}
    if s.llm_api_key:
        headers["Authorization"] = f"Bearer {s.llm_api_key}"
    body = {"model": s.llm_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    t0 = time.time()
    resp = await request_with_retry(
        "Language model",
        "POST",
        s.llm_base_url.rstrip("/") + "/v1/chat/completions",
        json=body,
        headers=headers,
        timeout=s.llm_timeout_s,
        semaphore=_semaphore(),
    )
    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError) as e:
        raise ServiceError("Language model", "returned an unexpected response", str(e)) from e
    usage = data.get("usage") or {}
    record(llm_calls=1, prompt_tokens=int(usage.get("prompt_tokens", 0)), completion_tokens=int(usage.get("completion_tokens", 0)))
    import logging

    logging.getLogger("podcast.llm").info("llm task=%s %.1fs tokens=%s", task, time.time() - t0, usage)
    return clean_output(text)


async def chat_json(
    messages: list[dict[str, Any]],
    *,
    task: str,
    expect: type = dict,
    temperature: float = 0.4,
    max_tokens: int = 1500,
    retries: int = 2,
) -> Any:
    """Chat and parse JSON; on a parse failure, show the model its output and ask it to fix it."""
    msgs = list(messages)
    last_err: Exception | None = None
    for _ in range(retries + 1):
        text = await chat(msgs, task=task, temperature=temperature, max_tokens=max_tokens)
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
