"""OCR for scanned PDF pages.

Default (OCR_MODE=llm) sends the rasterised page to the LLM's vision input as an
OpenAI ``image_url`` part, image before text. OCR_MODE=bodhan uses Bodhan's Indic-OCR:
POST {base}/v1/chat/completions with model "indic-ocr" and exactly one user message
holding one ``image_url`` data URI (no prompt, no language field), ``Authorization:
Bearer <key>``; the page comes back as reading-ordered Markdown in
choices[0].message.content. OCR_MODE=http posts the PNG to a dedicated endpoint
returning {"text": ...}.
"""
from __future__ import annotations

import asyncio
import base64
import time

from ..config import get_settings
from .base import ServiceError, request_with_retry
from .llm import chat

BODHAN_BASE = "https://api.bodhan.ai"

OCR_PROMPT = (
    "Transcribe all text on this textbook page exactly, in reading order. Keep headings on their own lines, "
    "keep paragraph breaks, write tables as rows with ' | ' between cells. Skip page numbers and running headers. "
    "Output only the transcribed text."
)


def ocr_available() -> bool:
    s = get_settings()
    if s.ocr_mode == "off":
        return False
    if s.ocr_mode == "http":
        return bool(s.ocr_base_url)
    if s.ocr_mode == "bodhan" and s.ocr_ready():
        return True
    # LLM vision. Its offline fake is only for MOCK_MODE (tests/demos): with a real deployment that
    # merely lacks the LLM key, fake OCR text would silently replace the student's chapter.
    return s.mock_mode or (s.llm_ready() and s.llm_vision)


def _image_part(png: bytes, mime: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(png).decode()}"}}


def bodhan_request(png: bytes, mime: str = "image/png") -> dict:
    """The Indic-OCR request body: one user message, one image (one page), nothing else."""
    return {"model": get_settings().ocr_model, "table_format": "markdown", "max_tokens": 4096,
            "messages": [{"role": "user", "content": [_image_part(png, mime)]}]}


_next_slot = 0.0
_pace_lock: asyncio.Lock | None = None
_pace_loop: asyncio.AbstractEventLoop | None = None


async def _pace() -> None:
    """Space Bodhan OCR calls evenly. A burst of 3-4 pages gets HTTP 429 and a ~40 s wait (measured
    2026-10-31: 6 pages at 4 at a time took 192 s); one page every 5 s ran 6 for 6 at ~1 s each."""
    global _next_slot, _pace_lock, _pace_loop
    limit = get_settings().ocr_requests_per_minute
    if limit <= 0:
        return
    loop = asyncio.get_running_loop()
    if _pace_lock is None or _pace_loop is not loop:
        _pace_lock, _pace_loop = asyncio.Lock(), loop
    async with _pace_lock:
        now = time.monotonic()
        wait = _next_slot - now
        _next_slot = max(now, _next_slot) + 60.0 / limit
    if wait > 0:
        await asyncio.sleep(wait)


async def bodhan_ocr(png: bytes, mime: str = "image/png") -> str:
    s = get_settings()
    url = (s.ocr_base_url or BODHAN_BASE).rstrip("/") + "/v1/chat/completions"
    resp = await request_with_retry("OCR service", "POST", url, json=bodhan_request(png, mime),
                                    headers={"Authorization": f"Bearer {s.ocr_api_key}"}, timeout=s.ocr_timeout_s,
                                    pace=_pace)
    try:
        return resp.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise ServiceError("OCR service", "returned an unexpected response", str(e)) from e


async def ocr_page(png: bytes, mime: str = "image/png") -> str:
    """Text of ONE page image (Bodhan's OCR takes exactly one page per request)."""
    s = get_settings()
    if s.ocr_mode == "bodhan" and s.ocr_ready() and not s.mock_mode:
        return await bodhan_ocr(png, mime)
    if s.ocr_mode == "http":
        headers = {"Authorization": f"Bearer {s.ocr_api_key}"} if s.ocr_api_key else {}
        resp = await request_with_retry("OCR service", "POST", s.ocr_base_url, files={"file": ("page", png, mime)},
                                        headers=headers, timeout=s.ocr_timeout_s)
        return resp.json().get("text", "")
    messages = [{"role": "user", "content": [_image_part(png, mime), {"type": "text", "text": OCR_PROMPT}]}]
    return await chat(messages, task="ocr", temperature=0.0, max_tokens=3000)
