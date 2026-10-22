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

import base64

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


def bodhan_request(png: bytes, mime: str = "image/png") -> dict:
    """The Indic-OCR request body: one user message, one image (one page), nothing else."""
    b64 = base64.b64encode(png).decode()
    return {"model": get_settings().ocr_model, "table_format": "markdown", "max_tokens": 4096,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}]}


async def bodhan_ocr(png: bytes, mime: str = "image/png") -> str:
    s = get_settings()
    url = (s.ocr_base_url or BODHAN_BASE).rstrip("/") + "/v1/chat/completions"
    resp = await request_with_retry("OCR service", "POST", url, json=bodhan_request(png, mime),
                                    headers={"Authorization": f"Bearer {s.ocr_api_key}"}, timeout=s.ocr_timeout_s)
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
                                        headers=headers, timeout=120)
        return resp.json().get("text", "")
    b64 = base64.b64encode(png).decode()
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": OCR_PROMPT},
    ]}]
    return await chat(messages, task="ocr", temperature=0.0, max_tokens=3000)
