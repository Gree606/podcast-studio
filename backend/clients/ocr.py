"""OCR for scanned PDF pages.

Default (OCR_MODE=llm) sends the rasterised page to Gemma's vision input as an
OpenAI ``image_url`` part; Gemma 4 lists document OCR as a supported capability
and wants the image before the text. OCR_MODE=http posts the PNG to a dedicated
endpoint returning {"text": ...}.
"""
from __future__ import annotations

import base64

from ..config import get_settings
from .base import request_with_retry
from .llm import chat

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
    return s.effective_mock() or s.llm_vision


async def ocr_page(png: bytes) -> str:
    s = get_settings()
    if s.ocr_mode == "http":
        headers = {"Authorization": f"Bearer {s.ocr_api_key}"} if s.ocr_api_key else {}
        resp = await request_with_retry("OCR service", "POST", s.ocr_base_url, files={"file": ("page.png", png, "image/png")},
                                        headers=headers, timeout=120)
        return resp.json().get("text", "")
    b64 = base64.b64encode(png).decode()
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": OCR_PROMPT},
    ]}]
    return await chat(messages, task="ocr", temperature=0.0, max_tokens=3000)
