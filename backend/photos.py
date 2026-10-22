"""Photo-to-podcast: turn photos of textbook pages into a Source via OCR.

Bodhan's indic-ocr reads exactly ONE page per request, so every photo is one page
and one OCR call; pages are read in the order the user gave them and joined.
Phone photos need care before OCR: they are often stored sideways (EXIF
orientation), are 12+ megapixels (slow to upload, no better to read), and may be
PNG screenshots or other formats. We straighten, downscale and re-encode as JPEG.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re

from PIL import Image, ImageOps, UnidentifiedImageError

from .clients.base import ServiceError
from .clients.ocr import ocr_available, ocr_page
from .ingest import IngestError, ingest_text
from .schemas import Source

log = logging.getLogger("podcast.photos")

MAX_PAGES = 6
MAX_PHOTO_BYTES = 15 * 1024 * 1024
LONG_SIDE = 2200  # px: plenty for print-size text, keeps each upload to ~0.5-1 MB
MIN_SIDE = 500  # px: smaller than this, body text is unreadable
MIN_PAGE_WORDS = 15


def prepare_photo(data: bytes, label: str) -> bytes:
    """Return an upright, reasonably sized JPEG of the photo, or raise IngestError with a fix."""
    if len(data) > MAX_PHOTO_BYTES:
        raise IngestError(f"{label} is too large (max 15 MB). Try a normal camera photo instead of a scan at full resolution.")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # phones store "portrait" photos sideways + an EXIF flag
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise IngestError(f"{label} isn't an image we can read. Please use JPG or PNG "
                          "(on iPhone: Settings > Camera > Formats > Most Compatible).") from e
    if min(img.size) < MIN_SIDE:
        raise IngestError(f"{label} is too small ({img.size[0]}x{img.size[1]} px) to read. Take the photo closer to the page.")
    img.thumbnail((LONG_SIDE, LONG_SIDE))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=88, optimize=True)
    return out.getvalue()


SUBJECT_HEADERS = {"science", "mathematics", "maths", "history", "geography", "economics", "civics", "biology",
                   "chemistry", "physics", "english", "social science", "political science", "democratic politics"}
_FURNITURE = [
    re.compile(r"^\$*\s*\d{1,3}\s*\$*$"),                      # page numbers, incl. OCR's LaTeX "$$101$$"
    re.compile(r"^\|?[\s:|-]+\|?$"),                             # empty markdown table rules
    re.compile(r"^re\w{2,6}\s+20\d\d\s*[-–]\s*\d{2,4}$", re.I),  # NCERT "Reprint 2026-27" (often misread "Report")
    re.compile(r"^\(?[a-z]\)?$", re.I),                           # lone figure labels "(D)"
    re.compile(r"^(fig\.?|figure)\s*\d+(\.\d+)*\b", re.I),        # figure captions
]


def clean_page_text(text: str) -> str:
    """Drop page furniture (numbers, reprint labels, running subject headers, figure labels)."""
    lines = []
    for idx, ln in enumerate((text or "").splitlines()):
        t = ln.strip()
        bare = t.strip("#* ").strip()
        if any(rx.match(bare) for rx in _FURNITURE):
            continue
        if idx < 6 and bare.lower() in SUBJECT_HEADERS:  # running header at the top of the page
            continue
        lines.append(ln.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def drop_repeated_headers(pages: list[str]) -> list[str]:
    """Remove short lines that open several pages (running headers), keeping their first occurrence."""
    if len(pages) < 2:
        return pages
    tops = [[ln.strip() for ln in p.splitlines() if ln.strip()][:3] for p in pages]
    seen: dict[str, int] = {}
    for top in tops:
        for ln in set(top):
            if len(ln.split()) <= 6:
                seen[ln] = seen.get(ln, 0) + 1
    repeated = {ln for ln, n in seen.items() if n > 1}
    if not repeated:
        return pages
    out, already = [], set()  # keep the FIRST occurrence: on page 1 a repeated chapter name is the real title
    for p in pages:
        kept, head = [], 0
        for ln in p.splitlines():
            if ln.strip():
                head += 1
            if head <= 3 and ln.strip() in repeated:
                if ln.strip() in already:
                    continue
                already.add(ln.strip())
            kept.append(ln)
        out.append("\n".join(kept).strip())
    return out


async def ingest_photos(photos: list[tuple[str, bytes]], max_words: int, title: str | None = None) -> Source:
    """OCR each photo (one page per request, in order) and build a Source from the pages' text."""
    if not photos:
        raise IngestError("Add at least one photo of a textbook page.")
    if len(photos) > MAX_PAGES:
        raise IngestError(f"Please send at most {MAX_PAGES} pages at a time.")
    if not ocr_available():
        raise IngestError("Reading photos needs the OCR service, which isn't configured right now. Please upload a PDF or paste the text.")
    images = [prepare_photo(data, f"Photo {i + 1}" + (f" ({name})" if name else "")) for i, (name, data) in enumerate(photos)]

    sem = asyncio.Semaphore(2)  # gentle on the OCR service's rate limit

    async def read(i: int, jpeg: bytes) -> str:
        async with sem:
            try:
                return clean_page_text(await ocr_page(jpeg, "image/jpeg"))
            except ServiceError as e:
                log.warning("OCR failed on photo %d: %s", i + 1, e)
                raise IngestError(f"We couldn't read photo {i + 1}: the OCR service {e.user_message}. Please try again.") from e

    texts = drop_repeated_headers(list(await asyncio.gather(*(read(i, img) for i, img in enumerate(images)))))
    notes = [f"Read {len(texts)} photographed page{'s' if len(texts) != 1 else ''} with OCR (one page per photo)."]
    for i, t in enumerate(texts):
        if len(t.split()) < MIN_PAGE_WORDS:
            notes.append(f"Photo {i + 1}: very little text found. Is it blurry, dark or cut off?")
    total = sum(len(t.split()) for t in texts)
    if total < 80:
        raise IngestError("We could only read a few words from your photo(s). Retake them flat, in good light, "
                          "with the whole page in frame (or add more pages).")
    # Pages are joined in order; a heading on a later page starts a new section as usual.
    src = ingest_text("\n\n".join(texts), max_words, origin="photo", title=title)
    if not title and src.title in ("", "Introduction"):
        src.title = "My textbook pages"
    src.ocr_pages = list(range(1, len(texts) + 1))
    src.page_texts = texts
    src.notes = notes + src.notes
    return src
