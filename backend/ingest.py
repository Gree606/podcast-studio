"""Ingestion: PDF or pasted text -> cleaned sections -> retrievable chunks.

NCERT PDFs have running headers ("Science"), page numbers, boxed activities,
figure captions and end-of-chapter exercises. We strip the repetitive furniture,
detect headings from font size and numbering ("1.2 Types of Reactions"), and
split into ~170-word passages that the planner can search and cite.
Pages with almost no text layer (scans) are rasterised and sent to OCR.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

from .clients.ocr import ocr_available, ocr_page
from .schemas import Chunk, Section, Source, new_id

log = logging.getLogger("podcast.ingest")

CHUNK_WORDS = 170
SKIP_SECTION = re.compile(r"^\s*(exercises?|questions?|intext questions|group activity|project work|let us recall|more to know)\b", re.I)
NUMBERED_HEADING = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){1,2})\s+[A-Z][^.!?]{2,90}$")
CHAPTER_LINE = re.compile(r"^\s*(chapter|unit)\s+\d+\b", re.I)


@dataclass
class Line:
    text: str
    size: float
    bold: bool
    page: int


@dataclass
class RawSection:
    title: str
    paras: list[tuple[str, int]] = field(default_factory=list)  # (text, page)


class IngestError(ValueError):
    """Raised with a user-facing message when a source can't be used."""


def render_page_png(page: pymupdf.Page, dpi: int = 170) -> bytes:
    return page.get_pixmap(dpi=dpi).tobytes("png")


def _merge_overlap(a: str, b: str) -> str:
    """Join two fragments of the same line that overlap ("1.1 CHEMICAL EQUA" + "AL EQUATIONS")."""
    for k in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return a + " " + b


# NCERT's Hindi PDFs repeat a vowel sign with its nasal mark in the text layer ("क्योंों", "होंोंगे"),
# the Indic version of the faux-bold duplicates below. Keep one copy, in every Indian script's block
# (vowel signs at +0x3E..+0x4C, the signs that follow them at +0x00..+0x03 and the +0x62/+0x63 vowels).
_BLOCKS = range(0x0900, 0x0D80, 0x80)
_SIGN = "".join(f"\\u{b + 0x3E:04x}-\\u{b + 0x4C:04x}\\u{b + 0x62:04x}\\u{b + 0x63:04x}" for b in _BLOCKS)
_NASAL = "".join(f"\\u{b:04x}-\\u{b + 0x03:04x}" for b in _BLOCKS)
_REPEATED_SIGNS = re.compile(f"([{_SIGN}][{_NASAL}]?)\\1+")


def dedupe_signs(text: str) -> str:
    return _REPEATED_SIGNS.sub(r"\1", text)


def _page_lines(page: pymupdf.Page, pno: int) -> list[Line]:
    """Text lines in reading order, with NCERT's faux-bold duplicates removed.

    NCERT PDFs draw headings several times at the same position (and in
    overlapping fragments) to fake bold/shadow effects; we keep one copy.
    """
    out: list[Line] = []
    kept: list[tuple[Line, list[float]]] = []  # (line, bbox) on this page for dedup
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for ln in block.get("lines", []):
            spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = dedupe_signs(re.sub(r"\s+", " ", "".join(s["text"] for s in ln["spans"])).strip())
            size = max(s["size"] for s in spans)
            bold = any(("bold" in s.get("font", "").lower()) or (s.get("flags", 0) & 16) for s in spans)
            x0, y0, x1, _ = ln["bbox"]
            dup = False
            for other, ob in kept:
                if abs(ob[1] - y0) < 2.5 and abs(other.size - size) < 0.6 and x0 < ob[2] - 1 and x1 > ob[0] + 1:
                    if x0 >= ob[0] - 1.5 and x1 <= ob[2] + 1.5:
                        dup = True  # fully contained duplicate
                    elif x1 > ob[2]:
                        other.text = _merge_overlap(other.text, text)
                        ob[2] = x1
                        dup = True
                    break
            if dup:
                continue
            line = Line(text=text, size=size, bold=bold, page=pno)
            kept.append((line, [x0, y0, x1]))
            out.append(line)
        out.append(Line(text="", size=0, bold=False, page=pno))  # block break = paragraph break
    return out


def _furniture(lines_by_page: list[list[Line]]) -> set[str]:
    """Lines that repeat on many pages (running headers/footers)."""
    n = len(lines_by_page)
    if n < 3:
        return set()
    cnt: Counter[str] = Counter()
    for lines in lines_by_page:
        cnt.update({re.sub(r"\d+", "#", ln.text.lower()) for ln in lines if ln.text and len(ln.text) < 80})
    return {k for k, c in cnt.items() if c >= max(3, 0.4 * n)}


ACTIVITY = re.compile(r"^\s*(activity|example|box)(\s+\d+(\.\d+)*)?\s*$", re.I)


def _pretty_heading(t: str) -> str:
    t = re.sub(r"^[A-Z](?=[A-Z][a-z])", "", t.strip())  # stray glyph from a margin box: "OWhat you..."
    if re.fullmatch(r"chapter", t, re.I):
        return "Introduction"
    return t.title() if t.isupper() and len(t) > 4 else t


DANGLING = re.compile(r"\b(and|of|in|the|for|from|to|are|—|-|with)$", re.I)


def _guess_title(lines: list[Line], furniture: set[str] = frozenset()) -> str:
    """Largest-font text on the first page, ignoring 'CHAPTER 1' / 'UNIT 3' / numbers / footers like
    'Reprint 2026-27'. (Repeated lines without digits stay: NCERT repeats the chapter name on every page.)"""
    def footer(t: str) -> bool:
        k = re.sub(r"\d+", "#", t.lower())
        return "#" in k and k in furniture
    first = [ln for ln in lines if ln.text and ln.page == (lines[0].page if lines else 1) and not footer(ln.text)]
    cands = [ln for ln in first if sum(ch.isalpha() for ch in ln.text) >= 4
             and not CHAPTER_LINE.match(ln.text) and not re.match(r"^\s*unit\s*\d+", ln.text, re.I)]
    if not cands:
        return ""
    top = max(ln.size for ln in cands)
    parts = [ln.text for ln in cands if ln.size >= top * 0.9][:3]
    return _pretty_heading(" ".join(parts))


def _sections_from_lines(lines: list[Line], furniture: set[str]) -> tuple[str, list[RawSection]]:
    sizes = Counter(round(ln.size) for ln in lines if ln.text and len(ln.text) > 30)
    body = sizes.most_common(1)[0][0] if sizes else 11
    title = _guess_title(lines, furniture)
    sections: list[RawSection] = [RawSection(title="Introduction")]
    buf: list[str] = []
    buf_page = 0
    dropcap = ""
    last_heading_size = 0.0

    def flush() -> None:
        nonlocal buf
        if buf:
            sections[-1].paras.append((" ".join(buf), buf_page))
            buf = []

    for ln in lines:
        t = ln.text
        if not t:
            flush()
            last_heading_size = 0.0
            continue
        if re.sub(r"\d+", "#", t.lower()) in furniture or re.fullmatch(r"\d+|[ivxlc]{2,}", t.lower()):
            continue
        if CHAPTER_LINE.match(t) and len(t) < 20 or (title and len(t) > 3 and t in title and ln.size >= body * 1.5):
            continue
        toks = t.split()
        if len(toks) >= 6 and sum(len(x) == 1 for x in toks) / len(toks) > 0.6:
            continue  # letter-spaced running header ("T h e  R i s e  o f ...")
        letters = sum(ch.isalpha() for ch in t)  # any script: counting only A-Z dropped every Hindi line
        if letters < 3:
            # decorative drop cap ("C" of "Consider") or stray "?" glyphs from margin boxes
            if len(t) == 1 and t.isalpha() and ln.size >= body * 1.8:
                dropcap = t
            continue
        if dropcap:
            t, dropcap = dropcap + t, ""
        if ACTIVITY.match(t):
            flush()
            buf_page = ln.page
            buf.append(_pretty_heading(t) + ":")
            continue
        words = len(t.split())
        is_big = ln.size >= body * 1.2
        is_heading = bool(
            (NUMBERED_HEADING.match(t) and (ln.size >= body * 1.05 or ln.bold))
            or (is_big and words <= 12 and not t.endswith((".", ",", ";")))
        )
        if is_heading:
            # a heading wrapped onto two lines: join to the previous heading
            if last_heading_size and abs(ln.size - last_heading_size) < 0.6 and not buf and not sections[-1].paras:
                sections[-1].title = sections[-1].title + " " + _pretty_heading(t)
                continue
            flush()
            sections.append(RawSection(title=_pretty_heading(t)))
            last_heading_size = ln.size
            continue
        if last_heading_size and not buf and not sections[-1].paras and DANGLING.search(sections[-1].title) and words <= 6:
            sections[-1].title += " " + _pretty_heading(t)  # wrapped heading whose 2nd line is set smaller
            continue
        last_heading_size = 0.0
        if not buf:
            buf_page = ln.page
        if buf and buf[-1].endswith("-") and t[:1].islower():  # de-hyphenate line breaks
            buf[-1] = buf[-1][:-1] + t
        else:
            buf.append(t)
    flush()
    # fold tiny fragments (box titles, "Do You Know?" stubs) into the previous section
    merged: list[RawSection] = []
    for sec in sections:
        wc = sum(len(p.split()) for p, _ in sec.paras)
        if merged and wc < 40 and not SKIP_SECTION.match(sec.title):
            merged[-1].paras += sec.paras
        else:
            merged.append(sec)
    return title, merged


def _sections_from_text(text: str) -> tuple[str, list[RawSection]]:
    title = ""
    sections: list[RawSection] = [RawSection(title="Introduction")]
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        first = lines[0].strip()
        md = re.match(r"^#{1,4}\s+(.*)", first)
        heading = None
        if md:
            heading = md.group(1).strip()
        elif NUMBERED_HEADING.match(first) or (len(first.split()) <= 10 and first.isupper() and len(first) > 3):
            heading = first
        elif len(lines) == 1 and len(first.split()) <= 8 and (first[:1].isupper() or first[:1].isdigit()) and not first.endswith((".", "?", "!", ":", ",")):
            heading = first
        if heading is not None:
            if not title and not sections[-1].paras and len(sections) == 1:
                title = heading.title() if heading.isupper() else heading
                rest = "\n".join(lines[1:]).strip()
                if rest:
                    sections[-1].paras.append((re.sub(r"\s+", " ", rest), 0))
                continue
            sections.append(RawSection(title=heading))
            lines = lines[1:]
        para = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if para:
            sections[-1].paras.append((para, 0))
    return title, sections


def _build_source(title: str, raw: list[RawSection], origin: str, max_words: int, notes: list[str], ocr_pages: list[int]) -> Source:
    chunks: list[Chunk] = []
    sections: list[Section] = []
    total = 0
    skipped: list[str] = []
    for rs in raw:
        if not rs.paras:
            continue
        if SKIP_SECTION.match(rs.title):
            skipped.append(rs.title)
            continue
        sec_id = f"s{len(sections) + 1}"
        ids: list[str] = []
        cur: list[str] = []
        cur_page: int | None = None
        words = 0

        def emit() -> None:  # noqa: B023 - called only within this loop iteration
            nonlocal cur, words
            if cur:
                cid = f"c{len(chunks) + 1}"
                chunks.append(Chunk(id=cid, section=rs.title, text=" ".join(cur), page=cur_page))
                ids.append(cid)
                cur, words = [], 0

        sec_words = 0
        for para, page in rs.paras:
            sentences = re.split(r"(?<=[.!?])\s+", para) if len(para.split()) > CHUNK_WORDS else [para]
            for sent in sentences:
                w = len(sent.split())
                if words and words + w > CHUNK_WORDS:
                    emit()
                if not cur:
                    cur_page = page or None
                cur.append(sent)
                words += w
                sec_words += w
        emit()
        if ids:
            sections.append(Section(id=sec_id, title=rs.title, word_count=sec_words, chunk_ids=ids))
            total += sec_words
    if skipped:
        notes.append("Skipped end-of-chapter material: " + ", ".join(dict.fromkeys(skipped)) + ".")
    if total < 80:
        raise IngestError("We couldn't find enough readable text in this source (need at least a few paragraphs).")
    if total > max_words:
        notes.append(f"This chapter is long ({total} words). Pick a few sections for a focused episode, "
                     f"or we'll use the first ~{max_words} words.")
    return Source(id=new_id("src_"), title=title or (sections[0].title if sections else "Untitled chapter"),
                  origin=origin, sections=sections, chunks=chunks, word_count=total, notes=notes, ocr_pages=ocr_pages)


async def ingest_pdf(data: bytes, max_words: int, origin: str = "upload") -> Source:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise IngestError("That file doesn't look like a valid PDF.") from e
    if doc.page_count == 0:
        raise IngestError("The PDF has no pages.")
    if doc.page_count > 60:
        raise IngestError("Please upload a single chapter (60 pages or fewer).")
    notes: list[str] = []
    lines_by_page = [_page_lines(p, i + 1) for i, p in enumerate(doc)]
    scanned = [i for i, ls in enumerate(lines_by_page) if sum(len(ln.text) for ln in ls) < 80]
    ocr_pages: list[int] = []
    ocr_text: dict[int, str] = {}
    if scanned:
        if ocr_available():
            sem = asyncio.Semaphore(4)

            async def do(i: int) -> None:
                async with sem:
                    try:
                        ocr_text[i] = await ocr_page(render_page_png(doc[i]))
                        ocr_pages.append(i + 1)
                    except Exception as e:  # one bad page shouldn't sink the chapter
                        log.warning("OCR failed on page %d: %s", i + 1, e)

            await asyncio.gather(*(do(i) for i in scanned))
            if ocr_pages:
                notes.append(f"Pages {', '.join(map(str, sorted(ocr_pages)))} were scanned images; text was read with OCR.")
        else:
            notes.append(f"{len(scanned)} page(s) look scanned and OCR is unavailable; they were skipped.")
    if ocr_text and len(ocr_text) == len(lines_by_page):
        title, raw = _sections_from_text("\n\n".join(ocr_text[i] for i in sorted(ocr_text)))
    else:
        furniture = _furniture(lines_by_page)
        all_lines: list[Line] = []
        for i, ls in enumerate(lines_by_page):
            if i in ocr_text:
                all_lines += [Line(text=p.strip(), size=0, bold=False, page=i + 1) for p in ocr_text[i].split("\n")]
            else:
                all_lines += ls
        title, raw = _sections_from_lines(all_lines, furniture)
    if not title:
        meta_title = (doc.metadata or {}).get("title") or ""
        title = meta_title if 3 < len(meta_title) < 120 else ""
    return _build_source(title, raw, origin, max_words, notes, sorted(ocr_pages))


def ingest_text(text: str, max_words: int, origin: str = "paste", title: str | None = None) -> Source:
    text = (text or "").strip()
    if len(text.split()) < 80:
        raise IngestError("Please paste at least a few paragraphs (80+ words) of the chapter.")
    t, raw = _sections_from_text(text)
    return _build_source(title or t, raw, origin, max_words, [], [])


def select_chunks(source: Source, section_ids: list[str], max_words: int) -> tuple[list[Chunk], list[str]]:
    """Chunks for the chosen sections (all if none chosen), capped at ``max_words``."""
    by_id = {c.id: c for c in source.chunks}
    secs = [s for s in source.sections if not section_ids or s.id in section_ids] or source.sections
    out: list[Chunk] = []
    words = 0
    for s in secs:
        for cid in s.chunk_ids:
            w = len(by_id[cid].text.split())
            if out and words + w > max_words:
                return out, [x.title for x in secs]
            out.append(by_id[cid])
            words += w
    return out, [x.title for x in secs]
