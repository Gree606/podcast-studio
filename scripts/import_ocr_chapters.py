"""Add NCERT chapters in Indian languages to the bundled samples, from an OCR'd page corpus.

The corpus has one JPEG per textbook page, layout-aware OCR per page (Bodhan Indic-OCR:
blocks labelled Paragraph / Section-title / Footer / ...) and a metadata CSV mapping pages to
books and chapters. For each chapter listed in CHAPTERS this writes:

  samples/<id>.md                 the chapter text: headings as "##", running heads, page
                                  numbers, pictures, questions and exercises left out
  samples/samples.json            a card for it (title, class, subject, language, source)
  samples/textbooks/<id>.pdf      the original pages, downscaled, so anyone can check the
                                  episode against the book

Usage: python scripts/import_ocr_chapters.py /path/to/ncert-scert
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

# id, book_id prefix, chapter number, language code (backend/languages.py), and the card text
CHAPTERS = [
    dict(id="hi-magnets", book="455e5ba5", chapter="4.0", language="hi", subject="Science", cls="6",
         book_name="जिज्ञासा (Curiosity)", title="चुम्बकों को जानें", blurb="Exploring Magnets: poles, compasses and how magnets attract and repel."),
    dict(id="hi-oceans-continents", book="ce81b75b", chapter="2.0", language="hi", subject="Social Science", cls="6",
         book_name="समाज का अध्ययन: भारत और उसके आगे (Exploring Society: India and Beyond)", title="महासागर एवं महाद्वीप",
         blurb="Oceans and Continents: how land and water share the Earth, oceans, continents and islands."),
    dict(id="ml-states-of-water", book="0af48b49", chapter="8.0", language="ml", subject="Science", cls="6",
         book_name="ക്യുരിയോസിറ്റി (Curiosity)", title="ജലാവസ്ഥകളിലൂടെയുള്ള ഒരു യാത്ര",
         blurb="A Journey through States of Water: solid, liquid, gas, evaporation and the water cycle."),
    dict(id="ml-india-bharat", book="f9eda87f", chapter="5.0", language="ml", subject="Social Science", cls="6",
         book_name="സാമൂഹത്തെ കുറിച്ച് അറിയുക: ഇന്ത്യയും അതിനപ്പുറവും (Exploring Society: India and Beyond)", title="ഇന്ത്യ, അതാണ് ഭാരതം",
         blurb="India, that is Bharat: the many names of our land and where they come from."),
]
LANG_NAMES = {"hi": "Hindi", "ml": "Malayalam", "ta": "Tamil"}

HEADINGS = {"Section-title", "Sub-section-title"}
TEXT = {"Paragraph", "List", "Infobox"}  # left out: Header, Footer, Image, captions, Table, Question, MCQ, ...
PDF_WIDTH = 1000  # px; enough to read the page, small enough to keep in git


def page_no(image_name: str) -> int:
    return int(re.search(r"Page(\d+)\.jpg$", image_name).group(1))


def chapter_pages(corpus: Path, book: str, chapter: str) -> list[str]:
    with open(corpus / "combined_metadata.csv", newline="", encoding="utf-8") as f:
        names = {r["image_name"] for r in csv.DictReader(f)
                 if r["book_id"].startswith(book) and r["chapter_number"] == chapter}
    return sorted(names, key=page_no)


# NCERT print codes ("0678CH04") and figure captions that the OCR labelled as paragraphs
CLUTTER = re.compile(r"^(\d{4}CH\d{2}|(चित्र|ചിത്രം|படம்|Fig)\.?\s*\d+(\.\d+)*.*)$")
# The end-of-chapter exercises start here; like the PDF importer, episodes don't read them out.
EXERCISES = re.compile(r"आइए, और अधिक सीखें|प्रश्न, क्रियाकलाप|നമുക്ക് നമ്മുടെ പഠനം|ചോദ്യങ്ങൾ, പ്രവർത്തനങ്ങൾ|Let us enhance|Questions, activities")


def chapter_markdown(corpus: Path, title: str, pages: list[str]) -> str:
    blocks = []
    for name in pages:
        p = corpus / "jsons-bodhan" / (name[:-4] + ".json")
        if p.exists():
            blocks.append([(b["label"], re.sub(r"\s+", " ", b.get("text") or "").strip())
                           for b in sorted(json.loads(p.read_text(encoding="utf-8"))["blocks"], key=lambda b: b["order"])])
    # running heads/feet: short lines printed on several pages
    seen: dict[str, int] = {}
    for page in blocks:
        for t in {t for _, t in page if len(t.split()) <= 15}:
            seen[t] = seen.get(t, 0) + 1
    out = [f"# {title}", ""]
    for page in blocks:
        for label, text in page:
            if not text or re.fullmatch(r"[\d\s.]+", text) or CLUTTER.match(text) or seen.get(text, 0) >= 2:
                continue
            if label in HEADINGS and EXERCISES.search(text):
                return "\n".join(out).strip() + "\n"
            if label in HEADINGS:
                out += [f"## {text}", ""]
            elif label in TEXT:
                out += [text, ""]
    return "\n".join(out).strip() + "\n"


def pages_pdf(corpus: Path, pages: list[str], dest: Path) -> None:
    imgs = []
    for name in pages:
        im = Image.open(corpus / "images" / name).convert("RGB")
        im = im.resize((PDF_WIDTH, round(im.height * PDF_WIDTH / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=60, optimize=True)  # recompress before the PDF embeds it
        imgs.append(Image.open(io.BytesIO(buf.getvalue())))
    dest.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(dest, "PDF", save_all=True, append_images=imgs[1:], resolution=100)


def main(corpus: Path) -> None:
    meta_path = SAMPLES / "samples.json"
    meta = [m for m in json.loads(meta_path.read_text(encoding="utf-8")) if m["id"] not in {c["id"] for c in CHAPTERS}]
    for c in CHAPTERS:
        pages = chapter_pages(corpus, c["book"], c["chapter"])
        md = chapter_markdown(corpus, c["title"], pages)
        (SAMPLES / f"{c['id']}.md").write_text(md, encoding="utf-8")
        pages_pdf(corpus, pages, SAMPLES / "textbooks" / f"{c['id']}.pdf")
        lang = LANG_NAMES[c["language"]]
        meta.append({"id": c["id"], "title": c["title"], "title_en": c["blurb"].split(":")[0], "subject": c["subject"], "class": c["cls"],
                     "language": c["language"],
                     "blurb": c["blurb"], "file": f"{c['id']}.md",
                     "source": f"NCERT Class {c['cls']} {c['subject']} in {lang}, {c['book_name']}, chapter {int(float(c['chapter']))} (pp. {page_no(pages[0])}-{page_no(pages[-1])}); text by Bodhan Indic-OCR",
                     "words": len(md.split())})
        print(f"{c['id']}: {len(pages)} pages, {len(md.split())} words")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
