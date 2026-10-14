"""Convert NCERT chapter PDFs into the bundled sample chapters (samples/*.md + samples.json).

Samples are stored as cleaned markdown (headings as '## ...'), not PDFs, so the repo
stays small and the "Try a sample" button responds instantly. Optionally limit each
sample to its first N sections so a TA's first episode is quick.

Usage: python scripts/make_samples.py <pdf_dir>
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.ingest import ingest_pdf  # noqa: E402

SAMPLES = [
    # id, pdf stem, title, subject, class, blurb, max sections
    ("chem-reactions", "jesc101", "Chemical Reactions and Equations", "Science", "10", "Rusting, rancidity and how to balance an equation.", 9),
    ("control-coordination", "jesc106", "Control and Coordination", "Science", "10", "Reflexes, the brain, plant movements and hormones.", 8),
    ("crop-production", "hesc101", "Crop Production and Management", "Science", "8", "From kharif and rabi crops to irrigation and storage.", 10),
    ("nationalism-europe", "jess301", "The Rise of Nationalism in Europe", "History", "10", "How nation-states were imagined and built in the 1800s.", 7),
    ("nutrition-plants", "gesc101", "Nutrition in Plants", "Science", "7", "Photosynthesis, parasites, and insect-eating plants.", 6),
]


async def main(pdf_dir: Path) -> None:
    out_dir = ROOT / "samples"
    out_dir.mkdir(exist_ok=True)
    meta = []
    for sid, stem, title, subject, klass, blurb, max_secs in SAMPLES:
        pdf = pdf_dir / f"{stem}.pdf"
        if not pdf.exists():
            print("missing", pdf)
            continue
        src = await ingest_pdf(pdf.read_bytes(), 100000)
        by_id = {c.id: c for c in src.chunks}
        lines = [f"# {title}", ""]
        for sec in src.sections[:max_secs]:
            lines += [f"## {sec.title}", ""]
            for cid in sec.chunk_ids:
                lines += [by_id[cid].text, ""]
        (out_dir / f"{sid}.md").write_text("\n".join(lines), encoding="utf-8")
        words = sum(s.word_count for s in src.sections[:max_secs])
        meta.append({"id": sid, "title": title, "subject": subject, "class": klass, "blurb": blurb,
                     "file": f"{sid}.md", "source": f"NCERT {stem}.pdf", "words": words})
        print(f"{sid}: {words} words, {min(max_secs, len(src.sections))} sections")
    (out_dir / "samples.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
