"""Photo-to-podcast: image preparation, one OCR call per page in order, limits, API round trip."""
import asyncio
import io

import pytest
from PIL import Image, ImageDraw

from backend import photos
from backend.ingest import IngestError


def _jpeg(w=1200, h=1600, exif_rotate=False, fmt="JPEG"):
    img = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(img).text((50, 50), "Photosynthesis happens in leaves", fill="black")
    buf = io.BytesIO()
    if exif_rotate:  # stored landscape, EXIF says "rotate 90" -> upright portrait after transpose
        exif = Image.Exif()
        exif[0x0112] = 6
        img.save(buf, format=fmt, exif=exif)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def test_prepare_photo_straightens_downscales_and_reencodes():
    out = Image.open(io.BytesIO(photos.prepare_photo(_jpeg(4000, 3000, exif_rotate=True), "Photo 1")))
    assert out.format == "JPEG"
    assert max(out.size) <= photos.LONG_SIDE
    assert out.size[1] > out.size[0]  # EXIF orientation applied: now portrait


def test_prepare_photo_accepts_png_and_rejects_bad_input():
    assert photos.prepare_photo(_jpeg(800, 1000, fmt="PNG"), "Photo 1")[:2] == b"\xff\xd8"  # PNG in, JPEG out
    with pytest.raises(IngestError, match="isn't an image"):
        photos.prepare_photo(b"not an image at all", "Photo 1")
    with pytest.raises(IngestError, match="too small"):
        photos.prepare_photo(_jpeg(300, 200), "Photo 1")


def test_clean_page_text_drops_page_numbers_and_table_rules():
    assert photos.clean_page_text("# Title\n\n12\n|---|---|\nReal text here.\n\n\n\nMore.") == "# Title\n\nReal text here.\n\nMore."


def test_ingest_photos_one_ocr_call_per_page_in_order(monkeypatch):
    calls = []

    async def fake_ocr(img, mime="image/png"):
        calls.append(mime)
        n = len(calls)
        return f"# Page {n} heading\n\n" + " ".join(f"word{n}_{i}" for i in range(60))

    monkeypatch.setattr(photos, "ocr_page", fake_ocr)
    monkeypatch.setattr(photos, "ocr_available", lambda: True)
    src = asyncio.run(photos.ingest_photos([("a.jpg", _jpeg()), ("b.jpg", _jpeg()), ("c.png", _jpeg(fmt="PNG"))], 9000))
    assert calls == ["image/jpeg"] * 3  # exactly one request per page, always JPEG
    assert src.origin == "photo" and src.ocr_pages == [1, 2, 3] and len(src.page_texts) == 3
    assert [t.split("\n")[0] for t in src.page_texts] == ["# Page 1 heading", "# Page 2 heading", "# Page 3 heading"]
    assert any("one page per photo" in n for n in src.notes)


def test_ingest_photos_limits_and_unreadable(monkeypatch):
    monkeypatch.setattr(photos, "ocr_available", lambda: True)

    async def few_words(img, mime="image/png"):
        return "blurry"

    monkeypatch.setattr(photos, "ocr_page", few_words)
    with pytest.raises(IngestError, match="only read a few words"):
        asyncio.run(photos.ingest_photos([("a.jpg", _jpeg())], 9000))
    with pytest.raises(IngestError, match="at most 6"):
        asyncio.run(photos.ingest_photos([("x.jpg", b"")] * 7, 9000))
    monkeypatch.setattr(photos, "ocr_available", lambda: False)
    with pytest.raises(IngestError, match="OCR service"):
        asyncio.run(photos.ingest_photos([("a.jpg", _jpeg())], 9000))


def test_api_accepts_photos(monkeypatch):
    from fastapi.testclient import TestClient

    from backend.main import app

    async def fake_ocr(img, mime="image/png"):
        return "# Nutrition in Plants\n\n" + "Plants make their own food by photosynthesis using sunlight. " * 12

    monkeypatch.setattr(photos, "ocr_page", fake_ocr)
    with TestClient(app) as c:
        r = c.post("/api/sources", files=[("images", ("p1.jpg", _jpeg(), "image/jpeg")), ("images", ("p2.jpg", _jpeg(), "image/jpeg"))])
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["origin"] == "photo" and d["ocr_pages"] == [1, 2] and len(d["page_texts"]) == 2
        assert d["title"] == "Nutrition in Plants"
        bad = c.post("/api/sources", files=[("images", ("x.jpg", b"nope", "image/jpeg"))])
        assert bad.status_code == 422 and "image" in bad.json()["detail"]


def test_furniture_from_real_ncert_photo_is_removed():
    """Regression from a real Bodhan OCR run on an NCERT page photo."""
    raw = "Report 2026-27\n\n$$101$$\n\nControl and Coordination\n\nIs there a difference in taste?\n\n(D)\n\nFigure 6.1 A neuron"
    assert photos.clean_page_text(raw) == "Control and Coordination\n\nIs there a difference in taste?"
    pages = ["Science\nUnit header\n\nBody one.", "Unit header\n\nBody two."]
    assert photos.drop_repeated_headers([photos.clean_page_text(p) for p in pages]) == ["Unit header\n\nBody one.", "Body two."]
