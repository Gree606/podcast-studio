"""Focused acceptance tests for GARGANTUA_PLAN.md criteria.
Runs offline in mock mode, CPU only, without network dependencies.
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend import config as config_mod
from backend.config import get_settings
from backend.ingest import ingest_text
from backend.main import app
from backend.quiz import build_quiz
from backend.schemas import Episode, EpisodeSettings, KeyFact, Segment

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "samples"


# ---------------------------------------------------------------------------
# Acceptance Criteria 2 & 5: Secrets loading priority & config masking
# ---------------------------------------------------------------------------
def test_secrets_file_precedence_and_api_masking(monkeypatch, tmp_path):
    """Criterion 2 & 5: PODCAST_SECRETS_FILE takes precedence over repo .env,
    and /api/config reports only booleans without leaking secret values.
    """
    secret_key = "AIzaSy_GEMINI_SUPER_SECRET_123"
    secrets_file = tmp_path / "secrets.env"
    secrets_file.write_text(f"LLM_API_KEY={secret_key}\nTTS_API_KEY=DUMMY_SARVAM\n")

    monkeypatch.setenv("PODCAST_SECRETS_FILE", str(secrets_file))
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert s.llm_api_key == secret_key
        assert s.llm_ready() is True
        assert s.tts_ready() is False

        with TestClient(app) as client:
            res = client.get("/api/config")
            assert res.status_code == 200
            data = res.json()
            assert data["secrets_path"] == str(secrets_file)
            assert data["configured"]["llm"] is True
            assert data["configured"]["tts"] is False
            assert secret_key not in res.text
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Acceptance Criterion 6: Kid presets contract
# ---------------------------------------------------------------------------
def test_kid_presets_contract():
    """Criterion 6: /api/config returns length and tone presets with kid labels,
    and all preset values must be accepted by EpisodeSettings.
    """
    with TestClient(app) as client:
        res = client.get("/api/config")
        assert res.status_code == 200
        presets = res.json().get("presets", {})

    assert "length" in presets and "tone" in presets
    for item in presets["length"]:
        assert item["label"] in ["Quick Bite", "Story Time", "Deep Dive"]
        s = EpisodeSettings(length=item["value"])
        assert s.length == item["value"]

    for item in presets["tone"]:
        assert item["label"] in ["Fun", "Balanced", "Exam Hero"]
        s = EpisodeSettings(tone=item["value"])
        assert s.tone == item["value"]


# ---------------------------------------------------------------------------
# Acceptance Criterion 7: Static file delivery & classic fallback
# ---------------------------------------------------------------------------
def test_static_routes_headers_and_fallback():
    """Criterion 7: GET / serves new UI; /app.css and /app.js serve correct media types;
    and /classic serves the fallback UI.
    """
    with TestClient(app) as client:
        r_index = client.get("/")
        assert r_index.status_code == 200
        assert r_index.headers["content-type"].startswith("text/html")
        assert "Chapter Chat" in r_index.text

        r_css = client.get("/app.css")
        assert r_css.status_code == 200
        assert r_css.headers["content-type"].startswith("text/css")

        r_js = client.get("/app.js")
        assert r_js.status_code == 200
        assert "javascript" in r_js.headers["content-type"]

        r_classic = client.get("/classic")
        assert r_classic.status_code == 200
        assert r_classic.headers["content-type"].startswith("text/html")
        assert "Podcast Studio" in r_classic.text


# ---------------------------------------------------------------------------
# Acceptance Criterion 8: Teacher view toggle contract
# ---------------------------------------------------------------------------
def test_teacher_view_css_and_dom_contract():
    """Criterion 8: Teacher panels (trace, quality, claims, sources, versions)
    are present in the DOM and conditioned on body.teacher in app.css.
    """
    index_html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    app_css = (ROOT / "frontend" / "app.css").read_text(encoding="utf-8")

    for panel in ("teacherTrace", "teacherQuality", "teacherClaims", "teacherSources", "teacherVersions"):
        assert f'id="{panel}"' in index_html

    assert ".teacher-only { display: none; }" in app_css
    assert "body.teacher .teacher-only { display: block; }" in app_css


# ---------------------------------------------------------------------------
# Acceptance Criterion 10: Quiz generation grounding & distractor filtering
# ---------------------------------------------------------------------------
def test_quiz_grounding_and_distractor_exclusion():
    """Criterion 10: Quiz questions cite verified key facts only, answer is in fact,
    no distractor appears in cited passage, and fewer than 3 questions drops quiz.
    """
    raw_md = (SAMPLES / "chem-reactions.md").read_text(encoding="utf-8")
    src = ingest_text(raw_md, 9000, origin="sample")
    by_id = {c.id: c for c in src.chunks}

    chunks = src.chunks[:6]
    segs = [
        Segment(
            id=f"seg_{i}",
            index=i,
            title=f"Segment {i}",
            goal="Goal",
            chunk_ids=[c.id],
            key_facts=[KeyFact(fact=c.text.split(".")[0].strip() + ".", chunk_id=c.id, verified=True)],
        )
        for i, c in enumerate(chunks[:4])
    ]
    ep = Episode(id="ep_quiz_test", source_id=src.id, title=src.title, settings=EpisodeSettings(), segments=segs)

    quiz = build_quiz(ep, src)
    if quiz:
        assert len(quiz) >= 3
        for q in quiz:
            assert q["chunk_id"] in by_id
            passage = by_id[q["chunk_id"]].text.lower()
            assert q["answer"].lower() in q["fact"].lower()
            for opt in q["options"]:
                if opt != q["answer"]:
                    assert opt.lower() not in passage, f"Distractor {opt} found in passage!"

    # Unverified facts must never generate quiz questions
    unverified_segs = [
        Segment(
            id="seg_unverified",
            index=0,
            title="Unverified",
            goal="Goal",
            chunk_ids=[chunks[0].id],
            key_facts=[KeyFact(fact="Rusting of iron is common.", chunk_id=chunks[0].id, verified=False)],
        )
    ]
    ep_unverified = Episode(id="ep_uv", source_id=src.id, title=src.title, settings=EpisodeSettings(), segments=unverified_segs)
    assert build_quiz(ep_unverified, src) == []


# ---------------------------------------------------------------------------
# Acceptance Criterion 12: Sanitized error messages (no stack traces)
# ---------------------------------------------------------------------------
def test_error_responses_contain_no_tracebacks():
    """Criterion 12: Bad requests and server errors return user-readable messages
    without leaking Python tracebacks or stack traces.
    """
    with TestClient(app, raise_server_exceptions=False) as client:
        r_val = client.post("/api/episodes", json={"source_id": 12345, "settings": "invalid"})
        assert r_val.status_code == 422
        assert "Traceback" not in r_val.text
        assert 'File "' not in r_val.text

        r_404 = client.get("/api/episodes/nonexistent_id")
        assert r_404.status_code == 404
        assert "Traceback" not in r_404.text
        assert 'File "' not in r_404.text
