"""Secrets live in a file outside the repo; keys are loaded but never exposed (plan criteria 2-5)."""
import json
import logging
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import config as config_mod
from backend import main as main_mod
from backend.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "AIzaSySENTINEL0123456789abcdef"


@pytest.fixture
def settings_from(monkeypatch, tmp_path):
    """Build fresh Settings from a temp repo .env and a temp secrets file."""
    fake_root = tmp_path / "repo"
    fake_root.mkdir()

    def make(secrets: str | None = None, dotenv: str | None = None, env: dict | None = None):
        path = tmp_path / "secrets.env"
        if secrets is not None:
            path.write_text(secrets)
        if dotenv is not None:
            (fake_root / ".env").write_text(dotenv)
        monkeypatch.setattr(config_mod, "ROOT", fake_root)
        monkeypatch.setenv("PODCAST_SECRETS_FILE", str(path))
        for k in ("LLM_API_KEY", "TTS_API_KEY", "ASR_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        get_settings.cache_clear()
        return get_settings()

    yield make
    get_settings.cache_clear()


# ---------------------------------------------------------------- criterion 2
def test_secrets_file_overrides_repo_env(settings_from):
    s = settings_from(secrets="LLM_API_KEY=from-secrets\nTTS_API_KEY=tts-from-secrets\n",
                      dotenv="LLM_API_KEY=from-dotenv\nASR_API_KEY=asr-from-dotenv\nLLM_MODEL=model-from-dotenv\n")
    assert s.llm_api_key == "from-secrets"
    assert s.tts_api_key == "tts-from-secrets"
    assert s.asr_api_key == "asr-from-dotenv"  # repo .env still supplies what the secrets file doesn't
    assert s.llm_model == "model-from-dotenv"


def test_environment_variable_overrides_both(settings_from):
    s = settings_from(secrets="LLM_API_KEY=from-secrets\n", dotenv="LLM_API_KEY=from-dotenv\n", env={"LLM_API_KEY": "from-env"})
    assert s.llm_api_key == "from-env"


def test_missing_secrets_file_is_fine(settings_from, tmp_path):
    s = settings_from()
    assert not (tmp_path / "secrets.env").exists()
    assert s.llm_api_key == ""


def test_default_secrets_path_is_outside_the_repo(monkeypatch):
    monkeypatch.delenv("PODCAST_SECRETS_FILE", raising=False)
    p = config_mod.secrets_path()
    assert p == Path("~/.config/podcast-studio/secrets.env").expanduser()
    assert ROOT not in p.parents


# ---------------------------------------------------------------- criterion 3
def test_sentinel_key_never_appears_in_responses_or_logs(settings_from, monkeypatch, caplog):
    settings_from(secrets=f"LLM_API_KEY={SENTINEL}\nTTS_API_KEY={SENTINEL}tts\nASR_API_KEY={SENTINEL}asr\n")
    assert get_settings().llm_api_key == SENTINEL

    def broken():
        raise RuntimeError("samples file is corrupt")

    caplog.set_level(logging.DEBUG)
    texts = []
    with TestClient(main_mod.app, raise_server_exceptions=False) as c:
        for r in (c.get("/api/health"), c.get("/api/config"), c.get("/api/episodes/ep_doesnotexist"),
                  c.post("/api/episodes", json={"source_id": 5, "settings": {"length": "huge"}}),
                  c.post("/api/sources", data={}), c.get("/api/sources/nope")):
            texts.append(r.text)
            assert r.status_code in (200, 400, 404, 422)
        monkeypatch.setattr(main_mod, "_samples", broken)
        r = c.get("/api/samples")
        assert r.status_code == 500 and "Traceback" not in r.text
        texts.append(r.text)
    blob = "\n".join(texts) + "\n" + caplog.text
    assert SENTINEL not in blob
    assert "samples file is corrupt" in caplog.text  # the logs did capture the failure, just not the key


# ---------------------------------------------------------------- criterion 4
def run_init(path: Path) -> str:
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "init_secrets.py"), "--path", str(path)],
                       capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return r.stdout + r.stderr


def test_init_secrets_creates_owner_only_template_and_never_overwrites(tmp_path):
    path = tmp_path / "cfg" / "secrets.env"
    out1 = run_init(path)
    assert path.exists()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    body = path.read_text()
    values = {}
    for line in body.splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            values[k] = v
    assert set(values) == {"LLM_API_KEY", "TTS_API_KEY", "ASR_API_KEY", "OCR_API_KEY"}
    assert all(v.startswith("DUMMY_") for v in values.values())
    assert not any(v in out1 for v in values.values())

    real = "LLM_API_KEY=AIzaSyOWNERREALKEY1\nTTS_API_KEY=sk_OWNERREALKEY2\nASR_API_KEY=OWNERREALKEY3\n"
    path.write_text(real)
    before = path.read_bytes()
    out2 = run_init(path)
    assert path.read_bytes() == before
    for secret in ("AIzaSyOWNERREALKEY1", "sk_OWNERREALKEY2", "OWNERREALKEY3"):
        assert secret not in out2
    assert "unchanged" in out2


# ---------------------------------------------------------------- criterion 5
def test_config_reports_configured_flags_only_never_paths(settings_from):
    placeholders = "LLM_API_KEY=DUMMY_REPLACE_WITH_GEMINI_KEY\nTTS_API_KEY=DUMMY_REPLACE_WITH_SARVAM_KEY\nASR_API_KEY=DUMMY_REPLACE_WITH_BODHAN_KEY\n"
    settings_from(secrets=placeholders)
    with TestClient(main_mod.app) as c:
        cfg = c.get("/api/config").json()
    assert "secrets_path" not in cfg and os.environ["PODCAST_SECRETS_FILE"] not in json.dumps(cfg)
    assert cfg["configured"] == {"llm": False, "tts": False, "asr": False, "ocr": False}

    settings_from(secrets="LLM_API_KEY=AIzaSyRealLooking123\nTTS_API_KEY=sk_real_looking_456\nASR_API_KEY=bodhan_real_789\n"
                          "OCR_API_KEY=bodhan_ocr_000\n", env={"OCR_MODE": "bodhan"})
    with TestClient(main_mod.app) as c:
        r = c.get("/api/config")
    assert r.json()["configured"] == {"llm": True, "tts": True, "asr": True, "ocr": True}
    for secret in ("AIzaSyRealLooking123", "sk_real_looking_456", "bodhan_real_789", "bodhan_ocr_000"):
        assert secret not in r.text
