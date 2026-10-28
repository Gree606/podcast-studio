"""Focused coverage for OCR credentials loaded through the configured secrets file."""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend import config as config_mod
from backend.config import get_settings
from backend.main import app


def prepare_config(monkeypatch, tmp_path, *, secret_key: str) -> None:
    """Use isolated config paths so tests never inspect the owner's secrets."""
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    (fake_root / ".env").write_text(
        "OCR_MODE=bodhan\nOCR_API_KEY=DUMMY_FROM_REPO_ENV\n",
        encoding="utf-8",
    )
    secret_file = tmp_path / "secrets.env"
    secret_file.write_text(f"OCR_API_KEY={secret_key}\n", encoding="utf-8")

    monkeypatch.setattr(config_mod, "ROOT", fake_root)
    monkeypatch.setenv("PODCAST_SECRETS_FILE", str(secret_file))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    for name in ("LLM_API_KEY", "TTS_API_KEY", "ASR_API_KEY", "OCR_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()


def test_ocr_key_load_order_and_environment_override(monkeypatch, tmp_path):
    # Acceptance criterion 2: secrets-file values override repo .env values; real environment overrides both.
    prepare_config(monkeypatch, tmp_path, secret_key="test-ocr-file-sentinel")
    try:
        assert get_settings().ocr_api_key == "test-ocr-file-sentinel"

        monkeypatch.setenv("OCR_API_KEY", "test-ocr-env-override")
        get_settings.cache_clear()
        assert get_settings().ocr_api_key == "test-ocr-env-override"
    finally:
        get_settings.cache_clear()


def test_ocr_secret_is_not_returned_by_config_or_health(monkeypatch, tmp_path):
    # Acceptance criterion 3: a sentinel provider key is absent from API responses.
    sentinel = "test-ocr-response-sentinel"
    prepare_config(monkeypatch, tmp_path, secret_key=sentinel)
    try:
        with TestClient(app) as client:
            responses = [
                client.get("/api/config"),
                client.get("/api/health"),
                client.get("/api/episodes/test-missing-episode"),
            ]
        assert all(response.status_code in (200, 404) for response in responses)
        assert sentinel not in "\n".join(response.text for response in responses)
    finally:
        get_settings.cache_clear()
