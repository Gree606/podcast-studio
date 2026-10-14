"""Runtime configuration, read from environment variables (and an optional .env file).

Every provider-specific value (URLs, keys, model names, voices) lives here so the
rest of the code never touches os.environ directly. Secrets are never logged.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- mode -------------------------------------------------------------
    # MOCK_MODE=1 swaps every external API for a deterministic local fake, so the
    # app, the smoke test and the unit tests run with no network or credentials.
    mock_mode: bool = False

    # --- LLM (OpenAI-compatible chat completions) --------------------------
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "google/gemma-3-27b-it"
    llm_timeout_s: float = 120.0
    llm_max_concurrency: int = 6
    llm_vision: bool = True  # whether the LLM accepts image_url parts (used for OCR fallback)

    # --- TTS ---------------------------------------------------------------
    tts_api_style: str = "openai"  # "openai" (/v1/audio/speech) or "json" (/synthesize, see clients/tts.py)
    tts_base_url: str = ""
    tts_api_key: str = ""
    tts_model: str = ""
    tts_voice_host: str = ""  # curious host
    tts_voice_expert: str = ""  # explainer
    tts_language: str = "en"
    tts_timeout_s: float = 90.0
    tts_max_concurrency: int = 4

    # --- ASR ---------------------------------------------------------------
    asr_base_url: str = ""
    asr_api_key: str = ""
    asr_model: str = ""
    asr_language: str = "en"
    asr_timeout_s: float = 90.0
    asr_max_concurrency: int = 4

    # --- OCR ---------------------------------------------------------------
    # "llm" = send rasterised pages to the vision LLM; "http" = dedicated OCR endpoint; "off".
    ocr_mode: str = "llm"
    ocr_base_url: str = ""
    ocr_api_key: str = ""

    # --- pipeline knobs ------------------------------------------------------
    data_dir: Path = ROOT / "data"
    max_input_words: int = 9000  # cap so one TA upload can't run for an hour
    max_concurrent_episodes: int = 3
    sample_rate: int = 24000
    # Total writer-candidate budget per episode is budget_per_segment * n_segments,
    # redistributed across segments by difficulty (Snell et al. compute-optimal allocation).
    budget_per_segment: float = 2.0
    max_candidates_per_segment: int = 4
    accept_score: float = 4.2  # early-stop threshold on the 1-5 rubric
    wer_threshold: float = 0.25  # ASR self-check: re-synthesise a turn above this WER
    max_tts_retries: int = 2
    planner_max_steps: int = 12

    @property
    def episodes_dir(self) -> Path:
        return self.data_dir / "episodes"

    @property
    def sources_dir(self) -> Path:
        return self.data_dir / "sources"

    def effective_mock(self) -> bool:
        """Fall back to mock mode if no LLM endpoint is configured at all."""
        return self.mock_mode or not self.llm_base_url


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.episodes_dir.mkdir(parents=True, exist_ok=True)
    s.sources_dir.mkdir(parents=True, exist_ok=True)
    return s
