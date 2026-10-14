"""Runtime configuration, read from environment variables (and an optional .env file).

Every provider-specific value (URLs, keys, model names, voices) lives here so the
rest of the code never touches os.environ directly. Secrets are never logged.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


_PLACEHOLDERS = ("dummy", "your", "change", "replace", "xxx", "todo", "<")


def is_set(value: str) -> bool:
    """True for a real-looking secret; False for empty or placeholder values like DUMMY_REPLACE_ME."""
    v = (value or "").strip().lower()
    return bool(v) and not v.startswith(_PLACEHOLDERS)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- mode -------------------------------------------------------------
    # MOCK_MODE=1 swaps every external API for a deterministic local fake, so the
    # app, the smoke test and the unit tests run with no network or credentials.
    mock_mode: bool = False

    # --- LLM ----------------------------------------------------------------
    # "gemini" = Google Generative Language API (generateContent);
    # "openai" = any OpenAI-compatible /v1/chat/completions server (e.g. vLLM serving Gemma).
    llm_provider: str = "gemini"
    llm_base_url: str = "https://generativelanguage.googleapis.com"
    llm_api_key: str = ""
    llm_model: str = "gemini-flash-latest"
    llm_timeout_s: float = 120.0
    llm_max_concurrency: int = 4
    llm_attempts: int = 5  # retries cover 429 rate limits (we honour the server's retry delay)
    # Gemini "thinking" tokens count against maxOutputTokens; this much headroom is added
    # on top of each call's answer budget so thinking can't starve the answer.
    llm_thinking_headroom: int = 4096
    llm_vision: bool = True  # whether the LLM accepts images (used for OCR of scanned pages)

    # --- TTS ----------------------------------------------------------------
    # "sarvam" = Sarvam AI /text-to-speech; "openai" = /v1/audio/speech; "json" = /synthesize.
    tts_api_style: str = "sarvam"
    tts_base_url: str = "https://api.sarvam.ai"
    tts_api_key: str = ""
    tts_model: str = "bulbul:v4-flash"
    tts_voice_host: str = "simran_en_conversation"  # curious host (Riya)
    tts_voice_expert: str = "dev_en_conversational"  # explainer (Arjun)
    tts_language: str = "en-IN"
    tts_pace: float = 1.05
    tts_timeout_s: float = 90.0
    tts_max_concurrency: int = 4
    tts_max_chars: int = 900  # longer turns are split at sentence boundaries

    # --- ASR (self-check of our own audio) -----------------------------------
    # OpenAI-style multipart POST {base}/v1/audio/transcriptions -> {"text": ...}
    asr_base_url: str = "https://api.bodhan.ai"
    asr_api_key: str = ""
    asr_auth_prefix: str = ""  # Authorization header value = prefix + key ("" for Bodhan, "Bearer " for OpenAI)
    asr_model: str = "indic-transcribe"
    asr_language: str = ""  # sent only if set
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

    def llm_ready(self) -> bool:
        return bool(self.llm_base_url) and (is_set(self.llm_api_key) or self.llm_provider == "openai")

    def tts_ready(self) -> bool:
        return bool(self.tts_base_url) and (is_set(self.tts_api_key) or self.tts_api_style != "sarvam")

    def asr_ready(self) -> bool:
        return bool(self.asr_base_url) and is_set(self.asr_api_key)

    def effective_mock(self) -> bool:
        """Demo mode unless both the LLM and TTS are configured with real keys.
        (ASR is optional: without it the audio self-check is skipped.)"""
        return self.mock_mode or not (self.llm_ready() and self.tts_ready())


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.episodes_dir.mkdir(parents=True, exist_ok=True)
    s.sources_dir.mkdir(parents=True, exist_ok=True)
    return s
