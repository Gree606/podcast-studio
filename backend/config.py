"""Runtime configuration, read from environment variables and two optional env files.

Every provider-specific value (URLs, keys, model names, voices) lives here so the
rest of the code never touches os.environ directly. Secrets are never logged.

Load order (later wins): the repo ``.env`` (non-secret settings), then the secrets
file outside the repo (``~/.config/podcast-studio/secrets.env``, or the path in
``PODCAST_SECRETS_FILE``), then real environment variables.
"""
from __future__ import annotations

import os
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SECRETS_FILE = Path("~/.config/podcast-studio/secrets.env")


def secrets_path() -> Path:
    """Where the API keys live: outside the repo, so nothing working in the repo sees them."""
    return Path(os.environ.get("PODCAST_SECRETS_FILE") or DEFAULT_SECRETS_FILE).expanduser()


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
    # "openai" = any OpenAI-compatible /v1/chat/completions server (e.g. vLLM serving Gemma);
    # "openrouter" = the same models through OpenRouter (https://openrouter.ai/api, model ids like
    # "google/gemini-3.6-flash"): paid per token, so no free-tier daily caps.
    llm_provider: str = "gemini"
    llm_base_url: str = "https://generativelanguage.googleapis.com"
    llm_api_key: str = ""
    # OpenRouter's key, kept separate so switching LLM_PROVIDER back to gemini keeps working
    # (falls back to LLM_API_KEY when empty)
    openrouter_api_key: str = ""
    # Keys for the voice service that is NOT the .env default, so each episode can pick its voices
    # (backend/providers.py). Bodhan voices fall back to ASR_API_KEY: one Bodhan key serves TTS, ASR and OCR.
    sarvam_api_key: str = ""
    bodhan_tts_api_key: str = ""
    llm_model: str = "gemini-3.6-flash"
    # --- Embeddings (deep agent's hybrid retrieval; backend/clients/embed.py) -----------------
    # OpenRouter's /v1/embeddings; the key falls back to OPENROUTER_API_KEY. "" model = keyword search only.
    embed_model: str = "google/gemini-embedding-001"
    embed_base_url: str = "https://openrouter.ai/api"
    embed_api_key: str = ""
    # Tried in order when the main model is overloaded (503), rate-limited (429) or retired (404).
    # Measured 2026-10-21: popular models regularly return 503 "high demand" for minutes at a time.
    llm_fallback_models: str = "gemini-3.5-flash,gemini-flash-latest,gemini-3.5-flash-lite,gemini-flash-lite-latest"
    llm_attempts_per_model: int = 2  # retries on one model before moving to the next
    # per request. A normal call takes 2-32 s; a hung model (seen: 128 s, then 503) should fail over to the next one fast
    llm_timeout_s: float = 45.0
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
    tts_voice_host: str = "dev_en_conversational"  # curious host (Error606, the alien)
    tts_voice_expert: str = "simran_en_conversation"  # explainer (Gree, the astronaut)
    tts_language: str = "en-IN"
    tts_pace: float = 1.05
    tts_timeout_s: float = 90.0
    tts_max_concurrency: int = 4
    tts_max_chars: int = 900  # longer turns are split at sentence boundaries
    # Bodhan indic-speak returns HTTP 429 after a short burst (~7 quick calls, measured 2026-10-23),
    # so the client paces itself; 0 = no pacing (Sarvam doesn't need it).
    tts_requests_per_minute: int = 0
    tts_auth_prefix: str = ""  # bodhan: Authorization header = prefix + key (no "Bearer")

    # --- ASR (self-check of our own audio) -----------------------------------
    # OpenAI-style multipart POST {base}/v1/audio/transcriptions -> {"text": ...}
    asr_base_url: str = "https://api.bodhan.ai"
    asr_api_key: str = ""
    asr_auth_prefix: str = ""  # Authorization header value = prefix + key ("" for Bodhan, "Bearer " for OpenAI)
    asr_model: str = "indic-transcribe"
    asr_language: str = ""  # "" = the TTS language ("en" for en-IN); "auto" = let the service detect it
    asr_timeout_s: float = 90.0
    # Bodhan rate-limits bursts (HTTP 429 seen at 4 parallel calls), so stay gentle and retry longer.
    asr_max_concurrency: int = 2
    asr_attempts: int = 5
    # Measured on api.bodhan.ai (2026-10-21): 14 calls/min sustained is fine; ~30/min gets HTTP 429
    # after ~16 calls. The client paces itself to this many calls in any 60 s window (0 = no pacing).
    asr_requests_per_minute: int = 14
    asr_min_words: int = 6  # shorter lines ("Right!", "Wait, really?") are not self-checked: WER on a few words is noise, and Bodhan allows ~14 calls/min

    # --- OCR ---------------------------------------------------------------
    # "bodhan" = Bodhan Indic-OCR (OpenAI-style /v1/chat/completions, model indic-ocr);
    # "llm" = send rasterised pages to the vision LLM; "http" = dedicated OCR endpoint; "off".
    ocr_mode: str = "llm"
    ocr_base_url: str = ""  # bodhan: defaults to https://api.bodhan.ai
    ocr_api_key: str = ""
    ocr_model: str = "indic-ocr"
    ocr_timeout_s: float = 120.0
    # "all": every PDF page is read with OCR (cleaner than many text layers, e.g. NCERT's Hindi PDFs
    # repeat vowel signs and scramble picture-heavy pages); a page whose OCR fails keeps its own text.
    # "scanned": OCR only pages without a text layer.
    pdf_ocr: str = "all"
    ocr_requests_per_minute: int = 12  # Bodhan OCR: steady pages, never bursts (see clients/ocr.py); 0 = no pacing

    # --- pipeline knobs ------------------------------------------------------
    data_dir: Path = ROOT / "data"
    max_input_words: int = 9000  # cap so one TA upload can't run for an hour
    max_concurrent_episodes: int = 3
    sample_rate: int = 24000
    # Total writer-candidate budget per episode is budget_per_segment * n_segments,
    # redistributed across segments by difficulty (Snell et al. compute-optimal allocation).
    budget_per_segment: float = 1.5
    max_candidates_per_segment: int = 4
    accept_score: float = 4.2  # early-stop threshold on the 1-5 rubric
    wer_threshold: float = 0.25  # ASR self-check: re-synthesise a turn above this WER
    max_tts_retries: int = 2
    planner_max_steps: int = 2  # tool steps before the outline: with the forced finish, <= 3 planner calls; each real-model step can take seconds to a minute; a TA won't wait for 14
    # "fast": 2 LLM calls per episode (write all, then verify+fix); "agent": planner + best-of-N + critic loop
    pipeline_mode: str = "fast"
    # Deep agent: at most this many LLM calls per job (generating, or regenerating one part).
    # Optional steps (extra drafts, revisions, claim fixes) give way first; planning, one draft,
    # its critic and the fact-check of every part always run. 0 = no cap.
    deep_max_llm_calls: int = 10
    # "lean": the deep agent in ~10 calls (agent/lean.py); "per_part": best-of-N per part (25-48 calls, eval ablation)
    agent_style: str = "lean"
    # Secret code for the entry gate. When set, anything that costs API calls (making episodes,
    # uploading chapters, regenerating) needs it; listening to and sharing episodes does not.
    access_code: str = ""
    # Gemini thinking level per fast-pipeline call ("" = the model's default, "low", "high").
    # Checking needs little deliberation; writing benefits from some.
    llm_write_thinking: str = ""
    llm_check_thinking: str = "low"

    @property
    def episodes_dir(self) -> Path:
        return self.data_dir / "episodes"

    @property
    def sources_dir(self) -> Path:
        return self.data_dir / "sources"

    def llm_ready(self) -> bool:
        return bool(self.llm_base_url) and (is_set(self.llm_key()) or self.llm_provider == "openai")

    def llm_key(self) -> str:
        if self.llm_provider == "openrouter" and is_set(self.openrouter_api_key):
            return self.openrouter_api_key
        return self.llm_api_key

    def embed_key(self) -> str:
        return self.embed_api_key if is_set(self.embed_api_key) else self.openrouter_api_key

    def embed_ready(self) -> bool:
        return bool(self.embed_model and self.embed_base_url) and is_set(self.embed_key())

    def tts_ready(self) -> bool:
        return bool(self.tts_base_url) and (is_set(self.tts_api_key) or self.tts_api_style not in ("sarvam", "bodhan"))

    def asr_language_code(self) -> str:
        """Language sent to the ASR. Without it, short English lines ("No way!") came back in
        Telugu or Gujarati script and the self-check had to be skipped."""
        if self.asr_language == "auto":
            return ""
        return self.asr_language or self.tts_language.split("-")[0].lower()

    def asr_ready(self) -> bool:
        return bool(self.asr_base_url) and is_set(self.asr_api_key)

    def ocr_ready(self) -> bool:
        if self.ocr_mode == "bodhan":
            return is_set(self.ocr_api_key)
        if self.ocr_mode == "http":
            return bool(self.ocr_base_url)
        return self.ocr_mode == "llm" and self.llm_ready() and self.llm_vision

    def configured(self) -> dict[str, bool]:
        """Which providers have real-looking keys (never the keys themselves)."""
        return {"llm": self.llm_ready(), "tts": self.tts_ready(), "asr": self.asr_ready(), "ocr": self.ocr_ready()}

    # Each service is real as soon as its own key is set, so e.g. real voices work while
    # the LLM key is still a placeholder. MOCK_MODE=1 forces every service to the local fake.
    def llm_mock(self) -> bool:
        return self.mock_mode or not self.llm_ready()

    def tts_mock(self) -> bool:
        return self.mock_mode or not self.tts_ready()

    def asr_mock(self) -> bool:
        """The ASR only checks our own audio: placeholder voices get the placeholder checker."""
        return self.tts_mock()

    def effective_mock(self) -> bool:
        """Demo mode (shown in the UI) unless both the LLM and TTS are configured with real keys.
        (ASR is optional: without it the audio self-check is skipped.)"""
        return self.llm_mock() or self.tts_mock()


@lru_cache
def _base_settings() -> Settings:
    s = Settings(_env_file=(ROOT / ".env", secrets_path()))
    s.episodes_dir.mkdir(parents=True, exist_ok=True)
    s.sources_dir.mkdir(parents=True, exist_ok=True)
    return s


# An episode's own provider choices (backend/providers.py) for the code running on its behalf.
# A ContextVar is per asyncio task, so episodes made with different providers can run side by side.
_override: ContextVar[Settings | None] = ContextVar("settings_override", default=None)


def get_settings() -> Settings:
    return _override.get() or _base_settings()


get_settings.cache_clear = _base_settings.cache_clear  # type: ignore[attr-defined]
