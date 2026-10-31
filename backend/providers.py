"""Which script AI and which voices an episode uses, chosen per episode in the app's Settings.

The repo ``.env`` sets the server default. The other provider of each kind is described here
and applied on top of it for the episodes that pick it, so switching never needs a restart
and one person's choice never changes anyone else's episode. An option is offered only
when its key is set; keys themselves never leave the server.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator

from .config import Settings, _base_settings, _override, is_set

GEMINI_URL = "https://generativelanguage.googleapis.com"
OPENROUTER_URL = "https://openrouter.ai/api"

LLM_CHOICES: dict[str, dict] = {
    "gemini": {"label": "Gemini", "hint": "Google AI Studio key", "settings": {
        "llm_provider": "gemini", "llm_base_url": GEMINI_URL, "llm_model": "gemini-3.6-flash",
        "llm_fallback_models": "gemini-3.5-flash,gemini-flash-latest,gemini-3.5-flash-lite,gemini-flash-lite-latest"}},
    "openrouter": {"label": "Gemini via OpenRouter", "hint": "paid per token, no daily cap", "settings": {
        "llm_provider": "openrouter", "llm_base_url": OPENROUTER_URL, "llm_model": "google/gemini-3.6-flash",
        "llm_fallback_models": "google/gemini-3.5-flash,google/gemini-3.7-flash,google/gemini-3.5-flash-lite"}},
}

# Voices picked by blind test (Bodhan, 2026-10-27: Deepika 4/5, Amit 2/5).
VOICE_CHOICES: dict[str, dict] = {
    "sarvam": {"label": "Sarvam bulbul", "hint": "Gree = Simran, Error606 = Dev", "settings": {
        "tts_api_style": "sarvam", "tts_base_url": "https://api.sarvam.ai", "tts_model": "bulbul:v4-flash",
        "tts_voice_host": "dev_en_conversational", "tts_voice_expert": "simran_en_conversation",
        "tts_requests_per_minute": 0, "tts_auth_prefix": ""}},
    "bodhan": {"label": "Bodhan indic-speak", "hint": "Gree = Deepika, Error606 = Amit", "settings": {
        "tts_api_style": "bodhan", "tts_base_url": "https://api.bodhan.ai", "tts_model": "indic-speak",
        "tts_voice_host": "Amit", "tts_voice_expert": "Deepika",
        "tts_requests_per_minute": 8, "tts_auth_prefix": ""}},
}


def default_llm(s: Settings) -> str:
    return "openrouter" if s.llm_provider == "openrouter" else "gemini"


def default_voices(s: Settings) -> str:
    return "bodhan" if s.tts_api_style == "bodhan" else "sarvam"


def _llm_key(s: Settings, choice: str) -> str:
    """The key this choice would use. LLM_API_KEY may hold either kind of key (it is also where an
    OpenRouter key went before OPENROUTER_API_KEY existed), so tell them apart by their prefixes:
    a Google key (AIza...) is no use to OpenRouter, and an OpenRouter key (sk-or-...) none to Google."""
    if choice == "openrouter":
        if is_set(s.openrouter_api_key):
            return s.openrouter_api_key
        return "" if s.llm_api_key.startswith("AIza") else s.llm_api_key
    return "" if s.llm_api_key.startswith("sk-or-") else s.llm_api_key


def _tts_key(s: Settings, choice: str) -> str:
    if choice == default_voices(s):
        return s.tts_api_key
    # Bodhan voices: their own key (BODHAN_TTS_API_KEY); only if that is empty, the ASR key is tried
    # (one Bodhan key can cover both, as on the course server)
    return s.sarvam_api_key if choice == "sarvam" else (s.bodhan_tts_api_key if is_set(s.bodhan_tts_api_key) else s.asr_api_key)


def _ready(s: Settings, kind: str, choice: str) -> bool:
    return s.mock_mode or is_set(_llm_key(s, choice) if kind == "llm" else _tts_key(s, choice))


def effective_llm(s: Settings) -> str:
    """The .env default, or the other script AI if only that one has a key (e.g. a copy run with a
    Gemini key but no OpenRouter key)."""
    d = default_llm(s)
    return d if _ready(s, "llm", d) else next((k for k in LLM_CHOICES if _ready(s, "llm", k)), d)


def effective_voices(s: Settings) -> str:
    """The .env default, or the other voices if only those have a key (e.g. a copy run with a Bodhan
    key but no Sarvam key gets Bodhan voices instead of placeholder tones)."""
    d = default_voices(s)
    return d if _ready(s, "voices", d) else next((k for k in VOICE_CHOICES if _ready(s, "voices", k)), d)


def settings_for(llm: str = "", voices: str = "", base: Settings | None = None) -> Settings:
    """The server settings with an episode's choices applied ("" = the effective default)."""
    s = base or _base_settings()
    llm, voices = llm or effective_llm(s), voices or effective_voices(s)
    update: dict = {}
    if llm in LLM_CHOICES and llm != default_llm(s):
        update |= LLM_CHOICES[llm]["settings"]
        if llm == "gemini":
            update["llm_api_key"] = _llm_key(s, "gemini")
        else:
            update["openrouter_api_key"] = _llm_key(s, "openrouter")
    if voices in VOICE_CHOICES and voices != default_voices(s):
        update |= VOICE_CHOICES[voices]["settings"] | {"tts_api_key": _tts_key(s, voices)}
    return s.model_copy(update=update) if update else s


def options(s: Settings | None = None) -> dict:
    """What the Settings panel offers: each choice, whether its key is set, and the default."""
    s = s or _base_settings()
    llm, voices = effective_llm(s), effective_voices(s)
    return {
        "llm": [{"id": k, "label": v["label"], "hint": v["hint"], "ready": _ready(s, "llm", k), "default": k == llm}
                for k, v in LLM_CHOICES.items()],
        "voices": [{"id": k, "label": v["label"], "hint": v["hint"], "ready": _ready(s, "voices", k), "default": k == voices}
                   for k, v in VOICE_CHOICES.items()],
    }


def check_choice(llm: str, voices: str) -> str | None:
    """A plain-language reason the choice can't be used, or None."""
    opts = options()
    for kind, choice, name in (("llm", llm, "script AI"), ("voices", voices, "voices")):
        if not choice:
            continue
        match = next((o for o in opts[kind] if o["id"] == choice), None)
        if match is None:
            return f"Unknown {name} option."
        if not match["ready"]:
            return f"{match['label']} isn't set up on this server yet (no key). Pick another in Settings."
    return None


@contextlib.contextmanager
def using(llm: str = "", voices: str = "") -> Iterator[Settings]:
    """Run the enclosed code (and the tasks it starts) with an episode's providers."""
    token = _override.set(settings_for(llm, voices))
    try:
        yield _override.get()
    finally:
        _override.reset(token)
