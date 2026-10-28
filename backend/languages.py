"""Languages an episode can be made in.

One table drives every stage: the writer's instructions, the TTS language code (Sarvam
bulbul and Bodhan indic-speak both speak these), the ASR language for the listen-back
check (Bodhan indic-transcribe), and the HTML ``lang`` used to render captions. The
chapter itself can be in any language; the fact-check always compares the script with
the chapter as written, so an English chapter can become a Tamil episode.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str  # our id, stored in EpisodeSettings.language
    name: str  # English name (prompts, logs)
    native: str  # shown in the picker
    tts: str  # BCP-47 code for the voices
    asr: str  # language hint for the speech recogniser
    html: str  # lang attribute for captions
    script: str  # how the writer should spell it


LANGUAGES: dict[str, Language] = {x.code: x for x in [
    Language("en", "English", "English", "en-IN", "en", "en", "English"),
    Language("hi", "Hindi", "हिन्दी", "hi-IN", "hi", "hi", "Devanagari"),
    Language("hinglish", "Hinglish", "Hinglish", "hi-IN", "hi", "hi", "Devanagari with English words in English"),
    Language("bn", "Bengali", "বাংলা", "bn-IN", "bn", "bn", "Bengali"),
    Language("ta", "Tamil", "தமிழ்", "ta-IN", "ta", "ta", "Tamil"),
    Language("te", "Telugu", "తెలుగు", "te-IN", "te", "te", "Telugu"),
    Language("kn", "Kannada", "ಕನ್ನಡ", "kn-IN", "kn", "kn", "Kannada"),
    Language("ml", "Malayalam", "മലയാളം", "ml-IN", "ml", "ml", "Malayalam"),
    Language("mr", "Marathi", "मराठी", "mr-IN", "mr", "mr", "Devanagari"),
    Language("gu", "Gujarati", "ગુજરાતી", "gu-IN", "gu", "gu", "Gujarati"),
    Language("pa", "Punjabi", "ਪੰਜਾਬੀ", "pa-IN", "pa", "pa", "Gurmukhi"),
    Language("od", "Odia", "ଓଡ଼ିଆ", "od-IN", "or", "or", "Odia"),
]}
DEFAULT = "en"


def get(code: str | None) -> Language:
    return LANGUAGES.get(code or DEFAULT, LANGUAGES[DEFAULT])


def writing_rule(code: str | None) -> str:
    """The instruction the writer (and the rewrite and fact-fix steps) gets for this language."""
    lang = get(code)
    if lang.code == "en":
        return "Language: write the whole dialogue in English."
    if lang.code == "hinglish":
        return ("Language: write in Hinglish, the way Indian students really talk: Hindi sentences in Devanagari script, "
                "with English technical terms and everyday English words written in English letters "
                "(e.g. \"तो photosynthesis में plant अपना खाना खुद बनाता है\").")
    return (f"Language: write the whole dialogue in natural, spoken {lang.name} ({lang.script} script), the way a friendly "
            f"teacher talks in a {lang.name}-medium classroom; not a word-for-word translation. Use the standard {lang.name} "
            "term for each science idea; if the chapter is in English you may say the English term once right after it. "
            "Write numbers as digits. Keep the two hosts' names exactly as they are.")
