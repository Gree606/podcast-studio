"""Error rate for the ASR self-check.

English lines: word error rate. Lines in Indian languages: the share of the line's own-script
letters that the recogniser did not hear (``indic_miss_rate``). Two things make plain WER
useless there, both seen on real episodes: scripts mix in English words in English letters
("autotrophs") while the recogniser writes everything in the Indian script ("ऑटोट्रॉफ्स"), and
spoken spellings differ from written ones (ग्री / घ्री). The miss rate ignores both but still
catches a cut-off take (75% missed), silence (100%) or the wrong sentence (86%).

Both sides are normalised the same way (lower-case, punctuation stripped, numbers
spelled out, contractions expanded) so that "5" vs "five" or "I'm" vs "I am" is not
an error. Without the contraction step, real ASR (which writes "I am") pushed ordinary
conversational lines over the re-record threshold.
"""
from __future__ import annotations

import re
import unicodedata

from ..textnorm import number_to_words

_WORD = re.compile(r"[a-z0-9']+")


def _non_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(1 for c in letters if ord(c) > 0x24F) / len(letters) > 0.5


def _strip_punct(text: str) -> str:
    """Punctuation and symbols (any script, incl. the Devanagari danda) become spaces; letters and
    their combining vowel signs stay together, so Indic words are not cut apart."""
    return "".join(" " if unicodedata.category(ch)[0] in "PSZ" and ch != "'" else ch for ch in text)
_SPECIAL = {"won't": "will not", "can't": "can not", "cannot": "can not", "shan't": "shall not", "let's": "let us", "ain't": "is not"}
# "'s" is only expanded after words where it can't be a possessive ("Riya's book" stays as is).
_IS_WORDS = "it|that|what|there|here|he|she|who|where|how|when|why|this"
_CONTRACTIONS = [
    (re.compile(r"\b(" + "|".join(re.escape(k) for k in _SPECIAL) + r")\b"), lambda m: _SPECIAL[m.group(1)]),
    (re.compile(r"n't\b"), lambda m: " not"),
    (re.compile(r"'re\b"), lambda m: " are"),
    (re.compile(r"'ve\b"), lambda m: " have"),
    (re.compile(r"'ll\b"), lambda m: " will"),
    (re.compile(r"'d\b"), lambda m: " would"),
    (re.compile(r"\bi'm\b"), lambda m: "i am"),
    (re.compile(rf"\b({_IS_WORDS})'s\b"), lambda m: m.group(1) + " is"),
]


def normalize_for_wer(text: str) -> list[str]:
    text = text.lower().replace("-", " ").replace("\u2019", "'").replace("\u2018", "'")
    for rx, rep in _CONTRACTIONS:
        text = rx.sub(rep, text)
    text = re.sub(r"\d+(?:\.\d+)?", lambda m: " " + number_to_words(m.group(0)) + " ", text)
    if _non_latin(text):
        return [w for w in _strip_punct(text).split() if w]
    return [w.strip("'") for w in _WORD.findall(text) if w.strip("'")]


def _indic_letters(text: str) -> list[str]:
    return [c for c in text if (c.isalpha() or unicodedata.category(c) in ("Mn", "Mc")) and ord(c) > 0x24F]


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def indic_miss_rate(reference: str, hypothesis: str) -> float | None:
    """Share of the reference's Indian-script letters missing (in order) from what was heard;
    None when the line has too little Indian-script text to judge (e.g. an all-English line)."""
    ref = _indic_letters(reference)
    if len(ref) < 8:
        return None
    return 1 - _lcs(ref, _indic_letters(hypothesis)) / len(ref)


def wer(reference: str, hypothesis: str, language: str | None = None) -> float | None:
    """Error rate of a take (see the module docstring); None = this line can't be judged."""
    if language not in (None, "en") or _non_latin(reference):
        return indic_miss_rate(reference, hypothesis)
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)
