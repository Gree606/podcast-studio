"""Word error rate for the ASR self-check.

Both sides are normalised the same way (lower-case, punctuation stripped, numbers
spelled out, contractions expanded) so that "5" vs "five" or "I'm" vs "I am" is not
an error. Without the contraction step, real ASR (which writes "I am") pushed ordinary
conversational lines over the re-record threshold.
"""
from __future__ import annotations

import re

from ..textnorm import number_to_words

_WORD = re.compile(r"[a-z0-9']+")
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
    return [w.strip("'") for w in _WORD.findall(text) if w.strip("'")]


def wer(reference: str, hypothesis: str) -> float:
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
