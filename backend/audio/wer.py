"""Word error rate for the ASR self-check.

Both sides are normalised the same way (lower-case, punctuation stripped, numbers
spelled out) so that "5" in ASR output vs "five" in the TTS text is not an error.
"""
from __future__ import annotations

import re

from ..textnorm import number_to_words

_WORD = re.compile(r"[a-z0-9']+")


def normalize_for_wer(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
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
