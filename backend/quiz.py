"""End-of-episode quiz, built deterministically from the episode's verified key facts.

Each question is a fill-in-the-blank: one specific term of a verified fact is blanked.
The wrong options are terms taken from *other* facts, and a wrong option is only used
if it cannot be a correct fill: it must not occur anywhere in the passage the fact
cites, and must not look like a form of the answer. Questions that can't get at
least two safe wrong options are dropped; with fewer than three good questions
there is no quiz. No model calls, so nothing here can invent a fact.
"""
from __future__ import annotations

import hashlib
import math
import re

from .schemas import Episode, Source

MIN_QUESTIONS = 2  # a 2-minute episode often has only a few verified facts
MAX_QUESTIONS = 5
MAX_WRONG = 3

# Letters of a word in any supported script. \b can't be used for Indian scripts: Python counts
# their vowel signs as non-word characters, so "चुम्बकों" would have a "boundary" inside it.
_L = r"[\w\u0900-\u0963\u0966-\u0D7F\u200c\u200d]"
_INDIC = r"[\u0900-\u0963\u0966-\u0D7F\u200c\u200d]{3,}"
_TERM = re.compile(rf"(?<!{_L})(?:\d+(?:\.\d+)?|[A-Za-z][A-Za-z-]{{3,}}|{_INDIC})(?!{_L})")


def _word(term: str) -> str:
    """Regex for ``term`` as a whole word, in any script."""
    return rf"(?<!{_L}){re.escape(term)}(?!{_L})"

_STOP = set(["about", "above", "after", "again", "against", "along", "also", "among", "another", "around", "because", "been", "before", "being", "below", "between", "both", "called", "cannot", "could", "does", "doing", "down", "during", "each", "either", "every", "from", "further", "have", "having", "here", "into", "itself", "just", "known", "like", "made", "make", "makes", "many", "more", "most", "much", "must", "near", "never", "often", "only", "other", "otherwise", "over", "same", "should", "since", "some", "such", "than", "that", "their", "them", "then", "there", "these", "they", "thing", "things", "this", "those", "through", "thus", "together", "under", "until", "upon", "very", "what", "when", "where", "which", "while", "whose", "will", "with", "within", "without", "would", "your", "yours", "used", "uses", "using", "example", "examples", "become", "becomes", "form", "forms", "part", "parts", "type", "types", "kind", "kinds", "take", "takes", "way", "ways", "were", "went", "learnt", "learn", "first", "second", "third", "different", "previous", "following", "given", "shown", "study", "studied", "think", "look", "came", "come", "comes", "find", "found", "seen", "show", "shows", "help", "helps", "need", "needs", "important", "various", "several", "certain", "little", "large", "small", "great", "good", "best", "better", "well", "even", "still", "again", "already", "always", "sometimes", "usually", "really", "है", "हैं", "था", "थे", "थी", "और", "में", "की", "के", "का", "को", "से", "पर", "यह", "वह", "ये", "वे", "एक", "भी", "लिए", "तथा", "होता", "होती", "होते", "करते", "करता", "करती", "जाता", "जाती", "जाते", "बहुत", "वाला", "वाले", "वाली", "सभी", "अधिक", "कुछ", "किसी", "सकते", "सकता", "सकती", "गया", "गई", "गए", "रहा", "रही", "रहे", "अपने", "अपनी", "इसके", "उसके", "इसमें", "उनके", "हमारे", "क्या", "कैसे", "क्यों", "जैसे", "ഒരു", "ഈ", "ആ", "എന്ന", "ഉണ്ട്", "ആണ്", "ഇത്", "അത്", "എന്നാൽ", "കൂടാതെ", "എന്നിവ", "ചെയ്യുന്നു", "എന്നും", "ഇവ", "അവ", "നമ്മുടെ", "നമുക്ക്", "എങ്ങനെ", "എന്താണ്", "ஒரு", "இந்த", "அந்த", "என்ற", "உள்ளது", "ஆகும்", "மற்றும்", "இது", "அது", "எனவே", "நமது", "எப்படி", "என்ன"])


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def _kind(term: str) -> str:
    return "num" if term[0].isdigit() else "word"


def _terms(fact: str) -> list[str]:
    """Candidate blanks: numbers and content words (not the opening word, not ALL-CAPS headings,
    not adverbs/participles, which make vague blanks)."""
    out, seen = [], set()
    for k, m in enumerate(_TERM.finditer(fact)):
        t = m.group(0).strip("-")
        low = t.lower()
        if len(t) < 2 or low in _STOP or low in seen:
            continue
        if _kind(t) == "word" and (k == 0 or (t.isupper() and len(t) > 4) or low.endswith(("ly", "ing", "ed"))):
            continue
        seen.add(low)
        out.append(t)
    return out


def _similar(a: str, b: str) -> bool:
    """True if b could be read as a form of a (plural, prefix, shared stem)."""
    a, b = a.lower(), b.lower()
    return a == b or a in b or b in a or (min(len(a), len(b)) >= 5 and a[:5] == b[:5])


def _blank(fact: str, term: str) -> str:
    return re.sub(_word(term), "_____", fact, flags=re.IGNORECASE)


def _chapter_terms(src: Source, topicality, limit: int = 40) -> list[str]:
    """The chapter's own key terms, most topical first (distractor candidates)."""
    seen: dict[str, str] = {}
    for c in src.chunks:
        for t in _terms(c.text):
            seen.setdefault(t.lower(), t)
    scored = [(topicality(t), t) for t in seen.values()]
    return [t for sc, t in sorted((x for x in scored if x[0] is not None), key=lambda x: (-x[0], x[1].lower()))][:limit]


def _nearby_numbers(answer: str) -> list[str]:
    """Plausible wrong numbers for a numeric blank (half, double, a little more or less)."""
    try:
        v = float(answer)
    except ValueError:
        return []
    out = [v * 2, v / 2, v + 10, v - 1, v + 1, v * 10] if v else [1, 2, 10]
    return [str(int(x)) if float(x).is_integer() else f"{x:g}" for x in out if x > 0]


def build_quiz(ep: Episode, src: Source | None) -> list[dict]:
    if src is None:
        return []
    by_id = {c.id: c for c in src.chunks}
    n_chunks = max(1, len(src.chunks))
    lowered = [c.text.lower() for c in src.chunks]
    # a key term should recur across passages; short chapters (a 6-page storybook) have too few passages for that
    min_df = 2 if n_chunks >= 8 else 1

    def topicality(term: str) -> float | None:
        """Words the chapter keeps coming back to (its key terms) score high; words seen in only
        one passage or in nearly all of them are not worth a question. Numbers always qualify."""
        if _kind(term) == "num":
            return 2.0
        low = term.lower()
        df = sum(1 for t in lowered if low in t)
        if df < min_df or df > max(2, 0.6 * n_chunks):
            return None
        tf = sum(len(re.findall(_word(low), t)) for t in lowered)
        return math.log1p(tf) + (0.5 if term[0].isupper() else 0)

    # 1. verified facts, one entry per distinct fact text
    facts, seen = [], set()
    for seg in ep.segments:
        for kf in seg.key_facts:
            key = kf.fact.strip().lower()
            if kf.verified is True and kf.chunk_id in by_id and key and key not in seen:
                seen.add(key)
                facts.append((seg, kf))

    # 2. the term to blank in each fact: the most chapter-specific one that the passage contains
    picked, used_answers = [], set()
    for seg, kf in facts:
        if kf.fact.rstrip().endswith("?"):
            continue
        passage = by_id[kf.chunk_id].text.lower()
        scored = [(topicality(t), t) for t in _terms(kf.fact) if t.lower() in passage]
        terms = [t for sc, t in sorted((x for x in scored if x[0] is not None), key=lambda x: (-x[0], x[1].lower()))]
        if not terms:
            continue
        for answer in terms:
            question = _blank(kf.fact, answer)
            if answer.lower() not in used_answers and len(question.split()) >= 5 and question.count("_____") == 1:
                used_answers.add(answer.lower())
                picked.append((seg, kf, answer, question, terms[:4]))
                break

    # 3. wrong options from other facts' terms, then the chapter's key terms (and, for numbers, nearby
    # numbers), kept only if they can't fill the blank: never a word of the cited passage
    chapter_terms = _chapter_terms(src, topicality)
    questions = []
    for i, (seg, kf, answer, question, _) in enumerate(picked):
        passage = by_id[kf.chunk_id].text.lower()
        pool = []
        for j, (_, _, _, _, terms) in enumerate(picked):
            if j != i:
                pool += [t for t in terms if _kind(t) == _kind(answer)]
        pool += [t for t in chapter_terms if _kind(t) == _kind(answer)]
        pool = [t for t in pool if t.isascii() == answer.isascii()]  # same script as the answer (no English option in a Hindi quiz)
        if _kind(answer) == "num":
            pool += _nearby_numbers(answer)
        wrong = []
        for t in sorted(dict.fromkeys(pool), key=lambda t: _h(kf.fact + t)):
            if t.lower() in passage or t.lower() in question.lower() or _similar(answer, t):
                continue
            if any(_similar(w, t) for w in wrong):
                continue
            wrong.append(t)
            if len(wrong) == MAX_WRONG:
                break
        if len(wrong) < 2:
            continue
        options = sorted([answer, *wrong], key=lambda t: _h(kf.fact + "|" + t))
        chunk = by_id[kf.chunk_id]
        questions.append({"id": f"q{len(questions) + 1}", "segment_id": seg.id, "segment_title": seg.title,
                          "question": question, "options": options, "answer": answer, "fact": kf.fact,
                          "chunk_id": kf.chunk_id, "section": chunk.section, "page": chunk.page})

    # 4. spread questions over segments: at most one per segment until each has had a turn
    order, used = [], set()
    for q in questions:
        if q["segment_id"] not in used:
            used.add(q["segment_id"])
            order.append(q)
    order += [q for q in questions if q not in order]
    order = order[:MAX_QUESTIONS]
    for k, q in enumerate(order, 1):
        q["id"] = f"q{k}"
    return order if len(order) >= MIN_QUESTIONS else []
