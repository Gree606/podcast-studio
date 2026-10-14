"""Text normalisation before TTS.

TTS engines read "H2O", "5 m/s" or "x^2" badly, which is one of the quickest ways
to sound like a machine. We rewrite symbols, units, chemical formulas and numbers
into the words a human host would say. The original text is kept for the
transcript; only ``tts_text`` goes to the synthesiser.
"""
from __future__ import annotations

import re

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _int_words(n: int) -> str:
    if n < 0:
        return "minus " + _int_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + _int_words(rest))
    for div, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= div:
            rest = n % div
            head = _int_words(n // div) + " " + name
            if rest == 0:
                return head
            return head + (" and " if rest < 100 else " ") + _int_words(rest)
    return str(n)


def number_to_words(token: str) -> str:
    """'42' -> 'forty two', '3.14' -> 'three point one four', '1,500' -> 'one thousand five hundred'."""
    token = token.replace(",", "")
    if "." in token:
        whole, frac = token.split(".", 1)
        w = _int_words(int(whole)) if whole else "zero"
        return w + " point " + " ".join(_ONES[int(d)] for d in frac if d.isdigit())
    try:
        n = int(token)
    except ValueError:
        return token
    # Years read as "nineteen ninety eight" rather than "one thousand nine hundred..."
    if 1100 <= n <= 2099 and n % 100 != 0 and len(token) == 4:
        return _int_words(n // 100) + " " + (("oh " + _ONES[n % 100]) if n % 100 < 10 else _int_words(n % 100))
    return _int_words(n)


_UNITS = {
    "m/s²": "metres per second squared", "m/s2": "metres per second squared", "m/s^2": "metres per second squared",
    "m/s": "metres per second", "km/h": "kilometres per hour", "km/hr": "kilometres per hour",
    "kmph": "kilometres per hour", "g/cm³": "grams per cubic centimetre", "kg/m³": "kilograms per cubic metre",
    "°C": "degrees Celsius", "°F": "degrees Fahrenheit", "K": "kelvin",
    "km": "kilometres", "cm": "centimetres", "mm": "millimetres", "nm": "nanometres", "m": "metres",
    "kg": "kilograms", "mg": "milligrams", "g": "grams", "mL": "millilitres", "ml": "millilitres", "L": "litres",
    "s": "seconds", "ms": "milliseconds", "h": "hours", "min": "minutes",
    "N": "newtons", "J": "joules", "kJ": "kilojoules", "W": "watts", "kW": "kilowatts", "kWh": "kilowatt hours",
    "V": "volts", "A": "amperes", "Ω": "ohms", "Hz": "hertz", "Pa": "pascals", "mol": "moles", "%": "percent",
}
_UNIT_RE = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(" + "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True)) + r")(?![\w/])"
)

_ELEMENTS = {
    "H": "H", "O": "O", "C": "C", "N": "N", "Na": "N A", "Cl": "C L", "Ca": "C A", "Mg": "M G", "Fe": "F E",
    "Cu": "C U", "Zn": "Z N", "K": "K", "S": "S", "P": "P", "Al": "A L", "Ag": "A G", "Pb": "P B", "Ba": "B A",
}
_FORMULA_RE = re.compile(r"\b((?:(?:Na|Cl|Ca|Mg|Fe|Cu|Zn|Al|Ag|Pb|Ba|[HOCNKSP])\d*){2,})\b")
_ELEM_TOKEN = re.compile(r"(Na|Cl|Ca|Mg|Fe|Cu|Zn|Al|Ag|Pb|Ba|[HOCNKSP])(\d*)")


def _formula_words(f: str) -> str:
    parts = []
    for el, num in _ELEM_TOKEN.findall(f):
        parts.append(_ELEMENTS[el])
        if num:
            parts.append(_int_words(int(num)))
    return " ".join(parts)


def _is_formula(token: str) -> bool:
    # Require a digit or ≥2 element symbols with at least one being multi-letter/compound-looking,
    # so ordinary capitalised words ("SO", "NO") in prose aren't spelled out.
    return bool(re.search(r"\d", token)) or token in {"NaCl", "HCl", "KCl", "CuO", "ZnO", "MgO", "CaO", "FeS"}


_SYMBOLS = [
    (r"\s*=\s*", " equals "), (r"\s*≈\s*", " is approximately "), (r"\s*×\s*", " times "),
    (r"\s*÷\s*", " divided by "), (r"\s*→\s*", " gives "), (r"\s*⟶\s*", " gives "), (r"\s*≤\s*", " is less than or equal to "),
    (r"\s*≥\s*", " is greater than or equal to "), (r"\s*<\s*", " is less than "), (r"\s*>\s*", " is greater than "),
    (r"(?<=\w)\^2\b", " squared"), (r"(?<=\w)²", " squared"), (r"(?<=\w)\^3\b", " cubed"), (r"(?<=\w)³", " cubed"),
    (r"\s*\+\s*", " plus "), (r"\s*&\s*", " and "), (r"\s*@\s*", " at "), (r"\bi\.e\.", "that is"),
    (r"\be\.g\.", "for example"), (r"\betc\.", "et cetera"), (r"\bvs\.?\b", "versus"), (r"\bFig\.", "figure"),
    (r"\bNCERT\b", "N C E R T"), (r"\bDNA\b", "D N A"), (r"\bRNA\b", "R N A"), (r"\bATP\b", "A T P"),
    (r"π", "pi"), (r"Δ", "delta"), (r"λ", "lambda"), (r"μ", "micro"), (r"°", " degrees"),
]


def normalize_for_tts(text: str) -> str:
    t = text
    t = re.sub(r"[*_#`]+", "", t)  # stray markdown
    t = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", t)
    t = re.sub(r"\(\s*(?:see\s+)?(?:fig(?:ure)?|table|activity)\.?\s*[\d.]+\s*\)", "", t, flags=re.I)

    def unit_sub(m: re.Match) -> str:
        num, unit = m.group(1), m.group(2)
        words = number_to_words(num)
        name = _UNITS[unit]
        if words == "one" and unit != "%":
            head, _, tail = name.partition(" ")
            if head.endswith("s") and head not in ("hertz", "celsius"):
                head = head[:-1]
            name = (head + " " + tail).strip()
        return f"{words} {name}"

    t = _UNIT_RE.sub(unit_sub, t)
    t = _FORMULA_RE.sub(lambda m: _formula_words(m.group(1)) if _is_formula(m.group(1)) else m.group(1), t)
    for pat, rep in _SYMBOLS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"(\d)\s*-\s*(\d)", r"\1 to \2", t)  # ranges "5-10"
    t = re.sub(r"\d[\d,]*(?:\.\d+)?", lambda m: number_to_words(m.group(0)), t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
