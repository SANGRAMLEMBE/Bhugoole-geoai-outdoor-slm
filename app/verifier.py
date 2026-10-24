"""Grounding verifier: every number, time and place name in the SLM's answer must exist in the fact sheet."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ALLOWED_CAPS = {
    "i", "you", "your", "the", "a", "an", "it", "if", "at", "on", "in", "for", "and", "but", "or", "so", "to",
    "aqi", "uv", "pm2.5", "pm10", "no2", "o3", "am", "pm", "ist", "km", "osm", "india", "indian",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "good", "satisfactory", "moderate", "poor", "very", "severe", "n", "s", "e", "w", "ne", "nw", "se", "sw",
    "north", "south", "east", "west", "celsius", "c", "spf", "n95", "osrm", "open-meteo", "cpcb",
    # label words small models like to add ("Practical Tip: ...")
    "practical", "tip", "tips", "note", "recommendation", "remember", "reminder", "summary", "advice",
    "caution", "warning", "important", "bonus", "pro",
}

# Fact-sheet field labels: an answer that contains them is a copy of the sheet, not an answer.
_SHEET_LABEL = re.compile(r"(?im)^[\s*#>-]*(?:best option|conditions|reason|caveat|hard limits|alternatives|avoid|"
                          r"prepare|watch for|overall|note|requested time verdict|search area|time asked)\s*\**\s*:")
_CATS = ["very poor", "satisfactory", "moderate", "severe", "good", "poor"]  # longest match first
_FACT_CAT = re.compile(r"AQI \d+ \(?(Good|Satisfactory|Moderate|Poor|Very Poor|Severe)\b", re.I)
_ANSWER_CAT = re.compile(r"\b(?:aqi|air quality|air)\b[^.;,]{0,25}?\b(" + "|".join(_CATS) + r")\b", re.I)
# When the sheet says "not advised", the answer has to say so too, and must not call it advised or safe
_SHEET_NOT_ADVISED = re.compile(r"(?m)^(?:OVERALL|REQUESTED TIME VERDICT): NOT ADVISED")
_SAYS_NOT_ADVISED = re.compile(r"\b(?:not|isn't|is not|aren't) (?:advised|advisable|recommended|safe|a good)\b|"
                               r"\b(?:unsafe|should not|shouldn't|better to skip)\b", re.I)
_SAYS_ADVISED = re.compile(r"\b(?:is|are) (?:advised|advisable|recommended|safe|ideal|perfect)\b|"
                           r"\b(?:suitable|good|great|perfect|ideal) (?:choice|time|option)\b", re.I)

_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?\b", re.I)
_TIME24 = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.]*:)")


@dataclass
class Report:
    ok: bool
    checked_numbers: int = 0
    checked_names: int = 0
    unsupported: list[str] = field(default_factory=list)

    def as_dict(self):
        return {"verified": self.ok, "numbers_checked": self.checked_numbers, "names_checked": self.checked_names,
                "unsupported": self.unsupported}


def _times(text: str) -> set[tuple[int, int]]:
    out = set()
    for m in _TIME.finditer(text):
        h = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
        out.add((h, int(m.group(2) or 0)))
    for m in _TIME24.finditer(text):
        out.add((int(m.group(1)), int(m.group(2))))
    return out


def _numbers(text: str) -> list[float]:
    stripped = _TIME.sub(" ", _TIME24.sub(" ", text))
    return [float(x) for x in _NUM.findall(stripped)]


def _norm(text: str) -> str:
    """Lower case, drop possessive 's and punctuation, so "Katraj - Dehu Road" matches "Katraj Dehu Road"."""
    return " " + " ".join(re.sub(r"[^a-z0-9]+", " ", re.sub(r"'s\b", "", text.lower())).split()) + " "


def verify(answer: str, fact_sheet: str) -> Report:
    rep = Report(ok=True)
    fact_l = _norm(fact_sheet)
    fact_nums = set(_numbers(fact_sheet))
    fact_times = _times(fact_sheet)
    for h, mi in fact_times:  # "6-7 AM" style ranges leave a bare hour behind
        fact_nums.update({float(h), float(h % 12 or 12)})
        if mi:
            fact_nums.add(float(mi))
    # Times
    for t in _times(answer):
        rep.checked_numbers += 1
        if t not in fact_times and not any(abs((t[0] * 60 + t[1]) - (f[0] * 60 + f[1])) <= 0 for f in fact_times):
            rep.unsupported.append(f"time {t[0]:02d}:{t[1]:02d}")
    # Numbers (allow rounding by 1 and trivial small counts)
    for n in _numbers(answer):
        rep.checked_numbers += 1
        if n <= 3 and float(n).is_integer():
            continue
        if not any(abs(n - f) <= 1.0 for f in fact_nums):
            rep.unsupported.append(f"number {n:g}")
    # Proper names: capitalised multi-word phrases, or capitalised words not at a sentence start.
    # Commas and brackets end a phrase, so "Powai, Mumbai" is two names, not one.
    for sent in re.split(r"(?<=[.!?:;])\s+|\n+", answer):
        for ci, chunk in enumerate(re.split(r"[,()\[\]\"]", sent)):
            tokens = re.findall(r"[A-Za-z][\w'.\-]*", chunk)
            i = 0
            while i < len(tokens):
                tok = tokens[i]
                if tok[0].isupper():
                    j = i
                    while j + 1 < len(tokens) and tokens[j + 1][0].isupper():
                        j += 1
                    phrase = " ".join(tokens[i:j + 1]).strip(".")
                    words = [w.lower().strip(".'") for w in phrase.split()]
                    sentence_start = ci == 0 and i == 0
                    if not all(w in ALLOWED_CAPS for w in words) and not (sentence_start and j == i):
                        rep.checked_names += 1
                        core = " ".join(w for w in phrase.split() if w.lower().strip(".'") not in ALLOWED_CAPS)
                        # a sentence's first word is capitalised anyway ("Near Powai", "Visit Hill Park")
                        rest = " ".join(core.split()[1:]) if sentence_start else ""
                        if core and _norm(core) not in fact_l and not (rest and _norm(rest) in fact_l):
                            rep.unsupported.append(f"name '{core}'")
                    i = j + 1
                else:
                    i += 1
    # The overall verdict must match the sheet
    if _SHEET_NOT_ADVISED.search(fact_sheet) and (not _SAYS_NOT_ADVISED.search(answer)
                                                   or _SAYS_ADVISED.search(answer)):
        rep.unsupported.append("verdict contradicts the fact sheet (it says not advised)")
    # Copying the fact sheet's layout instead of writing an answer
    if _SHEET_LABEL.search(answer):
        rep.unsupported.append("copied fact-sheet labels")
    # Air-quality category words must match a category the fact sheet reports
    fact_cats = {m.group(1).lower() for m in _FACT_CAT.finditer(fact_sheet)}
    if fact_cats:
        for m in _ANSWER_CAT.finditer(answer):
            rep.checked_names += 1
            if m.group(1).lower() not in fact_cats:
                rep.unsupported.append(f"air quality '{m.group(1).lower()}'")
    rep.unsupported = sorted(set(rep.unsupported))
    rep.ok = not rep.unsupported
    return rep
