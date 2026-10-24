"""Understand a free-form question: activity, place, radius, time and personal profile.

Rules run first (fast, predictable). The SLM is consulted only for slots the rules
could not fill, and anything it returns is validated against the question text so
it cannot introduce a place the user never mentioned.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .activities import ACTIVITIES
from .profiles import PROFILES
from .timeparse import TimeWindow, parse_time

STOP = {
    "in", "on", "at", "for", "this", "next", "tomorrow", "today", "tonight", "by", "during", "with", "and", "or",
    "but", "before", "after", "around", "between", "from", "to", "within", "morning", "evening", "afternoon",
    "night", "weekend", "any", "what", "which", "where", "when", "is", "are", "should", "can", "could", "please",
    "that", "the", "if", "while", "so", "then", "over", "under", "about", "near", "nearby", "coming", "early",
    "late", "sunset", "sunrise", "i", "we", "my", "our", "for", "starting", "till", "until", "who", "how", "good",
    "best", "spots", "spot", "place", "places", "area", "time", "times", "maybe", "also", "it", "lol", "pls",
    "plz", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "weekday", "sat", "sun",
    "km", "kms", "hours", "hour", "days", "day", "week", "via", "outdoors", "outside", "safe", "okay", "ok",
}
NOT_A_PLACE = {
    "the", "a", "an", "my", "our", "your", "this", "that", "next", "work", "office", "home", "college", "school",
    "morning", "evening", "afternoon", "night", "weekend", "least", "most", "all", "advance", "general", "time",
    "the morning", "the evening", "the afternoon", "the night", "the next", "the weekend", "the park", "a park",
    "the city", "town", "city", "outdoors", "outside", "here", "there", "me", "us", "it", "total", "case",
    "mind", "the sun", "sun", "rain", "the rain", "shade", "the shade", "peace", "about", "once", "pm", "am",
}
VERBS = {"go", "going", "play", "playing", "walk", "run", "jog", "do", "be", "take", "get", "have", "see", "visit",
         "head", "step", "fly", "ride", "click", "shoot", "watch", "spend", "plan", "organise", "organize", "host",
         "least", "advance", "full", "peace", "front", "case", "time", "work", "office", "home"}
SELF_PLACE = re.compile(r"\b(?:near|around|from|at|close to)\s+(?:me|my (?:home|house|place|office|area|location)|"
                        r"home|here|my current location)\b|\bnearby\b(?!\s+\w+\s+(?:of|to))|\bnear by\b|"
                        r"\b(?:to|for) (?:work|office|college|school)\b", re.I)
COMMUTE = re.compile(r"\b(?:to|from) (?:work|office|college|school|uni)\b|\bcommut\w*", re.I)
GENERIC_OUTDOOR = re.compile(r"\b(?:go(?:ing)? out(?:side|doors)?|step out|head(?:ing)? out|be outdoors|"
                             r"outdoors?|outside|fresh air)\b", re.I)
OUT_OF_SCOPE = re.compile(
    r"\b(?:movie|cinema|restaurant|hotel booking|flight|train ticket|recipe|stock|crypto|bitcoin|"
    r"homework|code|python|javascript|mall|shopping|gym|treadmill|indoor|netflix|song|lyrics|"
    r"capital of|president|prime minister|translate|essay|poem|joke)\b", re.I)


@dataclass
class ParsedQuery:
    question: str
    activity: str | None = None
    activity_text: str | None = None
    location: str | None = None
    location_is_self: bool = False
    radius_km: float | None = None
    profiles: list[str] = field(default_factory=list)
    commute: bool = False
    generic_outdoor: bool = False
    out_of_scope: bool = False
    time: TimeWindow | None = None
    parsed_by: list[str] = field(default_factory=lambda: ["rules"])
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("time", None)
        if self.time:
            d["time"] = {"label": self.time.label, "hours": len(self.time.hours()),
                         "specific_hour": self.time.specific_hour.isoformat() if self.time.specific_hour else None,
                         "target": self.time.target, "compare_days": self.time.compare_days,
                         "assumed": self.time.assumed, "notes": self.time.notes}
        return d


def detect_activity(text: str) -> tuple[str | None, str | None]:
    best = None  # (length, -order, id, matched)
    for order, (aid, act) in enumerate(ACTIVITIES.items()):
        for kw in act.keywords:
            for m in re.finditer(rf"\b{kw}\b", text, re.I):
                cand = (len(m.group(0)), -order, aid, m.group(0))
                if best is None or cand[:2] > best[:2]:
                    best = cand
    # "nature walk" is hiking, "photo walk" is photography: the longest match wins above.
    return (best[2], best[3]) if best else (None, None)


_AGE = re.compile(
    r"\b(?:i am|i'm|i’m|im|he is|he's|she is|she's|they are|they're|aged?|age of)\s+(\d{1,3})\b"
    r"(?!\s*(?:km|kms|k\b|min|mins|minutes|hours?|am|pm|a\.m|p\.m|:|%|people|of us))"
    r"|\b(\d{1,3})\s*(?:years?|yrs?|y/?o)\b", re.I)


def detect_ages(text: str) -> list[int]:
    """Ages written in the question: "I am 67", "she's 72", "age 70", "my 8 y/o"."""
    return [int(a or b) for a, b in _AGE.findall(text) if 0 < int(a or b) < 110]


def detect_profiles(text: str) -> list[str]:
    found = []
    for pid, p in PROFILES.items():
        if any(re.search(rf"\b{kw}\b", text, re.I) for kw in p.keywords):
            found.append(pid)
    ages = detect_ages(text)
    if any(a >= 60 for a in ages) and "senior" not in found:
        found.append("senior")
    if any(a <= 12 for a in ages) and "child" not in found:
        found.append("child")
    return found


def detect_radius(text: str) -> float | None:
    m = re.search(r"(?:within|under|less than|upto|up to|max(?:imum)?)\s+(\d+(?:\.\d+)?)\s*(km|kms|kilomet(?:er|re)s?|mi|miles?)\b", text, re.I) \
        or re.search(r"\b(\d+(?:\.\d+)?)\s*(km|kms|kilomet(?:er|re)s?|mi|miles?)\s+(?:radius|away|range|from|of)\b", text, re.I)
    if m:
        v = float(m.group(1))
        return v * 1.609 if m.group(2).lower().startswith("mi") else v
    if re.search(r"walking distance", text, re.I):
        return 2.0
    return None


def _clean_place(raw: str) -> str | None:
    words = re.split(r"\s+", raw.strip())
    kept = []
    for w in words:
        lw = w.lower().strip(",.?!;:'\"()")
        if not lw:
            break
        if lw in STOP and kept:
            break
        if lw in STOP and not kept:
            return None
        if re.match(r"^\d", lw) and not kept:
            return None
        kept.append(w.strip(",.?!;:\"()"))
        if re.search(r"[,.?!;:]$", w):
            break
        if len(kept) >= 5:
            break
    place = " ".join(kept).strip(" '")
    first = place.split()[0].lower() if place else ""
    if first in VERBS or detect_activity(first)[0]:
        return None
    place = re.sub(r"'s$", "", place)
    if not place or place.lower() in NOT_A_PLACE or len(place) < 2:
        return None
    return place


def detect_location(text: str) -> tuple[str | None, bool]:
    if SELF_PLACE.search(text) and not re.search(r"\b(?:near|around|in|at|of)\s+[A-Z]", text):
        return None, True
    patterns = [
        r"\b(?:within|in a radius of|radius of)\s+\d+(?:\.\d+)?\s*(?:km|kms|kilomet\w*|mi|miles?)\s+(?:of|from|around)\s+(?P<loc>[^?!,;]+)",
        r"\b\d+(?:\.\d+)?\s*(?:km|kms|kilomet\w*)\s+(?:of|from|around)\s+(?P<loc>[^?!,;]+)",
        r"\b(?:near|around|close to|next to|nearby|near to|outside|off)\s+(?P<loc>[^?!,;]+)",
        r"\b(?:in|at|from|towards)\s+(?P<loc>[^?!,;]+)",
    ]
    for pat in patterns:
        for m in re.finditer(f"(?={pat})", text, re.I):  # lookahead: overlapping matches
            place = _clean_place(m.group("loc"))
            if place:
                return place, False
    return None, bool(SELF_PLACE.search(text))


def parse_rules(question: str, now: datetime) -> ParsedQuery:
    q = ParsedQuery(question=question)
    q.activity, q.activity_text = detect_activity(question)
    q.profiles = detect_profiles(question)
    q.radius_km = detect_radius(question)
    q.location, q.location_is_self = detect_location(question)
    q.commute = bool(COMMUTE.search(question)) and q.activity in ("cycling", "walking", "running", None)
    if q.commute and q.activity is None and re.search(r"\bcycl|bik", question, re.I):
        q.activity = "cycling"
    if q.activity is None and GENERIC_OUTDOOR.search(question):
        q.generic_outdoor = True
    if q.activity is None and not q.generic_outdoor and OUT_OF_SCOPE.search(question):
        q.out_of_scope = True
    q.time = parse_time(question, now)
    return q


def needs_slm(q: ParsedQuery) -> bool:
    """Ask the SLM only when rules left a gap that wording could fill."""
    if q.out_of_scope:
        return False
    if q.activity is None and not q.generic_outdoor:
        return True
    if q.location is None and not q.location_is_self:
        # Is there an unexplained capitalised word or phrase? e.g. "Powai morning walk?"
        caps = re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*", q.question)
        ignore = {"I", "Is", "Where", "When", "What", "Which", "Can", "Could", "Should", "Best", "Good", "Planning",
                  "Outdoor", "How", "Any", "Please", "Today", "Tomorrow", "Tonight", "My", "We", "Our", "Hey", "Hi",
                  "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Will", "Do",
                  "Does", "Are", "Am", "Pm", "AM", "PM", "Need", "Want", "Looking", "Suggest", "Recommend", "Tell"}
        caps = [c for c in caps if c.split()[0] not in ignore]
        return bool(caps) or len(q.question.split()) >= 4
    return False


def merge_slm(q: ParsedQuery, slm: dict) -> ParsedQuery:
    """Merge SLM-extracted slots, rejecting anything not grounded in the question text."""
    if not slm:
        return q
    used = False
    act = (slm.get("activity") or "").strip().lower()
    if q.activity is None and act in ACTIVITIES:
        q.activity, q.activity_text, used = act, act, True
    if q.activity is None and act == "out_of_scope" and not q.generic_outdoor:
        q.out_of_scope, used = True, True
    loc = (slm.get("location") or "").strip()
    if q.location is None and loc and loc.lower() in q.question.lower() and loc.lower() not in NOT_A_PLACE:
        q.location, q.location_is_self, used = loc, False, True
    elif loc and loc.lower() not in q.question.lower():
        q.notes.append(f"Ignored a model-suggested place ('{loc}') that is not in the question.")
    for p in slm.get("profiles") or []:
        if p in PROFILES and p not in q.profiles and re.search(r"\w", q.question):
            # only accept profiles with some lexical support in the question
            hint = {"senior": r"old|senior|elder|parent|grand|age|retire", "child": r"kid|child|son|daughter|baby|toddler|school",
                    "asthma": r"asthm|breath|lung|wheez|allerg|copd", "pregnancy": r"pregnan|expect|trimester",
                    "heart": r"heart|cardi|bp|pressure"}[p]
            if re.search(hint, q.question, re.I):
                q.profiles.append(p)
                used = True
    if q.radius_km is None and isinstance(slm.get("radius_km"), (int, float)) and 0 < slm["radius_km"] <= 100:
        if re.search(rf"\b{int(slm['radius_km'])}\b", q.question):
            q.radius_km, used = float(slm["radius_km"]), True
    if used:
        q.parsed_by.append("slm")
    return q
