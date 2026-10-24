"""Resolve vague time expressions ("tomorrow evening", "this weekend", "after work") to concrete hours in IST."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from . import config

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WD_SHORT = {"mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

# part of day -> (start hour, end hour) ; end is exclusive, may exceed 24
PARTS = [
    (r"early morning|dawn|first light", "early morning", (5, 8)),
    (r"sunrise", "sunrise", (5, 8)),
    (r"late morning", "late morning", (9, 12)),
    (r"morning|subah|a\.?m\.? hours", "morning", (5, 11)),
    (r"noon|midday|lunch ?time", "midday", (11, 14)),
    (r"afternoon|dopahar", "afternoon", (12, 17)),
    (r"after (?:work|office|college|school)|post[- ]work|after my shift", "after work", (17, 21)),
    (r"sunset|dusk|golden hour|twilight", "sunset", (16, 20)),
    (r"evening|shaam|sham", "evening", (16, 20)),
    (r"tonight|tonite|night|nite|raat", "night", (19, 24)),
]


@dataclass
class TimeWindow:
    intervals: list[tuple[datetime, datetime]]
    label: str
    explicit_day: bool = False
    explicit_part: bool = False
    specific_hour: datetime | None = None
    target: str | None = None          # "sunset" / "sunrise" when the user is chasing the light
    compare_days: bool = False
    assumed: bool = False               # no time information at all in the question
    notes: list[str] = field(default_factory=list)

    def hours(self) -> list[datetime]:
        out, seen = [], set()
        for s, e in self.intervals:
            t = s
            while t < e:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
                t += timedelta(hours=1)
        return sorted(out)

    def days(self) -> list[date]:
        return sorted({t.date() for t in self.hours()})


def _next_weekday(today: date, wd: int, force_next: bool = False) -> date:
    delta = (wd - today.weekday()) % 7
    if force_next and delta == 0:
        delta = 7
    return today + timedelta(days=delta)


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "a couple of": 2,
              "couple of": 2, "few": 3, "a few": 3}


def _to24(h: int, ap: str | None, ctx: str) -> int:
    if ap:
        return h % 12 + (12 if ap[0] == "p" else 0)
    if h > 12:
        return h
    if re.search(r"evening|night|after work|\bpm\b|party|dinner|wedding|reception|sangeet|stargaz", ctx) and h < 12:
        return h + 12
    if not re.search(r"morning|\bam\b", ctx) and 1 <= h <= 6:
        return h + 12  # "at 5" usually means 5 PM for outdoor plans
    return h


def _clock_range(text: str) -> tuple[int, int] | None:
    m = re.search(r"\b(?:between|from)?\s*(\d{1,2})(?::\d\d)?\s*(a\.?m\.?|p\.?m\.?)?\s*(?:and|to|-|–|till|until)\s*"
                  r"(\d{1,2})(?::\d\d)?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])", text)
    if not m:
        return None
    h1 = _to24(int(m.group(3)), m.group(4), text)
    ap0 = m.group(2) or m.group(4)
    h0 = _to24(int(m.group(1)), ap0, text)
    if h0 > h1:  # "11 to 1 pm"
        h0 = _to24(int(m.group(1)), "am", text)
    return (h0, h1) if 0 <= h0 < h1 <= 24 else None


def _clock_times(text: str) -> list[tuple[int, int]]:
    """Explicit clock times as (hour24, minute), in order of appearance."""
    found = []
    for m in re.finditer(r"\b(\d{1,2})(?::|\.)?(\d{2})?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])", text):
        h = int(m.group(1))
        if 1 <= h <= 12:
            found.append((m.start(), _to24(h, m.group(3), text), int(m.group(2) or 0)))
    for m in re.finditer(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s*(?:a\.?m|p\.?m))", text):
        found.append((m.start(), int(m.group(1)), int(m.group(2))))
    for m in re.finditer(r"\b(?:at|by|around|before|after)\s+(\d{1,2})(?:\s*o'?clock)?\b"
                         r"(?!\s*(?:km|kms|min|:|\.\d|a\.?m|p\.?m|\d|%|hours?|hrs?|days?|people|kids|members|guests))", text):
        h = int(m.group(1))
        if 1 <= h <= 12:
            found.append((m.start(), _to24(h, None, text[max(0, m.start() - 40): m.end() + 40]), 0))
    return [(h, mi) for _, h, mi in sorted(found)]


def parse_time(text: str, now: datetime, horizon_days: int | None = None) -> TimeWindow:
    t = " " + text.lower().replace("’", "'") + " "
    horizon_days = horizon_days or config.FORECAST_DAYS
    today = now.date()
    days: list[date] = []
    range_end: datetime | None = None
    notes: list[str] = []
    compare = False

    # ---- days ----
    if re.search(r"day after tomorrow|\bparso\b", t):
        days.append(today + timedelta(days=2))
        t = re.sub(r"day after tomorrow|\bparso\b", " ", t)
    if re.search(r"\btomorrow\b|\btmrw?\b|\btomorow\b|\bkal\b", t):
        days.append(today + timedelta(days=1))
    if re.search(r"\btoday\b|\btonight\b|\btonite\b|\bthis (?:morning|afternoon|evening)\b|\baaj\b|\bnow\b|\bright now\b", t):
        days.append(today)
    m = re.search(r"(?:next|coming|following)\s+(\d+|one|two|three|four|five|six|seven|a couple of|couple of|a few|few)\s+(days?|hours?)", t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM_WORDS[m.group(1)]
        range_end = now + (timedelta(days=n) if m.group(2).startswith("day") else timedelta(hours=n))
    if re.search(r"\bthis week\b", t) and range_end is None:
        range_end = datetime.combine(today + timedelta(days=6 - today.weekday()), datetime.max.time(), now.tzinfo)
        if (range_end - now).days < 1:
            range_end = now + timedelta(days=3)
    if re.search(r"\bweek-?ends?\b", t):
        if today.weekday() == 5:
            wk = [today, today + timedelta(days=1)]
        elif today.weekday() == 6:
            wk = [today]
        else:
            sat = _next_weekday(today, 5)
            wk = [sat, sat + timedelta(days=1)]
        if re.search(r"\bnext week-?end\b", t) and today.weekday() >= 5:
            sat = _next_weekday(today, 5, force_next=True)
            wk = [sat, sat + timedelta(days=1)]
        days += wk
    for i, wd in enumerate(WEEKDAYS):
        for m in re.finditer(rf"\b(next|this|coming)?\s*{wd}s?\b", t):
            days.append(_next_weekday(today, i, force_next=(m.group(1) == "next")))
    for short, i in WD_SHORT.items():
        if re.search(rf"\b(?:on|this|next|coming)\s+{short}\b", t):
            days.append(_next_weekday(today, i))
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", t):
        days.append(_date(today, int(m.group(1)), MONTHS[m.group(2)]))
    for m in re.finditer(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\b", t):
        days.append(_date(today, int(m.group(2)), MONTHS[m.group(1)]))
    for m in re.finditer(r"\bon the (\d{1,2})(?:st|nd|rd|th)\b", t):
        d = int(m.group(1))
        cand = _date(today, d, today.month)
        if cand < today:
            cand = _date(today, d, today.month % 12 + 1)
        days.append(cand)
    days = sorted(set(d for d in days if d))
    if len(days) >= 2 and re.search(r"\bor\b|\bvs\.?\b|\bversus\b|which day|better day", t):
        compare = True

    # ---- parts of day ----
    parts = []
    for pat, name, span in PARTS:
        if re.search(rf"\b(?:{pat})\b", t):
            if name == "evening" and any(p[0] in ("sunset", "after work") for p in parts):
                continue
            if name == "morning" and any(p[0] in ("early morning", "late morning", "sunrise") for p in parts):
                continue
            parts.append((name, span))
    target = next((p for p, _ in parts if p in ("sunset", "sunrise")), None)

    clocks = _clock_times(t)
    crange = _clock_range(t)
    explicit_day = bool(days) or range_end is not None
    explicit_part = bool(parts) or bool(clocks) or bool(crange)

    # ---- build intervals ----
    tz = now.tzinfo
    start_floor = now.replace(minute=0, second=0, microsecond=0)
    if now.minute >= 40:
        start_floor += timedelta(hours=1)

    def day_iv(d: date, h0: int, h1: int):
        s = datetime.combine(d, datetime.min.time(), tz) + timedelta(hours=h0)
        return s, s + timedelta(hours=h1 - h0)

    specific = None
    intervals: list[tuple[datetime, datetime]] = []
    if crange:
        base = (days or [today])[0]
        s0, e0 = day_iv(base, *crange)
        if not days and e0 <= now:
            s0, e0 = s0 + timedelta(days=1), e0 + timedelta(days=1)
            notes.append("That time has already passed today, so I looked at tomorrow.")
        intervals = [(s0, e0)]
    elif clocks:
        h, mi = clocks[0]
        base_days = days or [today]
        d = base_days[0]
        specific = datetime.combine(d, datetime.min.time(), tz) + timedelta(hours=h)
        if not days and specific + timedelta(minutes=30) < now:
            specific += timedelta(days=1)
            notes.append(f"{h:02d}:{mi:02d} has already passed today, so I looked at tomorrow.")
        intervals = [(specific - timedelta(hours=2), specific + timedelta(hours=3))]
    else:
        if range_end is not None:
            d, rng_days = today, []
            while datetime.combine(d, datetime.min.time(), tz) < range_end:
                rng_days.append(d)
                d += timedelta(days=1)
            days = sorted(set(days) | set(rng_days))
        if not days:
            if parts:
                first = [day_iv(today, *sp) for _, sp in parts]
                days = [today] if any(e > now + timedelta(minutes=30) for _, e in first) else [today + timedelta(days=1)]
                if days[0] != today:
                    notes.append("That part of the day has passed today, so I looked at tomorrow.")
        if not days:  # nothing at all: next 48 hours
            intervals = [(start_floor, start_floor + timedelta(hours=48))]
        else:
            for d in days:
                for _, (h0, h1) in (parts or [("day", (0, 24))]):
                    intervals.append(day_iv(d, h0, h1))
        if range_end is not None:
            intervals = [(s, min(e, range_end)) for s, e in intervals if s < range_end]

    # ---- clip to now and forecast horizon ----
    horizon = start_floor + timedelta(days=horizon_days)
    clipped = [(max(s, start_floor), min(e, horizon)) for s, e in intervals]
    clipped = [(s, e) for s, e in clipped if e > s]
    if intervals and not clipped and all(e <= start_floor for _, e in intervals) and \
            all(e > start_floor - timedelta(days=1) for _, e in intervals):
        intervals = [(s + timedelta(days=1), e + timedelta(days=1)) for s, e in intervals]
        if specific is not None:
            specific += timedelta(days=1)
        notes.append("That time has already passed today, so I looked at the same time tomorrow.")
        clipped = [(max(s, start_floor), min(e, horizon)) for s, e in intervals]
        clipped = [(s, e) for s, e in clipped if e > s]
    if intervals and not clipped:
        if all(s >= horizon for s, _ in intervals):
            notes.append(f"That is beyond the {horizon_days}-day forecast horizon, so I can't give a grounded answer for it yet.")
        else:
            notes.append("That time has already passed.")
    elif len(clipped) < len(intervals) or any(e < ie for (s, e), (_, ie) in zip(clipped, intervals)):
        if any(e > horizon for _, e in intervals):
            notes.append(f"Forecasts only reach {horizon_days} days ahead, so later hours were left out.")
    return TimeWindow(clipped, _label(clipped, parts, specific, not explicit_day and not explicit_part),
                      explicit_day, explicit_part, specific, target, compare,
                      assumed=not explicit_day and not explicit_part, notes=notes)


def _date(today: date, d: int, m: int) -> date | None:
    try:
        cand = date(today.year, m, d)
    except ValueError:
        return None
    if cand < today - timedelta(days=1):
        try:
            cand = date(today.year + 1, m, d)
        except ValueError:
            return None
    return cand


def _label(intervals, parts, specific, assumed) -> str:
    if not intervals:
        return "no forecastable hours"
    if specific is not None:
        return specific.strftime("%a %d %b, %I:%M %p").replace(" 0", " ")
    if assumed:
        return "the next 48 hours (no time was given)"
    segs = []
    for s, e in intervals:
        end = "24:00" if e.hour == 0 and e.date() > s.date() else f"{e:%H:%M}"
        segs.append(f"{s:%a %d %b} {s:%H:%M}–{end}")
    return "; ".join(segs[:6]) + (" …" if len(segs) > 6 else "")
