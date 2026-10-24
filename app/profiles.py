"""Personal profiles that tighten the limits for a particular person."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    keywords: tuple[str, ...]
    heat_shift: float = 0.0        # lowers comfortable and maximum feels-like temperature (deg C)
    aqi_factor: float = 1.0        # multiplies AQI thresholds (<1 = stricter)
    uv_factor: float = 1.0         # multiplies UV penalty
    road_factor: float = 1.0       # multiplies traffic-exposure penalty
    travel_factor: float = 1.0     # multiplies travel-time penalty
    exertion_cap: float | None = None
    prepare: tuple[str, ...] = ()
    watch: tuple[str, ...] = ()


PROFILES: dict[str, Profile] = {
    p.id: p for p in [
        Profile(
            "asthma", "asthma / respiratory condition",
            (r"asthma(?:tic)?", r"copd", r"breathing (?:problem|issue)s?", r"bronchitis", r"wheez\w*",
             r"respiratory", r"allerg(?:y|ies|ic)", r"lung"),
            heat_shift=1.0, aqi_factor=0.6, road_factor=2.0, uv_factor=1.0,
            prepare=("Reliever inhaler within reach", "Warm up gently; keep the pace conversational"),
            watch=("Stop if you feel chest tightness, cough or wheeze", "Avoid roadside stretches and smoke",
                   "Thunderstorms can trigger asthma attacks"),
        ),
        Profile(
            "senior", "senior citizen",
            (r"senior(?:s| citizens?)?", r"elderly", r"old(?:er)? (?:people|person|man|woman|parents?)",
             r"(?:my |our )?(?:grand)?parents", r"grand(?:pa|ma|father|mother)", r"aged \d{2}",
             r"(?:6|7|8|9)\d[- ]?(?:year|yr)s?[- ]old", r"retired", r"\bdad\b", r"\bmom\b", r"\bmum\b",
             r"father", r"mother"),
            heat_shift=4.0, aqi_factor=0.75, uv_factor=1.4, road_factor=1.3, travel_factor=1.5, exertion_cap=1.5,
            prepare=("Water, a cap and any regular medication", "Comfortable footwear with grip", "A phone with emergency contacts"),
            watch=("Heat exhaustion builds faster with age: rest in shade", "Uneven paths and kerbs"),
        ),
        Profile(
            "child", "children",
            (r"kids?'?s?", r"child(?:ren)?", r"toddlers?", r"bab(?:y|ies)", r"son", r"daughter", r"school ?kids",
             r"under[- ]?1\d", r"(?:1[0-7]|\d)[- ]?(?:year|yr)s?[- ]old"),
            heat_shift=3.0, aqi_factor=0.7, uv_factor=1.5, road_factor=1.3, travel_factor=1.4,
            prepare=("Extra water and snacks; drinks breaks every 20 minutes", "Sunscreen and caps for kids",
                     "A basic first-aid kit"),
            watch=("Children overheat faster and rarely say so: watch for flushed faces and fatigue",),
        ),
        Profile(
            "pregnancy", "pregnancy",
            (r"pregnan(?:t|cy)", r"expecting", r"trimester"),
            heat_shift=5.0, aqi_factor=0.7, uv_factor=1.3, road_factor=1.5, travel_factor=1.3, exertion_cap=1.5,
            prepare=("Plenty of water and a snack", "Know where the nearest restroom and shade are"),
            watch=("Avoid overheating and strenuous climbs; check with your doctor for anything new",),
        ),
        Profile(
            "heart", "heart condition",
            (r"heart (?:condition|problem|patient|disease)", r"cardiac", r"blood pressure", r"hypertension", r"bp patient"),
            heat_shift=4.0, aqi_factor=0.7, uv_factor=1.2, road_factor=1.5, exertion_cap=1.5,
            prepare=("Prescribed medication", "A companion if possible"),
            watch=("Stop at chest pain, unusual breathlessness or dizziness and seek help",),
        ),
    ]
}


def combine(ids: list[str]) -> Profile:
    """Merge several profiles, keeping the strictest value of each limit."""
    ps = [PROFILES[i] for i in ids if i in PROFILES]
    if not ps:
        return Profile("none", "", ())
    caps = [p.exertion_cap for p in ps if p.exertion_cap]
    return Profile(
        "+".join(p.id for p in ps), " and ".join(p.label for p in ps), (),
        heat_shift=max(p.heat_shift for p in ps),
        aqi_factor=min(p.aqi_factor for p in ps),
        uv_factor=max(p.uv_factor for p in ps),
        road_factor=max(p.road_factor for p in ps),
        travel_factor=max(p.travel_factor for p in ps),
        exertion_cap=min(caps) if caps else None,
        prepare=tuple(x for p in ps for x in p.prepare),
        watch=tuple(x for p in ps for x in p.watch),
    )
