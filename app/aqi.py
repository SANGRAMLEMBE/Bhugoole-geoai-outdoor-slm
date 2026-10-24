"""India's National Air Quality Index (CPCB) computed from modelled pollutant concentrations.

CPCB defines the index on 24-hour (PM, NO2) and 8-hour (O3) averages. O3 is averaged
over the trailing 8 hours (app/data/weather.py); PM and NO2 are kept hourly so the
answer reflects exposure during the activity, so the result is an *indicative*
hourly AQI, labelled as such in every answer.
"""
from __future__ import annotations

from . import config

# (concentration low, high) -> (index low, high)
BREAKPOINTS = {
    "pm2_5": [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200), (90, 120, 200, 300),
              (120, 250, 300, 400), (250, 500, 400, 500)],
    "pm10": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200), (250, 350, 200, 300),
             (350, 430, 300, 400), (430, 600, 400, 500)],
    "nitrogen_dioxide": [(0, 40, 0, 50), (40, 80, 50, 100), (80, 180, 100, 200), (180, 280, 200, 300),
                         (280, 400, 300, 400), (400, 800, 400, 500)],
    "ozone": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 168, 100, 200), (168, 208, 200, 300),
              (208, 748, 300, 400), (748, 1000, 400, 500)],
}
CATEGORIES = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"), (300, "Poor"), (400, "Very Poor"),
              (10_000, "Severe")]
NAMES = {"pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO2", "ozone": "O3"}


def sub_index(pollutant: str, c: float | None) -> float | None:
    if c is None:
        return None
    for lo, hi, ilo, ihi in BREAKPOINTS[pollutant]:
        if c <= hi:
            return ilo + (ihi - ilo) * (max(c, lo) - lo) / (hi - lo)
    return 500.0


def naqi(values: dict) -> tuple[int | None, str | None]:
    """Return (AQI, dominant pollutant). Needs at least one PM value, as CPCB requires.

    Modelled CAMS ozone is far too high over Indian cities (e.g. 460 ug/m3 forecast for Mumbai),
    so by default it is left out and the index uses PM2.5, PM10 and NO2 (CPCB's minimum of three).
    """
    subs = {p: sub_index(p, values.get(p)) for p in BREAKPOINTS
            if p != "ozone" or config.AQI_USE_MODELLED_O3}
    if subs["pm2_5"] is None and subs["pm10"] is None:
        return None, None
    valid = {p: v for p, v in subs.items() if v is not None}
    dom = max(valid, key=valid.get)
    return round(valid[dom]), NAMES[dom]


def category(aqi: float | None) -> str:
    if aqi is None:
        return "unknown"
    return next(name for cap, name in CATEGORIES if aqi <= cap)
