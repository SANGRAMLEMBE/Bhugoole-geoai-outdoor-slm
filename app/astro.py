"""Sun position and moon phase (standard NOAA / Meeus approximations, accurate to ~0.5 deg)."""
from __future__ import annotations

import math
from datetime import datetime, timezone


def _julian_day(t: datetime) -> float:
    t = t.astimezone(timezone.utc)
    return t.timestamp() / 86400.0 + 2440587.5


def sun_position(lat: float, lon: float, t: datetime) -> tuple[float, float]:
    """(elevation_deg, azimuth_deg clockwise from north) for an aware datetime."""
    jd = _julian_day(t)
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    gmst = (18.697374558 + 24.06570982441908 * n) % 24
    lst = math.radians(((gmst * 15 + lon) % 360))
    ha = lst - ra
    phi = math.radians(lat)
    el = math.asin(math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(ha))
    az = math.atan2(-math.sin(ha), math.tan(dec) * math.cos(phi) - math.sin(phi) * math.cos(ha))
    return math.degrees(el), (math.degrees(az) + 360) % 360


def moon_illumination(t: datetime) -> float:
    """Illuminated fraction of the moon (0 new .. 1 full)."""
    days = _julian_day(t) - 2451550.1  # reference new moon 2000-01-06
    phase = (days % 29.530588853) / 29.530588853
    return (1 - math.cos(2 * math.pi * phase)) / 2
