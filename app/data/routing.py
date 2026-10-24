"""Travel time from the user's area to each candidate (OSRM table service)."""
from __future__ import annotations

import logging
from datetime import datetime

from .. import config
from ..geo import haversine_km
from .http import HTTP

log = logging.getLogger("routing")

FALLBACK_SPEED_KMH = 22.0   # typical urban Indian driving speed, used only if OSRM is unreachable
DETOUR = 1.35               # straight line -> road distance


async def travel_minutes(origin: tuple[float, float], dests: list[tuple[float, float]]) -> tuple[list[float], str]:
    """Free-flow driving minutes from origin to each destination, plus a label for the source used."""
    if not dests:
        return [], ""
    coords = ";".join(f"{lon:.5f},{lat:.5f}" for lat, lon in [origin, *dests])
    url = f"{config.OSRM_URL}/table/v1/driving/{coords}"
    try:
        data = await HTTP.get_json(url, {"sources": "0", "annotations": "duration,distance"},
                                   ttl=config.TTL_ROUTE, source="OSRM routing (OpenStreetMap road network)")
        row = data["durations"][0][1:]
        if data.get("code") == "Ok" and all(v is not None for v in row):
            return [v / 60 for v in row], "OSRM"
    except Exception as exc:  # noqa: BLE001
        log.warning("OSRM unavailable, using distance estimate: %s", exc)
    return [haversine_km(*origin, *d) * DETOUR / FALLBACK_SPEED_KMH * 60 for d in dests], "estimate"


def traffic_factor(t: datetime) -> tuple[float, str]:
    """Heuristic congestion multiplier for Indian metros. OSRM durations are free-flow.

    No free, open live-traffic feed exists for India, so this is a disclosed heuristic:
    weekday peaks 08-11 and 17-21 are slowest.
    """
    h, weekday = t.hour, t.weekday() < 5
    if weekday and (8 <= h < 11 or 17 <= h < 21):
        return 1.6, "weekday rush hour"
    if 7 <= h < 22:
        return (1.25, "daytime traffic") if weekday else (1.15, "weekend traffic")
    return 1.0, "light traffic"
