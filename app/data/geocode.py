"""Place name -> coordinates using Nominatim (OSM), with Photon (OSM) as fallback."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .. import config
from ..geo import haversine_km
from .http import HTTP

log = logging.getLogger("geocode")


@dataclass
class Place:
    name: str            # what the user typed
    display_name: str    # resolved full name
    lat: float
    lon: float
    kind: str            # e.g. suburb, city, park
    extent_km: float     # rough size of the matched area (diagonal / 2)
    source: str

    @property
    def is_city_scale(self) -> bool:
        return self.extent_km >= 6 or self.kind in {"city", "state_district", "county", "district", "town"}


def _extent_from_bbox(s: float, n: float, w: float, e: float) -> float:
    return haversine_km(s, w, n, e) / 2


async def _nominatim(q: str) -> list[Place]:
    data = await HTTP.get_json(
        config.NOMINATIM_URL,
        {"q": q, "format": "jsonv2", "limit": 5, "countrycodes": "in", "addressdetails": 0},
        ttl=config.TTL_GEOCODE,
        source="Nominatim (OpenStreetMap geocoding)",
    )
    out = []
    for d in data or []:
        try:
            s, n, w, e = (float(x) for x in d.get("boundingbox", [0, 0, 0, 0]))
            out.append(Place(q, d.get("display_name", q), float(d["lat"]), float(d["lon"]),
                             d.get("addresstype") or d.get("type") or "place",
                             _extent_from_bbox(s, n, w, e), "Nominatim"))
        except (KeyError, ValueError):
            continue
    return out


async def _photon(q: str) -> list[Place]:
    # Photon does not filter by country; bias to India's centre and filter below.
    data = await HTTP.get_json(
        config.PHOTON_URL, {"q": q, "limit": 5, "lat": 21.0, "lon": 78.0},
        ttl=config.TTL_GEOCODE, source="Photon (OpenStreetMap geocoding)",
    )
    out = []
    for f in (data or {}).get("features", []):
        p = f.get("properties", {})
        if p.get("countrycode", "").upper() not in ("IN", ""):
            continue
        lon, lat = f["geometry"]["coordinates"]
        ext = p.get("extent")  # [minLon, maxLat, maxLon, minLat]
        extent_km = _extent_from_bbox(ext[3], ext[1], ext[0], ext[2]) if ext else 1.0
        label = ", ".join(x for x in [p.get("name"), p.get("city"), p.get("state")] if x)
        out.append(Place(q, label or q, float(lat), float(lon), p.get("type", "place"), extent_km, "Photon"))
    return out


async def geocode(q: str) -> Place | None:
    """Return the best match for a free-text place name in India, or None."""
    q = q.strip().strip(",.")
    if not q:
        return None
    for attempt in (q, f"{q}, India"):
        for fn in (_nominatim, _photon):
            try:
                results = await fn(attempt)
            except Exception as exc:  # noqa: BLE001
                log.warning("geocoder %s failed for %r: %s", fn.__name__, attempt, exc)
                continue
            if results:
                return results[0]
    return None
