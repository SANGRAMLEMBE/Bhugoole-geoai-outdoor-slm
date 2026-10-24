"""OpenStreetMap via the Overpass API: candidate venues and their geospatial context."""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field

from .. import config
from ..activities import Activity
from ..geo import bbox_around, bbox_overlap_fraction, bearing_deg, compass, haversine_km, nearest_point_on_polyline
from .http import HTTP
from .osm_local import LOCAL_OSM, tags_match

log = logging.getLogger("overpass")


@dataclass
class Candidate:
    osm_type: str
    osm_id: int
    name: str
    named: bool
    lat: float
    lon: float
    kind: str
    quality: float
    area_km2: float
    distance_km: float
    lit: bool = False
    tags: dict = field(default_factory=dict)
    context: "Context | None" = None
    travel_min: float | None = None
    travel_source: str = ""
    elevation_m: float | None = None

    @property
    def key(self) -> str:
        return f"{self.osm_type}/{self.osm_id}"

    @property
    def osm_url(self) -> str:
        return f"https://www.openstreetmap.org/{self.osm_type}/{self.osm_id}"


@dataclass
class RoadExposure:
    name: str
    highway: str
    distance_km: float
    bearing: float           # direction from the venue towards the road
    weight: float            # traffic proxy by road class


@dataclass
class Source:
    name: str
    kind: str
    distance_km: float
    bearing: float


@dataclass
class Context:
    roads: list[RoadExposure] = field(default_factory=list)
    industry: list[Source] = field(default_factory=list)
    green_fraction: float = 0.0
    water: list[Source] = field(default_factory=list)
    water_points: list[tuple[float, float]] = field(default_factory=list)  # (distance_km, bearing)
    power_line_km: float | None = None

    def nearest_road(self) -> RoadExposure | None:
        return min(self.roads, key=lambda r: r.distance_km) if self.roads else None

    def nearest_water(self) -> Source | None:
        return min(self.water, key=lambda w: w.distance_km) if self.water else None

    def water_toward(self, azimuth: float, half_width: float = 25, max_km: float = 4.0) -> bool:
        return any(d <= max_km and abs((b - azimuth + 180) % 360 - 180) <= half_width
                   for d, b in self.water_points)


ROAD_WEIGHTS = {"motorway": 1.0, "motorway_link": 0.7, "trunk": 1.0, "trunk_link": 0.7,
                "primary": 0.8, "primary_link": 0.5, "secondary": 0.5}


async def _overpass(query: str, source: str) -> dict:
    """Hedged request: start the next mirror when one fails or is slow; first success wins.

    Public Overpass servers are often saturated, so the whole call is capped at
    OVERPASS_BUDGET_S and the answer degrades gracefully instead of hanging.
    """
    global _down_until

    async def one(url):
        return await HTTP.post_json(url, {"data": query}, ttl=config.TTL_OSM, source=source, retries=0)

    loop = asyncio.get_running_loop()
    if loop.time() < _down_until:  # circuit breaker: every mirror failed moments ago
        raise RuntimeError("Overpass unavailable (all mirrors failed recently; retrying shortly)")
    pending: set[asyncio.Task] = set()
    urls = list(config.OVERPASS_URLS)
    deadline = loop.time() + config.OVERPASS_BUDGET_S
    last: Exception | None = None
    try:
        while urls or pending:
            if urls:
                url = urls.pop(0)
                pending.add(asyncio.create_task(one(url), name=url))
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            wait = min(config.OVERPASS_HEDGE_S, remaining) if urls else remaining
            done, pending = await asyncio.wait(pending, timeout=wait, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception() is None:
                    return t.result()
                last = t.exception()
                log.warning("Overpass mirror %s failed: %s", t.get_name(), last or type(last).__name__)
    finally:
        for t in pending:
            t.cancel()
    _down_until = loop.time() + config.OVERPASS_COOLDOWN_S
    raise RuntimeError(f"All Overpass mirrors failed or timed out: {last!r}")


_down_until = 0.0


async def _osm(live, local) -> dict:
    """Live Overpass first, then the offline Geofabrik layer (see config.OSM_SOURCE_ORDER)."""
    order = config.OSM_SOURCE_ORDER
    if order == "local_first" and LOCAL_OSM.available:
        return await asyncio.to_thread(local)
    try:
        return await live()
    except Exception as exc:  # noqa: BLE001
        if order == "live_only" or not LOCAL_OSM.available:
            raise
        log.warning("Overpass failed (%s); answering from the offline OSM layer", exc)
        return await asyncio.to_thread(local)


def _center_and_area(el: dict) -> tuple[float, float, float] | None:
    if el.get("type") == "node":
        return el["lat"], el["lon"], 0.0
    b = el.get("bounds")
    if not b:
        c = el.get("center")
        return (c["lat"], c["lon"], 0.0) if c else None
    lat = (b["minlat"] + b["maxlat"]) / 2
    lon = (b["minlon"] + b["maxlon"]) / 2
    h = haversine_km(b["minlat"], lon, b["maxlat"], lon)
    w = haversine_km(lat, b["minlon"], lat, b["maxlon"])
    return lat, lon, h * w * 0.6  # bounding boxes over-estimate polygon area


async def find_venues(activity: Activity, lat: float, lon: float, radius_km: float) -> list[Candidate]:
    r = int(radius_km * 1000)
    body = "\n".join(f"  {v.ql}(around:{r},{lat:.5f},{lon:.5f});" for v in activity.venues)
    query = f"[out:json][timeout:40];\n(\n{body}\n);\nout tags bb qt 600;"
    data = await _osm(lambda: _overpass(query, "OpenStreetMap via Overpass API (venues)"),
                      lambda: LOCAL_OSM.venues(activity.venues, lat, lon, radius_km))
    out: list[Candidate] = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        geo = _center_and_area(el)
        if not geo:
            continue
        clat, clon, area = geo
        venue = next((v for v in activity.venues if tags_match(tags, v.ql, el.get("type"))), None)
        if venue is None:
            continue
        d = haversine_km(lat, lon, clat, clon)
        if d > radius_km * 1.15:
            continue
        name = tags.get("name:en") or tags.get("name")
        named = bool(name)
        if not name:
            name = f"Unnamed {venue.label}, {d:.1f} km {compass(bearing_deg(lat, lon, clat, clon))} of the search centre"
        out.append(Candidate(el["type"], el["id"], name, named, clat, clon, venue.label, venue.quality, area, d,
                             lit=tags.get("lit") == "yes", tags=tags))
    return _dedupe(out)


def _dedupe(cands: list[Candidate]) -> list[Candidate]:
    """Collapse clusters (e.g. six courts in one complex, or a park mapped twice)."""
    cands = sorted(cands, key=lambda c: (-c.quality, -c.named, -c.area_km2))
    kept: list[Candidate] = []
    for c in cands:
        dup = next((k for k in kept if haversine_km(k.lat, k.lon, c.lat, c.lon) < 0.25
                    or (c.named and k.named and k.name.lower() == c.name.lower()
                        and haversine_km(k.lat, k.lon, c.lat, c.lon) < 2.0)), None)
        if dup is None:
            kept.append(c)
    return kept


def shortlist(cands: list[Candidate], radius_km: float, activity: Activity, n: int) -> list[Candidate]:
    """Pick the places worth scoring hour by hour, using a cheap static prior."""
    def prior(c: Candidate) -> float:
        size = min(1.0, math.log10(1 + c.area_km2 * 100) / 2) if c.area_km2 else 0.3
        if activity.id in ("running", "walking", "picnic", "kite", "cycling") and c.area_km2 and c.area_km2 < 0.01:
            size *= 0.3  # pocket parks are poor for these
        closeness = max(0.0, 1 - c.distance_km / max(radius_km, 0.5))
        return 0.45 * c.quality + 0.2 * size + 0.1 * c.named + 0.25 * closeness
    return sorted(cands, key=prior, reverse=True)[:n]


def _context_query(cands: list[Candidate], want_power: bool) -> str:
    parts = ["[out:json][timeout:60];"]
    for c in cands:
        la, lo = f"{c.lat:.5f}", f"{c.lon:.5f}"
        s, w, n, e = bbox_around(c.lat, c.lon, 1.3)
        s2, w2, n2, e2 = bbox_around(c.lat, c.lon, 4.0)
        parts += [
            f'way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary)$"](around:1200,{la},{lo});',
            f"out tags geom({s:.5f},{w:.5f},{n:.5f},{e:.5f});",
            f'(nwr["landuse"~"^(industrial|quarry|landfill)$"](around:3000,{la},{lo});'
            f'nwr["man_made"="works"](around:3000,{la},{lo});nwr["power"="plant"](around:3000,{la},{lo}););',
            "out tags bb;",
            f'(nwr["leisure"~"^(park|garden|nature_reserve|golf_course)$"](around:600,{la},{lo});'
            f'nwr["landuse"~"^(forest|grass|meadow|recreation_ground|village_green)$"](around:600,{la},{lo});'
            f'nwr["natural"~"^(wood|scrub|grassland|wetland|heath)$"](around:600,{la},{lo}););',
            "out tags bb;",
            f'(way["natural"="coastline"](around:4000,{la},{lo});'
            f'nwr["natural"~"^(water|bay|beach)$"](around:4000,{la},{lo});'
            f'way["waterway"~"^(river|riverbank)$"](around:4000,{la},{lo}););',
            f"out tags geom({s2:.5f},{w2:.5f},{n2:.5f},{e2:.5f});",
        ]
        if want_power:
            parts += [f'way["power"="line"](around:300,{la},{lo});', "out tags geom;"]
    return "\n".join(parts)


def _geom_coords(el: dict) -> list[tuple[float, float]]:
    if el.get("type") == "way":
        return [(p["lat"], p["lon"]) for p in el.get("geometry", []) if p]
    if el.get("type") == "relation":
        pts = []
        for m in el.get("members", []):
            pts += [(p["lat"], p["lon"]) for p in m.get("geometry", []) or [] if p]
        return pts
    if el.get("type") == "node" and "lat" in el:
        return [(el["lat"], el["lon"])]
    return []


GREEN = {"leisure": {"park", "garden", "nature_reserve", "golf_course"},
         "landuse": {"forest", "grass", "meadow", "recreation_ground", "village_green"},
         "natural": {"wood", "scrub", "grassland", "wetland", "heath"}}


def _is_green(tags):
    return any(tags.get(k) in v for k, v in GREEN.items())


def _is_industry(tags):
    return tags.get("landuse") in {"industrial", "quarry", "landfill"} or tags.get("man_made") == "works" \
        or tags.get("power") == "plant"


UNNAMED_WATER = {"coastline": "sea / coastline", "water": "unnamed lake or pond", "river": "unnamed river",
                 "riverbank": "unnamed river", "bay": "unnamed bay", "beach": "unnamed beach"}


def _is_water(tags):
    return tags.get("natural") in {"water", "bay", "beach", "coastline"} or tags.get("waterway") in {"river", "riverbank"}


async def add_context(cands: list[Candidate], want_power: bool = False) -> None:
    """Attach roads, industry, green cover, water and power lines to each candidate (in place)."""
    if not cands:
        return
    data = await _osm(lambda: _overpass(_context_query(cands, want_power),
                                        "OpenStreetMap via Overpass API (geospatial context)"),
                      lambda: LOCAL_OSM.context([(c.lat, c.lon) for c in cands], want_power))
    elements = {}
    for el in data.get("elements", []):
        elements[(el["type"], el["id"])] = el  # the same feature may be returned for several venues
    els = list(elements.values())
    for c in cands:
        ctx = Context()
        green = 0.0
        for el in els:
            tags = el.get("tags", {})
            hw = tags.get("highway")
            if hw in ROAD_WEIGHTS:
                d, b = nearest_point_on_polyline(c.lat, c.lon, _geom_coords(el))
                if d <= 1.0:
                    ctx.roads.append(RoadExposure(tags.get("name:en") or tags.get("name") or tags.get("ref")
                                                  or f"unnamed {hw} road", hw, d, b, ROAD_WEIGHTS[hw]))
                continue
            if tags.get("power") == "line":
                d, _ = nearest_point_on_polyline(c.lat, c.lon, _geom_coords(el))
                ctx.power_line_km = d if ctx.power_line_km is None else min(ctx.power_line_km, d)
                continue
            if _is_industry(tags):
                geo = _center_and_area(el)
                if geo:
                    d = haversine_km(c.lat, c.lon, geo[0], geo[1])
                    if d <= 3.0:
                        ctx.industry.append(Source(tags.get("name") or tags.get("landuse") or "industrial area",
                                                   "industry", d, bearing_deg(c.lat, c.lon, geo[0], geo[1])))
                continue
            if _is_green(tags) and "bounds" in el:
                b = el["bounds"]
                green += 0.6 * bbox_overlap_fraction((b["minlat"], b["minlon"], b["maxlat"], b["maxlon"]),
                                                     c.lat, c.lon, 0.5)
                continue
            if _is_water(tags):
                coords = _geom_coords(el)
                if not coords:
                    continue
                d, b = nearest_point_on_polyline(c.lat, c.lon, coords)
                if d <= 4.0:
                    label = tags.get("name") or UNNAMED_WATER.get(tags.get("natural") or tags.get("waterway"),
                                                                  "unnamed water body")
                    ctx.water.append(Source(label, "water", d, b))
                    step = max(1, len(coords) // 60)
                    for la, lo in coords[::step]:
                        dd = haversine_km(c.lat, c.lon, la, lo)
                        if dd <= 4.0:
                            ctx.water_points.append((dd, bearing_deg(c.lat, c.lon, la, lo)))
        # Venues that are themselves green spaces count towards their own cover.
        if c.kind in {"park", "garden", "nature reserve", "golf course", "protected area"}:
            green = max(green, min(0.9, 0.25 + c.area_km2 * 2))
        ctx.green_fraction = min(1.0, green)
        ctx.roads = _merge_roads(ctx.roads)
        ctx.industry.sort(key=lambda s: s.distance_km)
        ctx.water.sort(key=lambda s: s.distance_km)
        c.context = ctx


def _merge_roads(roads: list[RoadExposure]) -> list[RoadExposure]:
    """One entry per named road (the closest segment), heaviest roads first."""
    best: dict[str, RoadExposure] = {}
    for r in roads:
        k = r.name.lower()
        if k not in best or r.distance_km < best[k].distance_km:
            best[k] = r
    return sorted(best.values(), key=lambda r: (r.distance_km / r.weight))[:8]
