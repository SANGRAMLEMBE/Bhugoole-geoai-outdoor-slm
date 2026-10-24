"""A synthetic stand-in for the live APIs, used ONLY by the offline test-suite.

It mimics the response *shapes* of Nominatim, Overpass, Open-Meteo and OSRM so the
whole pipeline can be exercised without network access. Values are invented and
deterministic; the running assistant never uses this module.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import respx

PLACES = {
    "powai": (19.1176, 72.9060, "suburb"), "koregaon park": (18.5362, 73.8940, "suburb"),
    "whitefield": (12.9698, 77.7500, "suburb"), "pune": (18.5204, 73.8567, "city"),
    "andheri": (19.1197, 72.8468, "suburb"), "bandra": (19.0596, 72.8295, "suburb"),
    "lonavala": (18.7546, 73.4062, "town"), "bengaluru": (12.9716, 77.5946, "city"),
    "hinjewadi": (18.5913, 73.7389, "suburb"), "shivaji park": (19.0270, 72.8383, "park"),
    "delhi": (28.6139, 77.2090, "city"),
}

VENUE_TEMPLATES = [
    ({"leisure": "park", "name": "Lakeside Garden"}, 0.8, 70, 0.006),
    ({"leisure": "park", "name": "Hill Park"}, 2.5, 200, 0.012),
    ({"leisure": "garden", "name": "Rose Garden"}, 1.6, 300, 0.004),
    ({"leisure": "track", "sport": "running"}, 1.1, 20, 0.002),
    ({"leisure": "pitch", "sport": "cricket", "name": "Gymkhana Cricket Ground"}, 3.0, 110, 0.004),
    ({"leisure": "park", "name": "Azad Maidan"}, 4.2, 160, 0.008),
    ({"leisure": "pitch", "sport": "soccer", "name": "Five-a-side Turf"}, 2.0, 250, 0.002),
    ({"leisure": "pitch", "sport": "tennis", "lit": "yes"}, 1.9, 40, 0.001),
    ({"leisure": "sports_centre", "sport": "tennis", "name": "City Tennis Club"}, 5.5, 320, 0.002),
    ({"leisure": "golf_course", "name": "Greens Golf Club"}, 12.0, 30, 0.02),
    ({"leisure": "golf_course", "name": "Hilltop Golf Course"}, 26.0, 120, 0.03),
    ({"historic": "fort", "name": "Rock Fort"}, 9.0, 200, 0.01),
    ({"natural": "peak", "name": "Eagle Peak"}, 14.0, 90, 0.0),
    ({"tourism": "viewpoint", "name": "Sunset Point"}, 3.5, 260, 0.0),
    ({"natural": "beach", "name": "Silver Beach"}, 4.0, 270, 0.01),
    ({"amenity": "events_venue", "name": "Royal Lawns"}, 3.2, 180, 0.003),
    ({"tourism": "picnic_site", "name": "Picnic Point"}, 2.8, 20, 0.0),
    ({"leisure": "nature_reserve", "name": "Green Valley Reserve"}, 7.0, 10, 0.5),
    ({"highway": "cycleway", "name": "Lake Cycle Track"}, 1.0, 100, 0.001),
    ({"man_made": "pier", "name": "Old Pier"}, 4.5, 250, 0.0),
]


def _offset(lat, lon, d_km, bearing):
    b = math.radians(bearing)
    return lat + d_km * math.cos(b) / 111.32, lon + d_km * math.sin(b) / (111.32 * math.cos(math.radians(lat)))


def nominatim(request: httpx.Request):
    q = parse_qs(urlparse(str(request.url)).query).get("q", [""])[0].lower().replace(", india", "")
    hit = PLACES.get(q.strip())
    if not hit:
        return httpx.Response(200, json=[])
    lat, lon, kind = hit
    ext = 0.08 if kind in ("city", "town") else 0.01
    return httpx.Response(200, json=[{
        "lat": str(lat), "lon": str(lon), "display_name": f"{q.title()}, Maharashtra, India",
        "addresstype": kind, "type": kind, "boundingbox": [str(lat - ext), str(lat + ext), str(lon - ext), str(lon + ext)],
    }])


def overpass(request: httpx.Request):
    body = parse_qs(request.content.decode())["data"][0]
    if "qt 600" in body:  # venue query
        m = re.search(r"around:(\d+),([\d.\-]+),([\d.\-]+)", body)
        r_m, lat, lon = int(m.group(1)), float(m.group(2)), float(m.group(3))
        els = []
        for i, (tags, d, b, area) in enumerate(VENUE_TEMPLATES):
            if d * 1000 > r_m:
                continue
            la, lo = _offset(lat, lon, d, b)
            if area:
                h = math.sqrt(area) / 111.32 / 2
                els.append({"type": "way", "id": 1000 + i, "tags": tags,
                            "bounds": {"minlat": la - h, "minlon": lo - h, "maxlat": la + h, "maxlon": lo + h}})
            else:
                els.append({"type": "node", "id": 1000 + i, "lat": la, "lon": lo, "tags": tags})
        return httpx.Response(200, json={"elements": els})
    # context query: one block per candidate
    els, seen = [], set()
    for m in re.finditer(r'way\["highway"[^\n]*around:1200,([\d.\-]+),([\d.\-]+)', body):
        lat, lon = float(m.group(1)), float(m.group(2))
        key = (round(lat, 3), round(lon, 3))
        if key in seen:
            continue
        seen.add(key)
        # An expressway running north-south 400 m EAST of every venue
        a = _offset(lat, lon, 0.4, 90)
        geom = [{"lat": a[0] + dy, "lon": a[1]} for dy in (-0.01, -0.005, 0, 0.005, 0.01)]
        els.append({"type": "way", "id": int(abs(lat * 1e4)) * 10 + 1, "tags": {"highway": "trunk", "name": "Eastern Express Highway"},
                    "geometry": geom})
        ind = _offset(lat, lon, 2.0, 100)
        els.append({"type": "way", "id": int(abs(lat * 1e4)) * 10 + 2, "tags": {"landuse": "industrial", "name": "MIDC Estate"},
                    "bounds": {"minlat": ind[0] - 0.003, "minlon": ind[1] - 0.003, "maxlat": ind[0] + 0.003, "maxlon": ind[1] + 0.003}})
        els.append({"type": "way", "id": int(abs(lat * 1e4)) * 10 + 3, "tags": {"leisure": "park"},
                    "bounds": {"minlat": lat - 0.002, "minlon": lon - 0.002, "maxlat": lat + 0.002, "maxlon": lon + 0.002}})
        w = _offset(lat, lon, 1.5, 270)
        els.append({"type": "way", "id": int(abs(lat * 1e4)) * 10 + 4, "tags": {"natural": "coastline"},
                    "geometry": [{"lat": w[0] + dy, "lon": w[1]} for dy in (-0.02, -0.01, 0, 0.01, 0.02)]})
    return httpx.Response(200, json={"elements": els})


def _hourly(lat, lon, days, start):
    times, H = [], {}
    keys = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation_probability",
            "precipitation", "weather_code", "cloud_cover", "visibility", "wind_speed_10m", "wind_gusts_10m",
            "wind_direction_10m", "uv_index", "is_day"]
    for k in keys:
        H[k] = []
    for i in range(days * 24):
        t = start + timedelta(hours=i)
        times.append(t.strftime("%Y-%m-%dT%H:%M"))
        hr = t.hour
        temp = 26 + 6 * math.sin((hr - 8) / 24 * 2 * math.pi) + (lat - 19) * -0.3
        H["temperature_2m"].append(round(temp, 1))
        H["apparent_temperature"].append(round(temp + 3.5 + (2 if 11 <= hr <= 16 else 0), 1))
        H["relative_humidity_2m"].append(70)
        storm = (t.day % 2 == 0 and 15 <= hr <= 17)
        H["precipitation_probability"].append(75 if storm else (35 if 13 <= hr <= 18 else 5))
        H["precipitation"].append(3.0 if storm else 0.0)
        H["weather_code"].append(95 if storm and hr == 16 else (61 if storm else 2))
        H["cloud_cover"].append(40 if 15 <= hr <= 19 else 20)
        H["visibility"].append(9000)
        H["wind_speed_10m"].append(14 if 12 <= hr <= 20 else 6)
        H["wind_gusts_10m"].append(28 if 12 <= hr <= 20 else 12)
        # land breeze from the east in the morning, sea breeze from the west in the afternoon/evening
        H["wind_direction_10m"].append(270 if 12 <= hr <= 21 else 90)
        H["uv_index"].append(max(0.0, round(9 * math.sin((hr - 6) / 12 * math.pi), 1)) if 6 <= hr <= 18 else 0.0)
        H["is_day"].append(1 if 6 <= hr <= 18 else 0)
    daily = {"time": [], "sunrise": [], "sunset": []}
    for d in range(days):
        day = start + timedelta(days=d)
        daily["time"].append(day.strftime("%Y-%m-%d"))
        daily["sunrise"].append(day.replace(hour=6, minute=31).strftime("%Y-%m-%dT%H:%M"))
        daily["sunset"].append(day.replace(hour=18, minute=7).strftime("%Y-%m-%dT%H:%M"))
    return {"latitude": lat, "longitude": lon, "elevation": 14.0, "hourly": {"time": times, **H}, "daily": daily}


def _aq(lat, lon, days, start):
    times, H = [], {k: [] for k in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "us_aqi"]}
    for i in range(days * 24):
        t = start + timedelta(hours=i)
        hr = t.hour
        times.append(t.strftime("%Y-%m-%dT%H:%M"))
        morning = 6 <= hr <= 10 or hr >= 21 or hr < 2
        H["pm2_5"].append(58.0 if morning else 28.0)
        H["pm10"].append(110.0 if morning else 70.0)
        H["nitrogen_dioxide"].append(45.0 if morning else 20.0)
        H["ozone"].append(40.0 if morning else 85.0)
        H["us_aqi"].append(150 if morning else 85)
    return {"latitude": lat, "longitude": lon, "hourly": {"time": times, **H}}


def _multi(request: httpx.Request, fn):
    qs = parse_qs(urlparse(str(request.url)).query)
    lats = [float(x) for x in qs["latitude"][0].split(",")]
    lons = [float(x) for x in qs["longitude"][0].split(",")]
    days = int(qs.get("forecast_days", ["7"])[0])
    start = FAKE_NOW.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    items = [fn(la, lo, days, start) for la, lo in zip(lats, lons)]
    return httpx.Response(200, json=items[0] if len(items) == 1 else items)


def osrm(request: httpx.Request):
    path = urlparse(str(request.url)).path
    coords = [tuple(map(float, c.split(","))) for c in path.split("/")[-1].split(";")]
    lon0, lat0 = coords[0]
    row = [0.0]
    for lon, lat in coords[1:]:
        d = math.hypot((lat - lat0) * 111.32, (lon - lon0) * 111.32 * math.cos(math.radians(lat0)))
        row.append(d * 1.3 / 25 * 3600)
    return httpx.Response(200, json={"code": "Ok", "durations": [row]})


FAKE_NOW = datetime(2026, 10, 23, 7, 0)


def install(router: respx.Router):
    router.get(url__regex=r"https://nominatim\.openstreetmap\.org/.*").mock(side_effect=nominatim)
    router.get(url__regex=r"https://photon\.komoot\.io/.*").mock(return_value=httpx.Response(200, json={"features": []}))
    router.post(url__regex=r"https://(overpass|z\.overpass|maps\.mail\.ru).*").mock(side_effect=overpass)
    router.get(url__regex=r"https://api\.open-meteo\.com/v1/forecast.*").mock(side_effect=lambda r: _multi(r, _hourly))
    router.get(url__regex=r"https://air-quality-api\.open-meteo\.com/.*").mock(side_effect=lambda r: _multi(r, _aq))
    router.get(url__regex=r"https://router\.project-osrm\.org/.*").mock(side_effect=osrm)
