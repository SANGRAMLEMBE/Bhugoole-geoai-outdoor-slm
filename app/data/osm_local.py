"""Offline OpenStreetMap layer, used when the live Overpass API is unavailable.

A compact SQLite copy of only the features the assistant queries (activity venues, major roads,
industry, green cover, water/coastline, power lines), built from a Geofabrik extract by
scripts/build_osm_fallback.py. Queries return elements in the same JSON shape as Overpass, so the
venue and context code in overpass.py is shared. Map data © OpenStreetMap contributors (ODbL).
"""
from __future__ import annotations

import array
import json
import logging
import re
import sqlite3
import threading
import zlib
from pathlib import Path

from .. import config
from ..geo import bbox_around, haversine_km
from .http import record_source

log = logging.getLogger("osm_local")

# Feature kinds (bit flags; a park can be both a venue and green cover)
VENUE, ROAD, INDUSTRY, GREEN, WATER, POWER = 1, 2, 4, 8, 16, 32

SCHEMA = """
CREATE TABLE IF NOT EXISTS feat(id INTEGER PRIMARY KEY, osm_type TEXT NOT NULL, osm_id INTEGER NOT NULL,
                                kinds INTEGER NOT NULL, tags TEXT NOT NULL, lat REAL, lon REAL, geom BLOB);
CREATE VIRTUAL TABLE IF NOT EXISTS feat_rtree USING rtree(id, minlat, maxlat, minlon, maxlon);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""

_SCALE = 1e5  # coordinates stored as 1e-5 degree integers (~1 m)


def tags_match(tags: dict, ql: str, osm_type: str | None = None) -> bool:
    """Evaluate a simple Overpass QL selector (type prefix and tag filters) against one element."""
    if osm_type is not None:
        prefix = ql.split("[", 1)[0]
        if prefix in ("node", "way", "relation") and prefix != osm_type:
            return False
    for m in re.finditer(r'\["([^"]+)"(?:(=|~)"([^"]*)")?\]', ql):
        k, op, v = m.group(1), m.group(2), m.group(3)
        if k not in tags:
            return False
        if op == "=" and tags[k] != v:
            return False
        if op == "~" and not re.search(v, tags[k]):
            return False
    return True


def encode_parts(parts: list[list[tuple[float, float]]]) -> bytes:
    """Polylines -> zlib-compressed delta-encoded int32s: [n_parts, len_1..len_n, dlat, dlon, ...]."""
    ints = array.array("i", [len(parts), *(len(p) for p in parts)])
    plat = plon = 0
    for part in parts:
        for la, lo in part:
            ila, ilo = round(la * _SCALE), round(lo * _SCALE)
            ints.extend((ila - plat, ilo - plon))
            plat, plon = ila, ilo
    return zlib.compress(ints.tobytes(), 6)


def decode_parts(blob: bytes) -> list[list[tuple[float, float]]]:
    ints = array.array("i")
    ints.frombytes(zlib.decompress(blob))
    n = ints[0]
    lens, pos = ints[1:1 + n], 1 + n
    parts, la, lo = [], 0, 0
    for ln in lens:
        part = []
        for _ in range(ln):
            la += ints[pos]
            lo += ints[pos + 1]
            pos += 2
            part.append((la / _SCALE, lo / _SCALE))
        parts.append(part)
    return parts


class Writer:
    """Append-only builder used by scripts/build_osm_fallback.py (and the tests)."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;" + SCHEMA)
        self.count = 0

    def add(self, osm_type: str, osm_id: int, kinds: int, tags: dict, bounds: tuple[float, float, float, float],
            center: tuple[float, float] | None = None, parts: list | None = None) -> None:
        s, w, n, e = bounds
        lat, lon = center or ((s + n) / 2, (w + e) / 2)
        cur = self.conn.execute(
            "INSERT INTO feat(osm_type, osm_id, kinds, tags, lat, lon, geom) VALUES (?,?,?,?,?,?,?)",
            (osm_type, osm_id, kinds, json.dumps(tags, ensure_ascii=False, separators=(",", ":")), lat, lon,
             encode_parts(parts) if parts else None))
        self.conn.execute("INSERT INTO feat_rtree VALUES (?,?,?,?,?)", (cur.lastrowid, s, n, w, e))
        self.count += 1
        if self.count % 100_000 == 0:
            self.conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    def close(self) -> None:
        self.conn.commit()
        self.conn.execute("VACUUM")
        self.conn.close()


class LocalOSM:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self.meta: dict[str, str] = {}

    @property
    def available(self) -> bool:
        if self._conn is None and self.path.is_file():
            try:
                self._conn = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True,
                                             check_same_thread=False)
                self.meta = dict(self._conn.execute("SELECT k, v FROM meta"))
            except sqlite3.Error as exc:
                log.error("offline OSM layer at %s is unreadable: %s", self.path, exc)
                self._conn = None
        return self._conn is not None

    @property
    def source_name(self) -> str:
        return (f"OpenStreetMap replica of India (rebuilt daily from Geofabrik), "
                f"data of {self.meta.get('data_date', 'unknown date')}")

    def _rows(self, lat: float, lon: float, radius_km: float, kinds: int):
        s, w, n, e = bbox_around(lat, lon, radius_km)
        with self._lock:
            return self._conn.execute(
                "SELECT f.osm_type, f.osm_id, f.tags, f.lat, f.lon, f.geom, r.minlat, r.maxlat, r.minlon, r.maxlon "
                "FROM feat_rtree r JOIN feat f ON f.id = r.id "
                "WHERE r.maxlat >= ? AND r.minlat <= ? AND r.maxlon >= ? AND r.minlon <= ? AND (f.kinds & ?) != 0",
                (s, n, w, e, kinds)).fetchall()

    @staticmethod
    def _element(row, with_geom: bool) -> dict:
        osm_type, osm_id, tags, lat, lon, geom, s, n, w, e = row
        el = {"type": osm_type, "id": osm_id, "tags": json.loads(tags)}
        if osm_type == "node":
            el["lat"], el["lon"] = lat, lon
        else:
            el["bounds"] = {"minlat": s, "minlon": w, "maxlat": n, "maxlon": e}
            el["center"] = {"lat": lat, "lon": lon}
        if with_geom and geom:
            parts = [[{"lat": la, "lon": lo} for la, lo in part] for part in decode_parts(geom)]
            if osm_type == "relation":
                el["members"] = [{"type": "way", "geometry": p} for p in parts]
            else:
                el["geometry"] = parts[0]
        return el

    def venues(self, venue_filters, lat: float, lon: float, radius_km: float, limit: int = 600) -> dict:
        """Elements matching any of the activity's venue filters, nearest first (like `out tags bb`)."""
        out = []
        for row in self._rows(lat, lon, radius_km, VENUE):
            el = self._element(row, with_geom=False)
            if any(tags_match(el["tags"], v.ql, el["type"]) for v in venue_filters):
                c = el.get("center") or el
                out.append((haversine_km(lat, lon, c["lat"], c["lon"]), el))
        out.sort(key=lambda x: x[0])
        record_source(self.source_name, str(self.path.name), cached=False)
        return {"elements": [el for _, el in out[:limit]]}

    def context(self, points: list[tuple[float, float]], want_power: bool) -> dict:
        """Roads, industry, green cover, water and power lines around each point (like the context query)."""
        seen: dict[tuple[str, int], dict] = {}
        layers = [(1.2, ROAD, True), (3.0, INDUSTRY, False), (0.6, GREEN, False), (4.0, WATER, True)]
        if want_power:
            layers.append((0.3, POWER, True))
        for lat, lon in points:
            for radius, kind, geom in layers:
                for row in self._rows(lat, lon, radius, kind):
                    key = (row[0], row[1])
                    if key not in seen or (geom and "geometry" not in seen[key] and "members" not in seen[key]):
                        seen[key] = self._element(row, with_geom=geom)
        record_source(self.source_name, str(self.path.name), cached=False)
        return {"elements": list(seen.values())}


LOCAL_OSM = LocalOSM(config.OSM_LOCAL_DB)
