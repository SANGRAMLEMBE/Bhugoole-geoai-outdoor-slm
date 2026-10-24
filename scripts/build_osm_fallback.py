"""Build the offline OpenStreetMap layer used when the live Overpass API is unavailable.

    curl -L -o osm_build/raw/india-latest.osm.pbf https://download.geofabrik.de/asia/india-latest.osm.pbf
    python scripts/build_osm_fallback.py osm_build/raw/india-latest.osm.pbf

Keeps only what the assistant queries: activity venues, major roads (with geometry), industrial areas,
green cover, water and coastline (with geometry) and power lines, in a SQLite file with an R-tree index
(osm_fallback/osm_india.sqlite). Map data © OpenStreetMap contributors, ODbL.

Needs `pip install osmium` (pyosmium); takes ~15-30 min for all of India on a workstation.
"""
import argparse
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import osmium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402
from app.activities import ACTIVITIES  # noqa: E402
from app.data.osm_local import GREEN, INDUSTRY, POWER, ROAD, VENUE, WATER, Writer  # noqa: E402
from app.data.overpass import ROAD_WEIGHTS, _is_green, _is_industry, _is_water  # noqa: E402

KEYS = ("leisure", "natural", "highway", "landuse", "man_made", "power", "tourism", "historic", "boundary",
        "amenity", "golf", "waterway", "club")
KEEP = set(KEYS) | {"name", "name:en", "sport", "water", "lit", "ref"}
MIN_GREEN_KM2 = 0.003    # smaller lawns barely change green cover within 500 m
MIN_WATER_KM2 = 0.02     # unnamed ponds smaller than this are left out
THIN_KM = 0.02           # drop polyline vertices closer than 20 m to the previous kept one


def _compile(ql: str):
    prefix = ql.split("[", 1)[0]
    conds = []
    for m in re.finditer(r'\["([^"]+)"(?:(=|~)"([^"]*)")?\]', ql):
        k, op, v = m.group(1), m.group(2), m.group(3)
        conds.append((k, op, re.compile(v) if op == "~" else v))
    return prefix, conds


VENUE_FILTERS = [_compile(ql) for ql in sorted({v.ql for a in ACTIVITIES.values() for v in a.venues})]


def is_venue(osm_type: str, tags: dict) -> bool:
    for prefix, conds in VENUE_FILTERS:
        if prefix in ("node", "way", "relation") and prefix != osm_type:
            continue
        for k, op, v in conds:
            val = tags.get(k)
            if val is None or (op == "=" and val != v) or (op == "~" and not v.search(val)):
                break
        else:
            return True
    return False


def bbox_area_km2(s, w, n, e) -> float:
    lat = (s + n) / 2
    return (n - s) * 111.32 * (e - w) * 111.32 * math.cos(math.radians(lat)) * 0.6


def kinds_for(osm_type: str, tags: dict, area_km2: float | None) -> int:
    k = VENUE if is_venue(osm_type, tags) else 0
    if osm_type == "way" and tags.get("highway") in ROAD_WEIGHTS:
        k |= ROAD
    if _is_industry(tags):
        k |= INDUSTRY
    if osm_type != "node" and _is_green(tags) and (area_km2 is None or area_km2 >= MIN_GREEN_KM2 or k & VENUE):
        k |= GREEN
    if _is_water(tags):
        always = tags.get("natural") in ("coastline", "bay", "beach") or tags.get("waterway") in ("river", "riverbank")
        if always or "name" in tags or area_km2 is None or area_km2 >= MIN_WATER_KM2:
            k |= WATER
    if osm_type == "way" and tags.get("power") == "line":
        k |= POWER
    return k


def thin(coords):
    if len(coords) <= 2:
        return coords
    out = [coords[0]]
    cos = math.cos(math.radians(coords[0][0]))
    lim = (THIN_KM / 111.32) ** 2
    for la, lo in coords[1:-1]:
        pla, plo = out[-1]
        if (la - pla) ** 2 + ((lo - plo) * cos) ** 2 >= lim:
            out.append((la, lo))
    out.append(coords[-1])
    return out


def bounds_of(coords):
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    return min(lats), min(lons), max(lats), max(lons)


def keep_tags(tags) -> dict:
    return {t.k: t.v for t in tags if t.k in KEEP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pbf")
    ap.add_argument("--out", default=config.OSM_LOCAL_DB)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    if out.exists():
        out.unlink()
    header = osmium.io.Reader(a.pbf, osmium.osm.osm_entity_bits.NOTHING).header()
    stamp = header.get("osmosis_replication_timestamp") or datetime.fromtimestamp(
        Path(a.pbf).stat().st_mtime, timezone.utc).isoformat()

    # ---- pass 1: multipolygon / boundary relations we care about, and their member ways ----
    rels: dict[int, dict] = {}
    way_rels: dict[int, list[int]] = {}
    fp = osmium.FileProcessor(a.pbf, osmium.osm.RELATION).with_filter(osmium.filter.KeyFilter(*KEYS))
    for r in fp:
        if r.tags.get("type") not in ("multipolygon", "boundary"):
            continue
        tags = keep_tags(r.tags)
        k = kinds_for("relation", tags, None) & ~(ROAD | POWER)
        if not k:
            continue
        rels[r.id] = {"tags": tags, "kinds": k, "b": [90.0, 180.0, -90.0, -180.0], "parts": []}
        for m in r.members:
            if m.type == "w" and m.role in ("outer", ""):
                way_rels.setdefault(m.ref, []).append(r.id)
    print(f"pass 1: {len(rels):,} relations, {len(way_rels):,} member ways ({time.time() - t0:.0f}s)", flush=True)

    # ---- pass 2: nodes and ways with locations ----
    w = Writer(out)
    n_nodes = n_ways = seen_ways = 0
    fp = (osmium.FileProcessor(a.pbf, osmium.osm.NODE | osmium.osm.WAY).with_locations()
          .with_filter(osmium.filter.KeyFilter(*KEYS).enable_for(osmium.osm.NODE)))
    for o in fp:
        if o.is_node():
            tags = keep_tags(o.tags)
            k = kinds_for("node", tags, None)
            if k:
                la, lo = o.location.lat, o.location.lon
                w.add("node", o.id, k, tags, (la, lo, la, lo), (la, lo))
                n_nodes += 1
            continue
        seen_ways += 1
        if seen_ways % 5_000_000 == 0:
            print(f"  {seen_ways / 1e6:.0f}M ways read, {n_ways:,} kept ({time.time() - t0:.0f}s)", flush=True)
        member_of = way_rels.get(o.id)
        if not member_of and not any(key in o.tags for key in KEYS):
            continue
        coords = [(n.location.lat, n.location.lon) for n in o.nodes if n.location.valid()]
        if len(coords) < 2:
            continue
        b = bounds_of(coords)
        for rid in member_of or ():
            rel = rels[rid]
            rb = rel["b"]
            rel["b"] = [min(rb[0], b[0]), min(rb[1], b[1]), max(rb[2], b[2]), max(rb[3], b[3])]
            if rel["kinds"] & WATER:
                rel["parts"].append(thin(coords))
        tags = keep_tags(o.tags)
        k = kinds_for("way", tags, bbox_area_km2(*b)) if tags else 0
        if not k:
            continue
        parts = [thin(coords)] if k & (ROAD | WATER | POWER) else None
        w.add("way", o.id, k, tags, b, None, parts)
        n_ways += 1
    print(f"pass 2: {n_nodes:,} nodes and {n_ways:,} ways kept of {seen_ways / 1e6:.1f}M ways "
          f"({time.time() - t0:.0f}s)", flush=True)

    n_rels = 0
    for rid, rel in rels.items():
        s, wl, n, e = rel["b"]
        if s > n:  # no member geometry in the extract
            continue
        k = rel["kinds"]
        if k & GREEN and not k & VENUE and bbox_area_km2(s, wl, n, e) < MIN_GREEN_KM2:
            k &= ~GREEN
        if not k:
            continue
        w.add("relation", rid, k, rel["tags"], (s, wl, n, e), None, rel["parts"] if k & WATER else None)
        n_rels += 1
    w.set_meta("source", "Geofabrik extract " + Path(a.pbf).name + " (https://download.geofabrik.de/asia/india.html)")
    w.set_meta("data_timestamp", stamp)
    w.set_meta("data_date", stamp[:10])
    w.set_meta("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    w.set_meta("license", "Map data (c) OpenStreetMap contributors, ODbL 1.0")
    w.set_meta("counts", f"nodes={n_nodes} ways={n_ways} relations={n_rels}")
    w.close()
    print(f"done: {n_nodes + n_ways + n_rels:,} features -> {out} ({out.stat().st_size / 1e6:.0f} MB, "
          f"{(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
