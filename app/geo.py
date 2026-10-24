"""Small geodesy helpers (no heavy GIS dependencies needed at this scale)."""
from __future__ import annotations

import math

R_EARTH_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two compass angles (0..180)."""
    d = abs(a - b) % 360
    return 360 - d if d > 180 else d


def compass(deg: float) -> str:
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[int(((deg % 360) + 22.5) // 45) % 8]


def bbox_around(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """(south, west, north, east)"""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(0.1, math.cos(math.radians(lat))))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def _to_xy(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """Local equirectangular projection in km around (lat0, lon0)."""
    x = math.radians(lon - lon0) * R_EARTH_KM * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * R_EARTH_KM
    return x, y


def nearest_point_on_polyline(lat0: float, lon0: float, coords: list[tuple[float, float]]):
    """Distance (km) and bearing (deg) from (lat0, lon0) to the closest point of a polyline."""
    if not coords:
        return math.inf, 0.0
    best_d, best_xy = math.inf, (0.0, 0.0)
    pts = [_to_xy(lat0, lon0, la, lo) for la, lo in coords]
    if len(pts) == 1:
        x, y = pts[0]
        return math.hypot(x, y), (math.degrees(math.atan2(x, y)) + 360) % 360
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        dx, dy = x2 - x1, y2 - y1
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, -(x1 * dx + y1 * dy) / seg2))
        px, py = x1 + t * dx, y1 + t * dy
        d = math.hypot(px, py)
        if d < best_d:
            best_d, best_xy = d, (px, py)
    bx, by = best_xy
    return best_d, (math.degrees(math.atan2(bx, by)) + 360) % 360


def bbox_overlap_fraction(feature_bbox, circle_lat, circle_lon, radius_km) -> float:
    """Approximate fraction of a circle covered by a feature's bounding box.

    feature_bbox = (south, west, north, east). Bounding boxes over-estimate
    irregular polygons, so callers should discount the result.
    """
    s, w, n, e = feature_bbox
    cs, cw, cn, ce = bbox_around(circle_lat, circle_lon, radius_km)
    iw, ie = max(w, cw), min(e, ce)
    is_, in_ = max(s, cs), min(n, cn)
    if iw >= ie or is_ >= in_:
        return 0.0
    inter = (ie - iw) * (in_ - is_)
    circle_box = (ce - cw) * (cn - cs)
    # circle area / its bounding square = pi/4
    return min(1.0, inter / circle_box)
