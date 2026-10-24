"""Place x hour scoring.

Every (place, hour) cell starts at 100 and loses points for explicit, named reasons
(heat, rain, air quality, upwind traffic, travel time ...). Hard limits ("blockers")
such as thunderstorms, darkness on a trail or Severe air cap the score. Because
every point has a reason attached, the final answer can always say *why*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import aqi as aqimod
from .activities import Activity
from .astro import moon_illumination, sun_position
from .data.overpass import Candidate
from .data.routing import traffic_factor
from .data.weather import WMO, Series
from .geo import angle_diff, compass
from .profiles import Profile

BLOCKED_CAP = 15


@dataclass
class Factor:
    key: str
    points: float          # negative = penalty, positive = bonus
    text: str


@dataclass
class Cell:
    place: Candidate
    time: datetime
    score: float = 100.0
    factors: list[Factor] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    cond: dict = field(default_factory=dict)

    def add(self, key: str, points: float, text: str):
        if abs(points) >= 0.5:
            self.factors.append(Factor(key, round(points, 1), text))

    def block(self, text: str):
        if text not in self.blockers:
            self.blockers.append(text)

    def finalize(self):
        s = 100 + sum(f.points for f in self.factors)
        s = max(0.0, min(100.0, s))
        if self.blockers:
            s = min(s, BLOCKED_CAP)
        self.score = round(s, 1)


@dataclass
class Session:
    place: Candidate
    start: datetime
    hours: list[Cell]
    score: float
    blockers: list[str]

    @property
    def end(self) -> datetime:
        return self.start + timedelta(hours=len(self.hours))


def _ex(activity: Activity) -> float:
    """Exertion multiplier: harder activities feel heat and pollution more."""
    return 0.75 + 0.25 * activity.exertion


def score_cell(place: Candidate, t: datetime, wx: dict, aq: dict | None, activity: Activity, profile: Profile,
               target: str | None, commute: bool) -> Cell:
    c = Cell(place, t)
    ex = _ex(activity)
    feels = wx.get("apparent_temperature")
    temp = wx.get("temperature_2m")
    rain_p = wx.get("precipitation_probability") or 0
    rain_mm = wx.get("precipitation") or 0.0
    code = wx.get("weather_code")
    ws = wx.get("wind_speed_10m") or 0.0
    gust = wx.get("wind_gusts_10m") or ws
    wdir = wx.get("wind_direction_10m")
    uv = wx.get("uv_index") or 0.0
    cloud = wx.get("cloud_cover")
    vis = wx.get("visibility")
    el, az = sun_position(place.lat, place.lon, t)
    c.cond = {
        "time": t.isoformat(), "temp_c": temp, "feels_like_c": feels, "humidity_pct": wx.get("relative_humidity_2m"),
        "rain_prob_pct": rain_p, "rain_mm": rain_mm, "weather": WMO.get(code, "unknown") if code is not None else None,
        "weather_code": code, "wind_kmh": ws, "gust_kmh": gust, "wind_from_deg": wdir,
        "wind_from": compass(wdir) if wdir is not None else None, "uv_index": uv, "cloud_pct": cloud,
        "visibility_m": vis, "sun_elevation_deg": round(el, 1), "sun_azimuth_deg": round(az),
    }

    # ---- heat / cold ----
    lo, hi = activity.feels_ideal
    hi -= profile.heat_shift
    fmax = activity.feels_max - profile.heat_shift
    if feels is not None:
        if feels >= fmax:
            c.block(f"feels like {feels:.0f}°C, above the safe limit of {fmax:.0f}°C for this activity"
                    + (f" and {profile.label}" if profile.label else ""))
        if feels > hi:
            d = feels - hi
            c.add("heat", -min(50, (d * 2.0 + d * d * 0.2) * ex), f"feels like {feels:.0f}°C (comfortable up to {hi:.0f}°C)")
        elif feels < lo:
            c.add("cold", -min(30, (lo - feels) * 1.5), f"feels like {feels:.0f}°C (chilly)")

    # ---- rain & storms ----
    if code is not None and code >= 95:
        c.block(f"{WMO.get(code, 'thunderstorm')} forecast (lightning risk)")
    if rain_mm >= 5:
        c.block(f"heavy rain forecast ({rain_mm:.1f} mm in the hour)")
    if rain_p > activity.rain_tolerance:
        c.add("rain", -min(40, (rain_p - activity.rain_tolerance) * 0.55), f"{rain_p:.0f}% chance of rain")
    if rain_mm >= 0.3:
        c.add("rain_mm", -min(25, 6 + rain_mm * 4), f"{rain_mm:.1f} mm of rain expected")

    # ---- wind ----
    if activity.min_wind:
        if ws < activity.min_wind:
            c.add("wind_low", -min(40, (activity.min_wind - ws) * 4.5), f"only {ws:.0f} km/h wind (needs {activity.min_wind:.0f}+)")
        else:
            c.add("wind_good", 6, f"steady {ws:.0f} km/h wind")
    if ws > activity.wind_max:
        c.add("wind", -min(35, (ws - activity.wind_max) * 1.3), f"wind {ws:.0f} km/h")
        if ws > activity.wind_max + 22:
            c.block(f"strong wind ({ws:.0f} km/h)")
    if gust > activity.gust_max:
        c.add("gust", -min(20, (gust - activity.gust_max) * 0.8), f"gusts to {gust:.0f} km/h")

    # ---- UV ----
    if uv > 5 and activity.uv_sensitivity > 0:
        c.add("uv", -min(25, (uv - 5) * 2.5 * activity.uv_sensitivity * profile.uv_factor), f"UV index {uv:.0f}")

    # ---- visibility ----
    if vis is not None and activity.id in ("hiking", "photography", "cycling", "golf", "birdwatching"):
        if vis < 1000:
            c.add("fog", -22, f"visibility only {vis/1000:.1f} km (fog/haze)")
        elif vis < 4000:
            c.add("haze", -8, f"visibility {vis/1000:.1f} km")

    # ---- daylight, golden hour, night sky ----
    if activity.daylight == "required" and el < -0.8:
        if activity.lit_ok and place.lit:
            c.add("dark_lit", -8, "after dark (floodlit venue)")
        else:
            c.block("before sunrise" if t.hour < 12 else "after sunset")
    elif activity.daylight == "preferred" and el < -6:
        c.add("dark", -6 if place.lit else -15, "after dark" + (" (lit venue)" if place.lit else ", poor visibility"))
    elif activity.daylight == "night":
        if el > -12:
            c.block("sky not dark enough yet")
        illum = moon_illumination(t)
        c.add("moon", -illum * 30, f"moon {illum*100:.0f}% illuminated")
        if cloud is not None:
            if cloud > 60:
                c.block(f"{cloud:.0f}% cloud cover hides the sky")
            c.add("cloud_night", -cloud * 0.5, f"{cloud:.0f}% cloud cover")
    if activity.id == "photography":
        wants = target or "golden hour"
        if -4 <= el <= 8:
            golden_evening = az > 180
            if target is None or (target == "sunset" and golden_evening) or (target == "sunrise" and not golden_evening):
                c.add("golden", 20, "golden hour light")
                if target == "sunset" and place.context and place.context.water_toward(az):
                    c.add("view", 10, f"open water towards the sunset (sun bearing {az:.0f}°)")
        elif target in ("sunset", "sunrise"):
            c.add("not_golden", -25, f"outside the {wants} window")
        elif el > 45:
            c.add("harsh", -10, "harsh overhead light")
        if el < -6:
            c.add("dark_photo", -20, "too dark for most photography")
        if cloud is not None:
            if 20 <= cloud <= 60 and -4 <= el <= 10:
                c.add("cloud_color", 6, f"{cloud:.0f}% cloud to catch colour")
            elif cloud > 85:
                c.add("overcast", -12, f"{cloud:.0f}% cloud (flat light)")
    if activity.id == "birdwatching" and 0 <= el <= 35 and az < 180:
        c.add("bird_hours", 8, "early-morning bird activity")

    # ---- air quality (regional) ----
    aqi_val = dom = None
    if aq:
        aqi_val, dom = aqimod.naqi(aq)
    c.cond.update({"aqi_in": aqi_val, "aqi_category": aqimod.category(aqi_val), "aqi_dominant": dom,
                   "pm2_5": aq.get("pm2_5") if aq else None, "pm10": aq.get("pm10") if aq else None})
    if aqi_val is not None:
        ok = (80 if activity.exertion >= 2.5 else 100) * profile.aqi_factor
        bad = (200 if activity.exertion >= 2 else 300) * profile.aqi_factor
        if aqi_val > ok:
            c.add("aqi", -min(45, (aqi_val - ok) / max(1, bad - ok) * 38 * ex),
                  f"AQI {aqi_val} ({aqimod.category(aqi_val)}, mainly {dom})")
        if aqi_val >= bad:
            c.block(f"AQI {aqi_val} ({aqimod.category(aqi_val)}) is too high for this activity"
                    + (f" with {profile.label}" if profile.label else ""))
        if activity.id == "photography" and aqi_val > 200:
            c.add("haze_aqi", -8, "haze from particulates will flatten contrast")

    # ---- local exposure: wind direction x roads / industry (changes hour by hour) ----
    ctx = place.context
    tfac, tlabel = traffic_factor(t)
    c.cond["traffic"] = tlabel
    if ctx and wdir is not None:
        exp, worst = 0.0, None
        for r in ctx.roads:
            upwind = angle_diff(r.bearing, wdir) <= 50
            if ws < 4 and r.distance_km < 0.5:
                e = r.weight * (1 - r.distance_km / 0.5)            # calm air: fumes pool near the road
            elif upwind and ws >= 4:
                e = r.weight * (1 - r.distance_km / 1.0)            # wind carries fumes onto the venue
            elif r.distance_km < 0.15:
                e = r.weight * 0.3                                   # right next to the road anyway
            else:
                e = 0.0
            if e > 0 and (worst is None or e > worst[0]):
                worst = (e, r, upwind)
            exp += e
        if exp > 0:
            mult = 1.5 if tfac >= 1.5 else (1.15 if tfac > 1.1 else 0.8)
            pts = -min(20, exp * 10 * mult * profile.road_factor * (0.6 + 0.2 * activity.exertion))
            _, r, upwind = worst
            if ws < 4:
                why = f"calm air near {r.name} ({r.distance_km*1000:.0f} m away) lets traffic fumes linger"
            elif upwind:
                why = (f"downwind of {r.name}: wind from the {compass(wdir)} carries its traffic fumes here "
                       f"({r.distance_km*1000:.0f} m {compass(r.bearing)})")
            else:
                why = f"right beside {r.name}"
            if tfac >= 1.5:
                why += " during rush hour"
            c.add("road_exposure", pts, why)
            c.cond["road_exposure"] = {"road": r.name, "distance_m": round(r.distance_km * 1000),
                                       "direction": compass(r.bearing), "upwind": upwind}
        elif ctx.roads and ws >= 4:
            r = ctx.nearest_road()
            if r.distance_km < 0.6:
                c.add("road_clear", 3, f"wind from the {compass(wdir)} blows {r.name}'s fumes away from here")
                c.cond["road_exposure"] = {"road": r.name, "distance_m": round(r.distance_km * 1000),
                                           "direction": compass(r.bearing), "upwind": False}
        ind = [s for s in ctx.industry if ws >= 3 and angle_diff(s.bearing, wdir) <= 45]
        if ind:
            e = sum(1 - s.distance_km / 3.0 for s in ind)
            s0 = min(ind, key=lambda s: s.distance_km)
            c.add("industry", -min(14, 10 * e * profile.road_factor ** 0.5),
                  f"downwind of an industrial area ({s0.name}, {s0.distance_km:.1f} km {compass(s0.bearing)})")
        # ---- green cover and water ----
        if ctx.green_fraction > 0.05:
            bonus = ctx.green_fraction * 8 + (ctx.green_fraction * 4 if feels and feels > 30 else 0)
            c.add("green", bonus, f"about {ctx.green_fraction*100:.0f}% green cover within 500 m (shade, cooler air)")
        w = ctx.nearest_water()
        if w and w.distance_km < 0.7 and feels and feels > 29:
            c.add("water", 3, f"{w.name} {w.distance_km*1000:.0f} m away (breeze)")
        if activity.id == "kite" and ctx.power_line_km is not None and ctx.power_line_km < 0.2:
            c.add("power", -30, f"power line {ctx.power_line_km*1000:.0f} m away")

    # ---- travel time with time-of-day traffic ----
    if place.travel_min is not None and not commute:
        depart = t - timedelta(minutes=place.travel_min)
        f2, lbl = traffic_factor(depart)
        eff = place.travel_min * f2
        c.cond["travel_min"] = round(eff)
        c.cond["travel_note"] = lbl
        if eff > activity.free_travel_min:
            c.add("travel", -min(28, (eff - activity.free_travel_min) * 0.45 * profile.travel_factor),
                  f"about {eff:.0f} min to get there ({lbl})")
    if commute and tfac >= 1.5:
        c.add("commute_traffic", -6 * profile.road_factor, "peak traffic on the roads you will ride")

    # ---- venue suitability ----
    if place.quality < 1:
        c.add("venue", -(1 - place.quality) * 15, f"{place.kind} rather than a dedicated venue")
    if not place.named and place.osm_id:
        c.add("unnamed", -8, "unnamed on the map, so it may be private or hard to find")
    c.finalize()
    return c


def build_sessions(cells: dict[str, dict[datetime, Cell]], window_hours: list[datetime],
                   activity: Activity) -> list[Session]:
    """A session starting at h spans the activity's typical duration."""
    dur = max(1, activity.duration_h)
    out = []
    for key, by_t in cells.items():
        for h in window_hours:
            hs = [by_t.get(h + timedelta(hours=i)) for i in range(dur)]
            if any(x is None for x in hs):
                hs = [x for x in hs if x is not None]
                if not hs or hs[0].time != h:
                    continue
            scores = [x.score for x in hs]
            blockers = sorted({b for x in hs for b in x.blockers})
            s = 0.5 * (sum(scores) / len(scores)) + 0.5 * min(scores)
            if blockers:
                s = min(s, BLOCKED_CAP)
            out.append(Session(hs[0].place, h, hs, round(s, 1), blockers))
    out.sort(key=lambda s: s.score, reverse=True)
    return out


def avoid_ranges(cells_by_t: dict[datetime, Cell], hours: list[datetime], threshold: float = 40,
                 exclude: set[datetime] | None = None) -> list[dict]:
    """Group consecutive poor hours at one place, with the dominant reason."""
    out, cur = [], None
    exclude = exclude or set()
    for h in hours:
        if h in exclude:
            cur = None
            continue
        c = cells_by_t.get(h)
        bad = c is not None and (c.blockers or c.score < threshold)
        if bad:
            reason = c.blockers[0] if c.blockers else min(c.factors, key=lambda f: f.points).text
            if cur and cur["end"] == h and cur["reason_key"] == _reason_key(c):
                cur["end"] = h + timedelta(hours=1)
            else:
                cur = {"start": h, "end": h + timedelta(hours=1), "reason": reason, "reason_key": _reason_key(c)}
                out.append(cur)
        else:
            cur = None
    for r in out:
        r.pop("reason_key")
    return out


def _reason_key(c: Cell) -> str:
    if c.blockers:
        return c.blockers[0].split()[0]
    return min(c.factors, key=lambda f: f.points).key


def why_bullets(session: Session, max_items: int = 5) -> list[str]:
    """Most important reasons for a session, positives first, then the main caveat."""
    agg: dict[str, Factor] = {}
    for c in session.hours:
        for f in c.factors:
            if f.key not in agg or abs(f.points) > abs(agg[f.key].points):
                agg[f.key] = f
    pos = sorted([f for f in agg.values() if f.points > 0], key=lambda f: -f.points)
    neg = sorted([f for f in agg.values() if f.points < -3 and f.key != "venue"], key=lambda f: f.points)
    return [f.text for f in pos[:3]] + [f"caveat: {f.text}" for f in neg[:max(1, max_items - len(pos[:3]))]]


def _when(t: datetime, day: bool = True) -> str:
    h = t.hour % 12 or 12
    s = f"{h}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"
    return f"{t:%a} {s}" if day else s


def contrast_reasons(best: Session, by_t: dict[datetime, Cell], hours: list[datetime],
                     alternatives: list[Session]) -> list[str]:
    """Why this hour beats other hours, and why this place beats other places (place x time reasoning)."""
    out = []
    b0 = best.hours[0]
    others = [by_t[h] for h in hours if h in by_t and h not in {c.time for c in best.hours}]
    if others:
        f0 = b0.cond.get("feels_like_c")
        hot = max(others, key=lambda c: c.cond.get("feels_like_c") or -99)
        if f0 is not None and hot.cond.get("feels_like_c") is not None and hot.cond["feels_like_c"] - f0 >= 3:
            out.append(f"cooler than later or earlier in the window (feels like {f0:.0f}°C vs "
                       f"{hot.cond['feels_like_c']:.0f}°C at {_when(hot.time)})")
        a0 = b0.cond.get("aqi_in")
        dirty = max(others, key=lambda c: c.cond.get("aqi_in") or -1)
        if a0 is not None and dirty.cond.get("aqi_in") is not None and dirty.cond["aqi_in"] - a0 >= 20:
            out.append(f"cleaner air than at {_when(dirty.time)} (AQI {a0} vs {dirty.cond['aqi_in']})")
        wet = max(others, key=lambda c: c.cond.get("rain_prob_pct") or 0)
        r0 = b0.cond.get("rain_prob_pct") or 0
        if (wet.cond.get("rain_prob_pct") or 0) - r0 >= 30:
            out.append(f"drier than at {_when(wet.time)} (rain chance {r0:.0f}% vs {wet.cond['rain_prob_pct']:.0f}%)")
        exposed = [c for c in others if any(f.key == "road_exposure" and f.points <= -4 for f in c.factors)]
        if exposed and not any(f.key == "road_exposure" and f.points <= -4 for c in best.hours for f in c.factors):
            e = exposed[0]
            r = e.cond.get("road_exposure", {})
            if r.get("upwind"):
                out.append(f"the wind direction helps: at {_when(e.time)} the wind blows from the {e.cond['wind_from']}, "
                           f"carrying fumes from {r['road']} onto this spot, but at {_when(best.start, False)} it blows from the "
                           f"{b0.cond['wind_from']}")
    for alt in alternatives[:2]:
        a0 = alt.hours[0]
        if (a0.cond.get("travel_min") or 0) - (b0.cond.get("travel_min") or 0) >= 10:
            out.append(f"closer than {alt.place.name} ({b0.cond.get('travel_min')} vs {a0.cond.get('travel_min')} min)")
            break
    return out[:3]
