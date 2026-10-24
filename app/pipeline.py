"""End-to-end: question -> parse -> fetch live data -> score place x hour -> grounded answer."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta

from . import config
from .activities import ACTIVITIES, Activity
from .compose import build_fact_sheet, compose_answer, fmt_day, fmt_t
from .data import geocode as geo_mod
from .data import overpass, routing, weather
from .data.http import SOURCE_LOG
from .data.overpass import Candidate
from .parser import ParsedQuery, merge_slm, needs_slm, parse_rules
from .profiles import combine
from .scoring import Cell, avoid_ranges, build_sessions, contrast_reasons, score_cell, why_bullets
from .slm import SLM_MODEL

log = logging.getLogger("pipeline")

VENUE_KINDS = {"park", "garden", "beach", "nature_reserve", "golf_course", "pitch", "stadium", "sports_centre",
               "viewpoint", "peak", "fort", "leisure", "recreation_ground", "track", "picnic_site", "lake"}


def _supported_list() -> str:
    return ", ".join(a.name.lower() for a in ACTIVITIES.values())


def _base(q: ParsedQuery, status: str, answer: str, t0: float, **extra) -> dict:
    return {"status": status, "answer": answer, "query": q.summary(), "sources": _sources(),
            "model": SLM_MODEL.info, "timings": {"total_s": round(time.time() - t0, 2)}, **extra}


def _sources() -> list[dict]:
    seen, out = set(), []
    for e in SOURCE_LOG.get() or []:
        k = (e["source"], e["ok"])
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out


def _hours_for(q: ParsedQuery, activity: Activity) -> list[datetime]:
    hours = q.time.hours()
    if not q.time.explicit_part:
        a, b = activity.usual_hours
        keep = [h for h in hours if a <= h.hour < b] if a < b else [h for h in hours if h.hour >= a or h.hour < b]
        if keep:
            hours = keep
    return hours


async def answer(question: str, lat: float | None = None, lon: float | None = None, location: str | None = None,
                 now: datetime | None = None, use_slm: bool = True) -> dict:
    token = SOURCE_LOG.set([])
    try:
        return await _answer(question.strip(), lat, lon, location, now, use_slm)
    finally:
        SOURCE_LOG.reset(token)


async def _answer(question, lat, lon, location, now, use_slm) -> dict:
    t0 = time.time()
    now = (now or datetime.now(config.TZ)).astimezone(config.TZ)
    timings = {}
    q = parse_rules(question, now)
    slm_ok = use_slm and SLM_MODEL.backend != "none"
    if slm_ok and needs_slm(q):
        try:
            t1 = time.time()
            q = merge_slm(q, await asyncio.to_thread(SLM_MODEL.parse, question))
            timings["slm_parse_s"] = round(time.time() - t1, 2)
        except Exception as exc:  # noqa: BLE001
            log.warning("SLM parse failed: %s", exc)

    # ---- graceful handling of incomplete or out-of-scope questions ----
    if not question:
        return _base(q, "clarify", "Ask me about an outdoor plan, for example: \"Where can I go running tomorrow "
                                   "evening near Koregaon Park?\"", t0)
    if q.out_of_scope:
        return _base(q, "out_of_scope",
                     "I can only help with outdoor plans: where and when to go, what to carry and what to avoid. "
                     f"I support {_supported_list()}. Try something like \"Best time for a picnic near "
                     "Sanjay Gandhi National Park this weekend?\"", t0)
    if q.activity is None and q.generic_outdoor:
        q.activity = "walking"
        q.notes.append("No specific activity was named, so I assessed general time outdoors (walking pace).")
    if q.activity is None:
        return _base(q, "clarify", "Which outdoor activity are you planning? I can advise on "
                                   f"{_supported_list()}.", t0, missing=["activity"])
    activity = ACTIVITIES[q.activity]
    profile = combine(q.profiles)

    # ---- where ----
    place_text = q.location or (location.strip() if location else None)
    origin = None
    resolved = None
    if place_text:
        t1 = time.time()
        try:
            resolved = await geo_mod.geocode(place_text)
        except Exception as exc:  # noqa: BLE001
            log.warning("geocoding failed: %s", exc)
        timings["geocode_s"] = round(time.time() - t1, 2)
        if resolved is None:
            return _base(q, "clarify", f"I couldn't find \"{place_text}\" on OpenStreetMap. Could you add the city "
                                       "or a nearby landmark (e.g. \"Aundh, Pune\")?", t0, missing=["location"])
        origin = (resolved.lat, resolved.lon)
    elif lat is not None and lon is not None:
        origin = (float(lat), float(lon))
    else:
        what = "your commute" if q.commute else activity.name.lower()
        return _base(q, "clarify",
                     f"Which area should I check for {what}? Tell me a neighbourhood or city (for example "
                     "\"near Powai\" or \"in Pune\"), or send your coordinates (lat/lon) with the request.",
                     t0, missing=["location"])

    # ---- when ----
    if not q.time.intervals:
        note = " ".join(q.time.notes) or "I couldn't find forecast hours for that time."
        return _base(q, "no_data", note + " Ask about a time within the next few days and I'll check it live.", t0)
    hours = _hours_for(q, activity)

    # ---- radius ----
    radius = q.radius_km or activity.default_radius_km
    if not q.radius_km and resolved is not None and resolved.is_city_scale:
        radius = max(radius, min(resolved.extent_km, 12.0))
    radius = min(radius, config.MAX_RADIUS_KM)

    # ---- candidate places ----
    notes = list(q.notes) + list(q.time.notes)
    cands: list[Candidate] = []
    place_label = resolved.display_name if resolved else f"{origin[0]:.4f}, {origin[1]:.4f}"
    if q.commute:
        cands = [Candidate("node", 0, "your route (starting area)", True, origin[0], origin[1], "commute area",
                           1.0, 0.0, 0.0)]
        notes.append("Commute check: I assessed conditions where you start; I don't know your exact route.")
    else:
        t1 = time.time()
        try:
            found = await overpass.find_venues(activity, origin[0], origin[1], radius)
            if len(found) < 3 and radius < config.MAX_RADIUS_KM:
                wider = min(radius * 2.5, config.MAX_RADIUS_KM)
                found = await overpass.find_venues(activity, origin[0], origin[1], wider)
                if found:
                    notes.append(f"Few mapped venues within {radius:.0f} km, so I widened the search to {wider:.0f} km.")
                    radius = wider
        except Exception as exc:  # noqa: BLE001
            log.warning("venue search failed: %s", exc)
            found = []
            notes.append("OpenStreetMap venue search was unavailable, so I assessed the area itself.")
        timings["venues_s"] = round(time.time() - t1, 2)
        cands = overpass.shortlist(found, radius, activity, config.MAX_CANDIDATES)
        if resolved is not None and not resolved.is_city_scale and (
                resolved.kind in VENUE_KINDS or any(w in resolved.display_name.lower().split(",")[0]
                                                    for w in ("park", "beach", "garden", "maidan", "ground", "lake",
                                                              "fort", "drive", "sea face", "point", "hill"))):
            if not any(c.distance_km < 0.3 for c in cands):
                cands = [Candidate("node", 0, resolved.display_name.split(",")[0], True, resolved.lat, resolved.lon,
                                   "the place you named", 0.95, 0.0, 0.0)] + cands[:config.MAX_CANDIDATES - 1]
        if not cands:
            cands = [Candidate("node", 0, f"{place_text or 'your area'} (general area)", True, origin[0], origin[1],
                               "general area", 0.7, 0.0, 0.0)]
            notes.append(f"OpenStreetMap has no mapped {activity.name.lower()} venues within {radius:.0f} km, so this "
                         "is advice on timing for the area itself.")

    # ---- live data, fetched in parallel ----
    days_needed = max(1, (max(hours) .date() - now.date()).days + 2) if hours else 2
    pts = [(c.lat, c.lon) for c in cands]
    t1 = time.time()
    results = await asyncio.gather(
        overpass.add_context(cands, want_power="power_lines" in activity.tags),
        weather.forecast(pts, min(days_needed, 16)),
        weather.air_quality(pts, min(days_needed, config.AQ_FORECAST_DAYS)),
        routing.travel_minutes(origin, pts),
        return_exceptions=True,
    )
    timings["live_data_s"] = round(time.time() - t1, 2)
    ctx_res, wx_res, aq_res, route_res = results
    if isinstance(ctx_res, Exception):
        log.warning("context failed: %s", ctx_res)
        notes.append("Geospatial context (roads, industry, green cover) could not be fetched this time, so "
                     "those factors were left out rather than guessed.")
    if isinstance(wx_res, Exception) or not wx_res:
        log.warning("weather forecast unavailable: %s", wx_res)
        return _base(q, "no_data", "I couldn't reach the live weather service just now, and I won't guess the "
                                   "weather. Please try again in a minute.", t0)
    if isinstance(aq_res, Exception) or aq_res is None:
        aq_res = None
        notes.append("Air-quality forecast was unavailable, so air quality is not part of this recommendation.")
    if isinstance(route_res, Exception):
        route_res = ([None] * len(cands), "")
    travel, travel_src = route_res
    for c, wx, tm in zip(cands, wx_res, travel):
        c.travel_min = tm if c.distance_km > 0.3 else 0.0
        c.travel_source = travel_src
        c.elevation_m = wx.elevation
    if travel_src == "estimate":
        notes.append("Routing service was unavailable; travel times are distance-based estimates.")
    offline = next((e["source"] for e in SOURCE_LOG.get() or [] if e["source"].startswith("OpenStreetMap extract")), None)
    if offline:
        notes.append(f"The live Overpass API was unavailable, so places and surroundings come from the {offline}.")

    # ---- score every place x hour ----
    dur = activity.duration_h
    scored_hours = sorted({h + timedelta(hours=i) for h in hours for i in range(dur)})
    cells: dict[str, dict[datetime, Cell]] = {}
    aq_missing = False
    for i, c in enumerate(cands):
        by_t = {}
        for h in scored_hours:
            wx = wx_res[i].at(h)
            if not wx or wx.get("apparent_temperature") is None:
                continue
            aq = aq_res[i].at(h) if aq_res else None
            if aq_res and (not aq or aq.get("pm2_5") is None):
                aq_missing = True
                aq = None
            by_t[h] = score_cell(c, h, wx, aq, activity, profile, q.time.target, q.commute)
        cells[c.key] = by_t
    if aq_missing:
        notes.append(f"Air-quality forecasts reach about {config.AQ_FORECAST_DAYS} days ahead; later hours were scored "
                     "without air quality.")
    sessions = build_sessions(cells, hours, activity)
    if not sessions:
        return _base(q, "no_data", "The live forecast did not cover the hours you asked about.", t0)

    best = sessions[0]
    per_place = {}
    for s in sessions:
        per_place.setdefault(s.place.key, s)
    alternatives = [s for k, s in per_place.items() if k != best.place.key][:3]
    best_time_same_place = [s for s in sessions if s.place.key == best.place.key]
    day_best = {}
    for s in sessions:
        day_best.setdefault(s.start.date(), s)

    specific = None
    if q.time.specific_hour is not None:
        sp = [s for s in sessions if s.start == q.time.specific_hour]
        specific = max(sp, key=lambda s: s.score) if sp else None

    # ---- assemble ----
    sun = wx_res[cands.index(best.place)].sun_events()
    result = {
        "activity": activity, "profile": profile, "q": q, "origin": origin, "place_label": place_label,
        "radius_km": radius, "best": best, "alternatives": alternatives, "specific": specific,
        "same_place": best_time_same_place, "day_best": day_best, "hours": hours, "cells": cells,
        "cands": cands, "sun": sun, "notes": notes, "now": now, "per_place": per_place,
        "avoid": avoid_ranges(cells[best.place.key], hours, exclude={c.time for c in best.hours}),
        "why": _merge_why(contrast_reasons(best, cells[best.place.key], hours, alternatives), why_bullets(best)),
    }
    fact_sheet = build_fact_sheet(result)
    t1 = time.time()
    text, grounding, writer = await compose_answer(result, fact_sheet, use_slm=slm_ok)
    timings["compose_s"] = round(time.time() - t1, 2)
    timings["total_s"] = round(time.time() - t0, 2)
    return _to_json(result, text, fact_sheet, grounding, writer, timings)


def _merge_why(contrast: list[str], bullets: list[str]) -> list[str]:
    pos = [b for b in bullets if not b.startswith("caveat")]
    neg = [b for b in bullets if b.startswith("caveat")]
    return contrast[:2] + pos[:max(1, 4 - len(contrast[:2]))] + neg[:2]


def _session_json(s, why=None) -> dict:
    c0 = s.hours[0]
    return {
        "place": s.place.name, "kind": s.place.kind, "lat": round(s.place.lat, 5), "lon": round(s.place.lon, 5),
        "osm_url": s.place.osm_url if s.place.osm_id else None, "distance_km": round(s.place.distance_km, 1),
        "travel_min": c0.cond.get("travel_min"), "start": s.start.isoformat(), "end": s.end.isoformat(),
        "label": f"{fmt_day(s.start)}, {fmt_t(s.start)}–{fmt_t(s.end)}", "score": s.score,
        "blockers": s.blockers, "conditions": c0.cond, "why": why or [],
    }


def _context_json(place) -> dict:
    """Roads, industry, green cover and water around one place, as fetched for this answer."""
    ctx = place.context
    water = ctx.nearest_water() if ctx else None
    return {
        "roads_within_1km": [{"name": x.name, "class": x.highway, "distance_m": round(x.distance_km * 1000),
                              "direction": x.bearing} for x in (ctx.roads if ctx else [])],
        "industry_within_3km": [{"name": x.name, "distance_km": round(x.distance_km, 2)} for x in (ctx.industry if ctx else [])],
        "green_cover_500m_pct": round(ctx.green_fraction * 100) if ctx else None,
        "nearest_water": {"name": water.name, "distance_m": round(water.distance_km * 1000)} if water else None,
        "elevation_m": place.elevation_m,
    }


def _to_json(r, text, fact_sheet, grounding, writer, timings) -> dict:
    from .compose import prepare_list, watch_list
    q, best = r["q"], r["best"]
    hours = r["hours"]
    places = [c for c in r["cands"]]
    per_place = r["per_place"]
    heat = {
        "hours": [h.isoformat() for h in hours],
        "places": [{"name": c.name, "lat": c.lat, "lon": c.lon, "kind": c.kind,
                    "osm_url": c.osm_url if c.osm_id else None, "distance_km": round(c.distance_km, 1),
                    "context": _context_json(c),
                    "best": _session_json(per_place[c.key], why_bullets(per_place[c.key], 3)) if c.key in per_place else None}
                   for c in places],
        "scores": [[(r["cells"][c.key].get(h).score if r["cells"][c.key].get(h) else None) for h in hours]
                   for c in places],
    }
    status = "ok"
    verdict = None
    if r["specific"] is not None:
        sc = r["specific"].score
        verdict = "go" if sc >= 65 else ("go_with_care" if sc >= 45 else "not_advised")
    if best.score <= 20:
        verdict = verdict or "not_advised"
    return {
        "status": status,
        "answer": text,
        "verdict": verdict,
        "recommendation": _session_json(best, r["why"]),
        "requested_time": _session_json(r["specific"]) if r["specific"] else None,
        "alternatives": [_session_json(s, why_bullets(s, 3)) for s in r["alternatives"]],
        "by_day": [{"date": d.isoformat(), **_session_json(s)} for d, s in sorted(r["day_best"].items())],
        "avoid": [{"start": a["start"].isoformat(), "end": a["end"].isoformat(),
                   "label": f"{fmt_day(a['start'])}, {fmt_t(a['start'])}–{fmt_t(a['end'])}", "reason": a["reason"]}
                  for a in r["avoid"]],
        "prepare": prepare_list(r),
        "watch_for": watch_list(r),
        "geospatial_context": _context_json(best.place),
        "heatmap": heat,
        "query": q.summary(),
        "search": {"centre": r["place_label"], "lat": r["origin"][0], "lon": r["origin"][1],
                   "radius_km": round(r["radius_km"], 1), "places_scored": len(places),
                   "hours_scored": len(hours)},
        "notes": r["notes"],
        "fact_sheet": fact_sheet,
        "grounding": {**grounding, "writer": writer},
        "sources": _sources(),
        "model": SLM_MODEL.info,
        "timings": timings,
        "attribution": "Map data © OpenStreetMap contributors (ODbL), via the Overpass API or a Geofabrik extract. "
                       "Weather and air quality: Open-Meteo "
                       "(CC BY 4.0), air quality from Copernicus CAMS. Routing: OSRM.",
    }
