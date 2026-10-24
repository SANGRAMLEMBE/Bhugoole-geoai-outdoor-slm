"""Turn scored results into a fact sheet, then into a grounded answer (SLM with verified fallback)."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from .verifier import verify

log = logging.getLogger("compose")


def fmt_t(t: datetime) -> str:
    h = t.hour % 12 or 12
    return f"{h}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"


def fmt_day(t) -> str:
    return f"{t:%a} {t.day} {t:%b}"


def _cond_line(c: dict) -> str:
    parts = []
    if c.get("feels_like_c") is not None:
        parts.append(f"feels like {c['feels_like_c']:.0f}°C (air {c['temp_c']:.0f}°C)")
    if c.get("weather"):
        parts.append(c["weather"])
    parts.append(f"rain chance {c.get('rain_prob_pct') or 0:.0f}%")
    if c.get("wind_from"):
        parts.append(f"wind from the {c['wind_from']} at {c['wind_kmh']:.0f} km/h")
    if c.get("uv_index") is not None and c["uv_index"] >= 1:
        parts.append(f"UV {c['uv_index']:.0f}")
    if c.get("aqi_in") is not None:
        parts.append(f"AQI {c['aqi_in']} {c['aqi_category']} (indicative, from modelled {c['aqi_dominant']})")
    return ", ".join(parts)


def _peak(session, key):
    vals = [c.cond.get(key) for c in session.hours if c.cond.get(key) is not None]
    return max(vals) if vals else None


def prepare_list(r) -> list[str]:
    a, p, s = r["activity"], r["profile"], r["best"]
    items = list(a.prepare) + list(p.prepare)
    feels, uv, rain = _peak(s, "feels_like_c"), _peak(s, "uv_index"), _peak(s, "rain_prob_pct")
    aqi = _peak(s, "aqi_in")
    if feels is not None and feels >= 30:
        items.append(f"Extra water and electrolytes (feels like up to {feels:.0f}°C)")
    if uv is not None and uv >= 6:
        items.append(f"Sunscreen SPF 30+, cap and sunglasses (UV up to {uv:.0f})")
    if rain is not None and rain >= 30:
        items.append(f"A light rain jacket or umbrella (rain chance up to {rain:.0f}%)")
    if aqi is not None and (aqi >= 150 or (aqi >= 100 and p.aqi_factor < 1)):
        items.append(f"A well-fitted N95 mask for the journey, and keep the effort light (AQI up to {aqi})")
    if any(c.cond.get("sun_elevation_deg", 10) < 0 for c in s.hours) and a.daylight != "night":
        items.append("Reflective clothing or a small light (part of this slot is after dark)")
    ctx = s.place.context
    if ctx and ctx.nearest_water() and ctx.nearest_water().distance_km < 0.5 and s.start.hour >= 16:
        items.append("Mosquito repellent (water nearby around dusk)")
    seen, out = set(), []
    for it in items:
        k = it.split("(")[0].strip().lower()[:30]
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out[:8]


def watch_list(r) -> list[str]:
    a, p, s = r["activity"], r["profile"], r["best"]
    items = []
    for c in s.hours:
        if c.cond.get("road_exposure") and c.cond["road_exposure"]["upwind"]:
            re_ = c.cond["road_exposure"]
            items.append(f"Traffic fumes from {re_['road']} ({re_['distance_m']} m {re_['direction']}) when the wind "
                         "blows from that side; use the far side of the venue")
            break
    later = [c for t, c in sorted(r["cells"][s.place.key].items()) if t >= s.end and t in set(r["hours"])]
    wet = [c for c in later if (c.cond.get("rain_prob_pct") or 0) >= 40]
    if wet:
        items.append(f"Rain chance rises to {wet[0].cond['rain_prob_pct']:.0f}% by {fmt_t(wet[0].time)} "
                     f"on {fmt_day(wet[0].time)}")
    hot = [c for c in later if (c.cond.get("feels_like_c") or 0) >= a.feels_max - p.heat_shift - 2]
    if hot:
        items.append(f"Heat builds: feels like {hot[0].cond['feels_like_c']:.0f}°C by {fmt_t(hot[0].time)}")
    gust = _peak(s, "gust_kmh")
    if gust and gust >= a.gust_max * 0.8:
        items.append(f"Gusts up to {gust:.0f} km/h")
    sun = r["sun"].get(s.start.strftime("%Y-%m-%d"))
    if sun and (a.daylight in ("required", "preferred") or a.id == "photography"):
        sr, ss = sun
        items.append(f"Sunrise {fmt_t(sr)}, sunset {fmt_t(ss)} on {fmt_day(s.start)}")
    items += list(a.watch) + list(p.watch)
    return items[:8]


def build_fact_sheet(r) -> str:
    a, p, q, best = r["activity"], r["profile"], r["q"], r["best"]
    L = [f"QUESTION: {q.question}",
         f"ACTIVITY: {a.name}" + (f"; PERSON: {p.label}" if p.label else ""),
         f"SEARCH AREA: {r['place_label']} (radius {r['radius_km']:.0f} km)",
         f"TIME ASKED: {q.time.label}"]
    for n in r["notes"]:
        L.append(f"NOTE: {n}")
    if r["specific"] is not None:
        sp = r["specific"]
        verdict = "GOOD" if sp.score >= 65 else ("OKAY WITH CARE" if sp.score >= 45 else "NOT ADVISED")
        L.append(f"REQUESTED TIME VERDICT: {verdict} (score {sp.score:.0f}/100) at {sp.place.name}, "
                 f"{fmt_day(sp.start)} {fmt_t(sp.start)}: {_cond_line(sp.hours[0].cond)}")
        if sp.blockers:
            L.append(f"  hard limits: {'; '.join(sp.blockers)}")
        neg = sorted({f.text for c in sp.hours for f in c.factors if f.points <= -5})
        if neg:
            L.append(f"  concerns: {'; '.join(neg[:4])}")
    if best.score <= 20:
        L.append("OVERALL: NOT ADVISED in this window; every option hits a hard limit.")
    L.append(f"BEST OPTION: {best.place.name} ({best.place.kind}"
             + (f", {best.place.distance_km:.1f} km from the search centre" if best.place.distance_km >= 0.3 else "")
             + (f", about {best.hours[0].cond['travel_min']} min by road" if best.hours[0].cond.get("travel_min") else "")
             + f"), {fmt_day(best.start)} {fmt_t(best.start)}–{fmt_t(best.end)}, score {best.score:.0f}/100")
    L.append(f"  conditions: {_cond_line(best.hours[0].cond)}")
    if best.blockers:
        L.append(f"  hard limits: {'; '.join(best.blockers)}")
    for w in r["why"]:
        L.append(f"  reason: {w}")
    if r["alternatives"]:
        L.append("ALTERNATIVES:")
        for s in r["alternatives"]:
            L.append(f"  - {s.place.name} ({s.place.kind}), {fmt_day(s.start)} {fmt_t(s.start)}–{fmt_t(s.end)}, "
                     f"score {s.score:.0f}/100: {_cond_line(s.hours[0].cond)}")
    if len(r["day_best"]) > 1:
        L.append("BEST BY DAY:")
        for d, s in sorted(r["day_best"].items()):
            L.append(f"  - {fmt_day(s.start)}: {s.place.name} at {fmt_t(s.start)}, score {s.score:.0f}/100"
                     + (f" (limits: {'; '.join(s.blockers)})" if s.blockers else ""))
    if r["avoid"]:
        L.append("AVOID:")
        for x in r["avoid"][:5]:
            L.append(f"  - {fmt_day(x['start'])} {fmt_t(x['start'])}–{fmt_t(x['end'])}: {x['reason']}")
    L.append("PREPARE: " + "; ".join(prepare_list(r)))
    L.append("WATCH FOR: " + "; ".join(watch_list(r)))
    return "\n".join(L)


def template_summary(r) -> str:
    a, best = r["activity"], r["best"]
    out = []
    if r["specific"] is not None:
        sp = r["specific"]
        verdict = ("Yes, good time" if sp.score >= 65 else "Okay, with care" if sp.score >= 45 else "Not advised")
        concerns = sp.blockers or sorted({f.text for c in sp.hours for f in c.factors if f.points <= -5})[:2]
        out.append(f"**{verdict}: {a.name.lower()} at {fmt_t(sp.start)} on {fmt_day(sp.start)}.** "
                   f"Conditions: {_cond_line(sp.hours[0].cond)}."
                   + (f" Main concern: {'; '.join(concerns)}." if concerns else ""))
        if best.score - sp.score >= 8:
            out.append(f"A better slot is {fmt_t(best.start)}–{fmt_t(best.end)} on {fmt_day(best.start)}"
                       + (f" at {best.place.name}" if best.place.key != sp.place.key else "")
                       + f" (score {best.score:.0f} vs {sp.score:.0f}).")
    elif best.score <= 20:
        out.append(f"**Not advised in this window.** Every option hits a hard limit: {'; '.join(best.blockers)}. "
                   f"The least-bad option is {best.place.name}, {fmt_day(best.start)} {fmt_t(best.start)}.")
    else:
        travel = best.hours[0].cond.get("travel_min")
        band = ("great" if best.score >= 82 else "good" if best.score >= 65 else "fair" if best.score >= 45
                else "poor")
        lead = "Best bet" if best.score >= 45 else "Least-bad option (conditions are poor throughout)"
        out.append(f"**{lead}: {best.place.name}** ({best.place.kind}"
                   + (f", about {travel} min away" if travel else "")
                   + f"), **{fmt_day(best.start)}, {fmt_t(best.start)}–{fmt_t(best.end)}** ({band} conditions, "
                   f"score {best.score:.0f}/100).")
        out.append(f"Expect {_cond_line(best.hours[0].cond)}.")
    pos = [w for w in r["why"] if not w.startswith("caveat")]
    neg = [w[len('caveat: '):] for w in r["why"] if w.startswith("caveat")]
    if pos:
        out.append("Why: " + "; ".join(pos) + ".")
    if neg and r["specific"] is None:
        out.append("Keep in mind: " + "; ".join(neg[:2]) + ".")
    if len(r["day_best"]) > 1 and r["q"].time.compare_days:
        days = sorted(r["day_best"].items(), key=lambda kv: -kv[1].score)
        (d1, s1), (d2, s2) = days[0], days[1]
        if s1.score - s2.score < 3:
            out.append(f"{fmt_day(s1.start)} and {fmt_day(s2.start)} look about the same (best slots score "
                       f"{s1.score:.0f} and {s2.score:.0f}), so pick whichever suits you; start early on either day.")
        else:
            worst = sorted({f.text for c in s2.hours for f in c.factors if f.points <= -5})
            out.append(f"{fmt_day(s1.start)} is the better day (best slot score {s1.score:.0f} vs {s2.score:.0f}"
                       + (f"; {fmt_day(s2.start)} is limited by {s2.blockers[0]}" if s2.blockers else
                          (f"; {fmt_day(s2.start)} has {worst[0]}" if worst else "")) + ").")
    if r["alternatives"] and r["specific"] is None:
        out.append("Alternatives: " + "; ".join(
            f"{s.place.name} ({fmt_day(s.start)} {fmt_t(s.start)}, score {s.score:.0f})" for s in r["alternatives"][:2]) + ".")
    return " ".join(out)


def _lists(r) -> str:
    parts = ["**Carry:** " + "; ".join(prepare_list(r)) + ".",
             "**Watch for:** " + "; ".join(watch_list(r)) + "."]
    if r["avoid"]:
        parts.append("**Avoid:** " + "; ".join(
            f"{fmt_day(x['start'])} {fmt_t(x['start'])}–{fmt_t(x['end'])} ({x['reason']})" for x in r["avoid"][:4]) + ".")
    if r["notes"]:
        parts.append("_Notes: " + " ".join(r["notes"]) + "_")
    return "\n\n".join(parts)


async def compose_answer(r, fact_sheet: str, use_slm: bool) -> tuple[str, dict, str]:
    from .slm import SLM_MODEL
    summary, writer, report = None, "template", None
    if use_slm and SLM_MODEL.backend != "none":
        feedback = None
        for _attempt in range(2):
            try:
                draft = await asyncio.to_thread(SLM_MODEL.compose, fact_sheet, feedback)
            except Exception as exc:  # noqa: BLE001
                log.warning("SLM compose failed: %s", exc)
                break
            report = verify(draft, fact_sheet)
            if report.ok and len(draft.split()) >= 25:
                summary, writer = draft, "slm"
                break
            feedback = ", ".join(report.unsupported[:6]) or "too short"
            log.info("SLM draft rejected by verifier: %s", feedback)
    if summary is None:
        summary = template_summary(r)
        rep2 = verify(summary, fact_sheet)
        grounding = {**rep2.as_dict(), "slm_draft_rejected": report.as_dict() if report else None}
        if writer == "template" and report is not None:
            writer = "template (SLM draft failed grounding check)"
    else:
        grounding = report.as_dict()
    return summary + "\n\n" + _lists(r), grounding, writer
