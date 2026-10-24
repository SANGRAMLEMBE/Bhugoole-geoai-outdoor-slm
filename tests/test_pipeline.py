import pytest

from app.activities import ACTIVITIES, REQUIRED_TEN
from app.pipeline import answer
from app.verifier import verify
from tests.test_understanding import EXAMPLES



@pytest.mark.parametrize("q", [e[0] for e in EXAMPLES])
async def test_examples_end_to_end(world, now, q):
    res = await answer(q, now=now, use_slm=False)
    if "cycle to work" in q:
        assert res["status"] == "clarify" and res["missing"] == ["location"]
        res = await answer(q, lat=19.1176, lon=72.906, now=now, use_slm=False)
        assert res["requested_time"] is not None and res["verdict"] in ("go", "go_with_care", "not_advised")
    assert res["status"] == "ok", res["answer"]
    assert res["grounding"]["verified"], res["grounding"]
    rec = res["recommendation"]
    assert rec["place"] and rec["start"] and 0 <= rec["score"] <= 100
    names = {p["name"] for p in res["heatmap"]["places"]}
    assert rec["place"] in names                         # never a place we did not fetch
    for p in res["heatmap"]["places"]:                   # every place carries its own surroundings and best slot
        assert "green_cover_500m_pct" in p["context"] and (p["best"] is None or p["best"]["place"] == p["name"])
    assert res["prepare"] and res["watch_for"] and res["sources"]
    assert any("Open-Meteo" in s["source"] for s in res["sources"])


@pytest.mark.parametrize("aid", list(ACTIVITIES))
async def test_every_activity(world, now, aid):
    word = {"running": "running", "cycling": "cycling", "golf": "golf", "cricket": "cricket practice",
            "football": "five-a-side football", "tennis": "tennis", "hiking": "hike", "walking": "evening walk",
            "picnic": "picnic", "event": "wedding", "photography": "photography", "yoga": "yoga",
            "birdwatching": "birdwatching", "kite": "kite flying", "stargazing": "stargazing"}[aid]
    when = "tonight" if aid == "stargazing" else "tomorrow"
    res = await answer(f"{word} {when} near Powai", now=now, use_slm=False)
    assert res["status"] == "ok", (aid, res["answer"])
    assert res["query"]["activity"] == aid
    assert res["grounding"]["verified"]


def test_ten_required_supported():
    assert set(REQUIRED_TEN) <= set(ACTIVITIES)


async def test_graceful_failures(world, now):
    assert (await answer("go running tomorrow", now=now, use_slm=False))["status"] == "clarify"
    assert (await answer("book a movie ticket", now=now, use_slm=False))["status"] == "out_of_scope"
    assert (await answer("picnic near Atlantis city", now=now, use_slm=False))["status"] == "clarify"
    assert (await answer("picnic near Powai on 30 Nov", now=now, use_slm=False))["status"] == "no_data"
    assert (await answer("", now=now, use_slm=False))["status"] == "clarify"


def test_wind_shift_changes_exposure(now):
    """The PS example: a ground downwind of an expressway at 8 AM can be pleasant by evening once the wind shifts."""
    from datetime import timedelta
    from app.activities import ACTIVITIES
    from app.data.overpass import Candidate, Context, RoadExposure
    from app.profiles import combine
    from app.scoring import score_cell
    ground = Candidate("way", 1, "Test Ground", True, 19.1, 72.9, "park", 0.8, 0.05, 1.0)
    ground.context = Context(roads=[RoadExposure("Western Express Highway", "trunk", 0.3, 90.0, 1.0)])
    base = {"apparent_temperature": 26, "temperature_2m": 25, "precipitation_probability": 0, "wind_speed_10m": 12,
            "wind_gusts_10m": 18, "uv_index": 0, "weather_code": 1}
    weekday = now + timedelta(days=(7 - now.weekday()) % 7)   # a Monday
    morning = score_cell(ground, weekday.replace(hour=8), {**base, "wind_direction_10m": 90}, None,
                         ACTIVITIES["running"], combine([]), None, False)
    evening = score_cell(ground, weekday.replace(hour=8), {**base, "wind_direction_10m": 270}, None,
                         ACTIVITIES["running"], combine([]), None, False)
    assert any(f.key == "road_exposure" and f.points < -5 for f in morning.factors)
    assert not any(f.key == "road_exposure" for f in evening.factors)
    assert evening.score > morning.score + 5
    asthma = score_cell(ground, weekday.replace(hour=8), {**base, "wind_direction_10m": 90}, None,
                        ACTIVITIES["running"], combine(["asthma"]), None, False)
    assert asthma.score < morning.score


def test_verifier_catches_inventions():
    sheet = "BEST OPTION: Hill Park, Sat 24 Oct 6:00 PM–7:00 PM, score 84/100\n conditions: feels like 33°C, AQI 85"
    assert verify("Go to Hill Park at 6:00 PM on Sat 24 Oct; it feels like 33°C and AQI is 85.", sheet).ok
    bad = verify("Go to Juhu Beach at 5:30 PM; it feels like 29°C.", sheet)
    assert not bad.ok and any("Juhu" in u for u in bad.unsupported) and any("29" in u for u in bad.unsupported)


def test_verifier_semantic_checks():
    sheet = ("BEST OPTION: Hill Park, Sat 24 Oct 8:00 AM–9:00 AM, score 86/100\n"
             "  conditions: feels like 24°C, AQI 55 Satisfactory (indicative, from modelled O3)")
    ok = "Head to Hill Park at 8:00 AM on Sat 24 Oct, when air quality is Satisfactory. Practical Tip: carry water."
    assert verify(ok, sheet).ok
    assert not verify("Go to Hill Park at 8:00 AM; the air quality is good then.", sheet).ok
    assert not verify("NOT ADVISED.\nBEST OPTION: Hill Park, score 86/100\ncaveat: AQI 55", sheet).ok
    road_sheet = "SEARCH AREA: Powai, Powai Ward, Mumbai Zone, Maharashtra\n  reason: away from Katraj - Dehu Road Bypass"
    assert verify("Near Powai, Powai Ward, Mumbai Zone, Maharashtra, stay away from Katraj Dehu Road Bypass's "
                  "traffic.", road_sheet).ok
    assert not verify("Near Powai, stay away from Eastern Express Highway.", road_sheet).ok
    blocked = ("OVERALL: NOT ADVISED in this window; every option hits a hard limit.\n"
               "BEST OPTION: Hill Park (park), Sun 25 Oct 5:00 AM–6:00 AM, score 15/100")
    assert not verify("Yoga at Hill Park at 5:00 AM on Sun 25 Oct is advised. Avoid the rain.", blocked).ok
    assert verify("Yoga at Hill Park is not advised on Sun 25 Oct; even 5:00 AM hits a hard limit.", blocked).ok
    assert not verify("Hill Park is a suitable choice, though yoga is not advised on Sun 25 Oct.", blocked).ok
    assert not verify("**Best Option: Hill Park**, Sun 25 Oct 5:00 AM", blocked).ok


def test_unnamed_venue_ranks_below_named_one(now):
    from app.activities import ACTIVITIES
    from app.data.overpass import Candidate, Context
    from app.profiles import combine
    from app.scoring import score_cell
    wx = {"apparent_temperature": 26, "temperature_2m": 25, "precipitation_probability": 0, "wind_speed_10m": 8,
          "wind_gusts_10m": 12, "uv_index": 2, "weather_code": 1, "wind_direction_10m": 90}
    t = now.replace(hour=8)
    named = Candidate("way", 1, "Greens Golf Club", True, 19.1, 72.9, "golf course", 1.0, 0.3, 10.0)
    unnamed = Candidate("way", 2, "Unnamed golf course", False, 19.1, 72.9, "golf course", 1.0, 0.3, 10.0)
    for c in (named, unnamed):
        c.context = Context()
    golf, nobody = ACTIVITIES["golf"], combine([])
    assert score_cell(named, t, wx, None, golf, nobody, None, False).score > \
        score_cell(unnamed, t, wx, None, golf, nobody, None, False).score


async def test_slm_path_with_fallback(world, now, monkeypatch):
    """A (fake) SLM that hallucinates is rejected twice, then the template answer is used."""
    from app import slm as slm_mod
    calls = []
    monkeypatch.setattr(slm_mod.SLM_MODEL, "backend", "fake")
    monkeypatch.setattr(slm_mod.SLM_MODEL, "compose",
                        lambda sheet, fb=None: calls.append(fb) or "Head to Imaginary Grounds at 5:45 PM when it is 21°C, "
                                                                 "a lovely cool evening with clean air and no traffic.")
    monkeypatch.setattr(slm_mod.SLM_MODEL, "parse", lambda q: {})
    res = await answer("running tomorrow evening near Powai", now=now, use_slm=True)
    assert len(calls) == 2 and calls[1]
    assert res["grounding"]["writer"].startswith("template")
    assert "Imaginary" not in res["answer"]


async def test_slm_path_accepts_grounded_text(world, now, monkeypatch):
    from app import slm as slm_mod
    monkeypatch.setattr(slm_mod.SLM_MODEL, "backend", "fake")
    monkeypatch.setattr(slm_mod.SLM_MODEL, "parse", lambda q: {})

    def grounded(sheet, fb=None):
        line = next(l for l in sheet.splitlines() if l.startswith("BEST OPTION"))
        place = line.split(": ", 1)[1].split(" (")[0]
        return (f"Your best choice is {place}. The fact sheet shows it has the best balance of heat, air and travel, "
                "so start then, carry water, and avoid the hottest hours of the day when the heat builds up.")
    monkeypatch.setattr(slm_mod.SLM_MODEL, "compose", grounded)
    res = await answer("running tomorrow evening near Powai", now=now, use_slm=True)
    assert res["grounding"]["writer"] == "slm" and res["grounding"]["verified"]
