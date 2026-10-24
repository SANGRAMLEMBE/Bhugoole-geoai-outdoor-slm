"""The offline OpenStreetMap layer that answers when the live Overpass API fails."""
import httpx
import pytest

from app.data import osm_local, overpass
from app.data.osm_local import GREEN, INDUSTRY, ROAD, VENUE, WATER, LocalOSM, Writer, decode_parts, encode_parts
from app.pipeline import answer
from tests.fake_world import PLACES, _offset

POWAI = PLACES["powai"][:2]


def _box(lat, lon, half_km):
    dlat = half_km / 111.32
    return lat - dlat, lon - dlat, lat + dlat, lon + dlat


@pytest.fixture
def offline_db(tmp_path, monkeypatch):
    """A tiny extract around Powai: three parks, a track, an expressway, an industrial estate and a lake."""
    w = Writer(tmp_path / "osm.sqlite")
    lat, lon = POWAI
    for i, (name, d, b) in enumerate([("Lakeside Garden", 0.8, 70), ("Hill Park", 2.5, 200), ("Rose Garden", 1.6, 300)]):
        la, lo = _offset(lat, lon, d, b)
        w.add("way", 500 + i, VENUE | GREEN, {"leisure": "park", "name": name}, _box(la, lo, 0.15))
    la, lo = _offset(lat, lon, 1.1, 20)
    w.add("way", 510, VENUE, {"leisure": "track"}, _box(la, lo, 0.05))
    road = [_offset(lat, lon, 0.6, 90 + dy) for dy in (-40, -20, 0, 20, 40)]
    w.add("way", 520, ROAD, {"highway": "trunk", "name": "Eastern Express Highway"},
          (min(p[0] for p in road), min(p[1] for p in road), max(p[0] for p in road), max(p[1] for p in road)),
          None, [road])
    la, lo = _offset(lat, lon, 2.0, 100)
    w.add("way", 530, INDUSTRY, {"landuse": "industrial", "name": "MIDC Estate"}, _box(la, lo, 0.3))
    lake = [_offset(lat, lon, 1.0, a) for a in range(250, 300, 10)]
    w.add("relation", 540, WATER, {"natural": "water", "name": "Powai Lake"},
          (min(p[0] for p in lake), min(p[1] for p in lake), max(p[0] for p in lake), max(p[1] for p in lake)),
          None, [lake[:3], lake[2:]])
    w.set_meta("data_date", "2026-10-21")
    w.close()
    db = LocalOSM(tmp_path / "osm.sqlite")
    monkeypatch.setattr(overpass, "LOCAL_OSM", db)
    monkeypatch.setattr(overpass, "_down_until", 0.0)
    yield db
    overpass._down_until = 0.0


def test_geometry_round_trip():
    parts = [[(19.12345, 72.90001), (19.12, 72.91)], [(-1.5, 100.25)]]
    back = decode_parts(encode_parts(parts))
    assert [[(round(a, 5), round(b, 5)) for a, b in p] for p in back] == parts


def test_type_prefix_respected():
    assert osm_local.tags_match({"natural": "peak", "name": "X"}, 'node["natural"="peak"]["name"]', "node")
    assert not osm_local.tags_match({"natural": "peak", "name": "X"}, 'node["natural"="peak"]["name"]', "way")


async def test_answers_from_offline_layer_when_overpass_is_down(world, now, offline_db):
    world.post(url__regex=r"https://(overpass|z\.overpass|maps\.mail\.ru).*").mock(return_value=httpx.Response(504))
    res = await answer("Where can I go running tomorrow evening near Powai?", now=now, use_slm=False)
    assert res["status"] == "ok", res["answer"]
    assert res["search"]["places_scored"] >= 3                       # several places, not one "general area"
    assert {"Lakeside Garden", "Hill Park"} <= {p["name"] for p in res["heatmap"]["places"]}
    assert any(s["source"].startswith("OpenStreetMap replica") for s in res["sources"])
    assert any("Overpass servers were busy" in n for n in res["notes"])
    assert res["grounding"]["verified"]


async def test_offline_context_has_roads_industry_water(world, offline_db):
    world.post(url__regex=r"https://(overpass|z\.overpass|maps\.mail\.ru).*").mock(return_value=httpx.Response(504))
    lat, lon = _offset(*POWAI, 0.8, 70)
    park = overpass.Candidate("way", 500, "Lakeside Garden", True, lat, lon, "park", 0.8, 0.07, 0.8)
    await overpass.add_context([park])
    ctx = park.context
    assert ctx.roads and ctx.roads[0].name == "Eastern Express Highway" and ctx.roads[0].distance_km < 0.5
    assert ctx.industry and ctx.industry[0].name == "MIDC Estate"
    assert ctx.nearest_water().name == "Powai Lake"
    assert ctx.green_fraction > 0


async def test_live_overpass_still_preferred(world, now, offline_db):
    res = await answer("Where can I go running tomorrow evening near Powai?", now=now, use_slm=False)
    assert not any(s["source"].startswith("OpenStreetMap replica") for s in res["sources"])
