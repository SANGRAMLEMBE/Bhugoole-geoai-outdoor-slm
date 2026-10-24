from datetime import datetime

import pytest

from app import config
from app.parser import parse_rules
from app.timeparse import parse_time

NOW = datetime(2026, 10, 23, 20, 15, tzinfo=config.TZ)  # a Friday evening

EXAMPLES = [
    ("Where can I go running tomorrow evening near Koregaon Park?", "running", "Koregaon Park", [], None),
    ("Best time this weekend for a kids' cricket match within 15 km of Whitefield?", "cricket", "Whitefield", ["child"], 15),
    ("I have mild asthma, where's good for a morning walk near Powai in the next two days?", "walking", "Powai", ["asthma"], None),
    ("Is it okay to cycle to work at 8 AM?", "cycling", None, [], None),
    ("Planning an outdoor birthday party on Sunday afternoon in Pune, what should I keep in mind?", "event", "Pune", [], None),
    ("Good place and time for golf on Saturday within 40 km of Andheri?", "golf", "Andheri", [], 40),
    ("Where can I take sunset photos this evening around Bandra?", "photography", "Bandra", [], None),
    ("Is Saturday or Sunday better for a hike near Lonavala?", "hiking", "Lonavala", [], None),
    ("When is it safe for my parents to walk outdoors tomorrow in Bengaluru?", "walking", "Bengaluru", ["senior"], None),
    ("Outdoor tennis after work today near Hinjewadi, any good spots?", "tennis", "Hinjewadi", [], None),
]


@pytest.mark.parametrize("q,act,loc,profiles,radius", EXAMPLES)
def test_examples_parse(q, act, loc, profiles, radius):
    p = parse_rules(q, NOW)
    assert p.activity == act
    assert p.location == loc
    assert sorted(p.profiles) == sorted(profiles)
    assert p.radius_km == radius


def test_messy_inputs():
    assert parse_rules("what's the capital of France", NOW).out_of_scope
    p = parse_rules("is it safe to go outside tomorrow in Delhi?", NOW)
    assert p.generic_outdoor and p.location == "Delhi"
    p = parse_rules("my 70 year old dad wants to jog at Marine Drive tomorrow morning", NOW)
    assert p.activity == "running" and p.location == "Marine Drive" and p.profiles == ["senior"]
    p = parse_rules("play cricket in the evening at Shivaji Park", NOW)
    assert p.location == "Shivaji Park"
    assert parse_rules("Is it okay to cycle to work at 8 AM?", NOW).commute


def test_time_windows():
    w = parse_time("tomorrow evening", NOW)
    assert w.intervals[0][0].day == 24 and w.intervals[0][0].hour == 16
    w = parse_time("this weekend", NOW)
    assert [d.day for d in w.days()] == [24, 25]
    w = parse_time("Saturday or Sunday", NOW)
    assert w.compare_days
    w = parse_time("at 8 AM", NOW)
    assert w.specific_hour.day == 24 and w.specific_hour.hour == 8 and w.notes
    w = parse_time("sunset photos this evening", NOW)       # already past -> tomorrow, with a note
    assert w.target == "sunset" and w.intervals[0][0].day == 24 and w.notes
    w = parse_time("morning walk in the next two days", NOW)
    assert {d.day for d in w.days()} == {24, 25}
    w = parse_time("between 6 and 8 am tomorrow", NOW)
    assert (w.intervals[0][0].hour, w.intervals[0][1].hour) == (6, 8)
    w = parse_time("picnic on 20 Nov", NOW)
    assert not w.intervals and "horizon" in w.notes[0]
    assert parse_time("go for a run", NOW).assumed
    w = parse_time("five a side football tmrw nite", NOW)    # slang spelling of night
    assert w.intervals[0][0].day == 24 and w.intervals[0][0].hour == 19
