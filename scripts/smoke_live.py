"""Live smoke test against the REAL open APIs (run this on your machine before submitting).

    python scripts/smoke_live.py                       # in-process, template writer only (fast)
    python scripts/smoke_live.py --slm                 # in-process with the SLM (needs llama-cpp-python)
    python scripts/smoke_live.py --url https://<your-endpoint>                     # deployed endpoint
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUESTIONS = [
    "Where can I go running tomorrow evening near Koregaon Park?",
    "Best time this weekend for a kids' cricket match within 15 km of Whitefield?",
    "I have mild asthma, where's good for a morning walk near Powai in the next two days?",
    "Is it okay to cycle to work at 8 AM?",
    "Planning an outdoor birthday party on Sunday afternoon in Pune, what should I keep in mind?",
    "Good place and time for golf on Saturday within 40 km of Andheri?",
    "Where can I take sunset photos this evening around Bandra?",
    "Is Saturday or Sunday better for a hike near Lonavala?",
    "When is it safe for my parents to walk outdoors tomorrow in Bengaluru?",
    "Outdoor tennis after work today near Hinjewadi, any good spots?",
    # unseen / messy
    "five a side football tmrw nite near Andheri West?",
    "my wife is pregnant, picnic near Cubbon Park sunday?",
    "birdwatching near Thane creek this weekend",
    "kite flying at Juhu beach tomorrow",
    "yoga in the park near Indiranagar tomorrow early morning",
    "is it safe to go outside tomorrow in Delhi?",
    "football",
    "book me a flight to Goa",
]


def show(q, d, dt):
    rec = d.get("recommendation") or {}
    print(f"\n### {q}\n status={d.get('status')} verdict={d.get('verdict')} time={dt:.1f}s "
          f"writer={(d.get('grounding') or {}).get('writer')} verified={(d.get('grounding') or {}).get('verified')}")
    if rec:
        print(f" best: {rec.get('place')} | {rec.get('label')} | score {rec.get('score')} | "
              f"{(d.get('search') or {}).get('places_scored')} places x {(d.get('search') or {}).get('hours_scored')} h")
    print(" " + (d.get("answer") or "").split("\n\n")[0][:600])
    srcs = sorted({s["source"] + ("" if s["ok"] else " (FAILED)") for s in d.get("sources", [])})
    if srcs:
        print(" sources:", "; ".join(srcs))


async def local(use_slm):
    if not use_slm:
        os.environ["SLM_BACKEND"] = "none"
    from app.pipeline import answer
    for q in QUESTIONS:
        t = time.time()
        d = await answer(q, use_slm=use_slm)
        show(q, d, time.time() - t)


def remote(url):
    import httpx
    with httpx.Client(timeout=180) as c:
        print("health:", c.get(url.rstrip("/") + "/health").json())
        for q in QUESTIONS:
            t = time.time()
            r = c.post(url.rstrip("/") + "/ask", json={"question": q})
            show(q, r.json(), time.time() - t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--slm", action="store_true")
    a = ap.parse_args()
    remote(a.url) if a.url else asyncio.run(local(a.slm))
