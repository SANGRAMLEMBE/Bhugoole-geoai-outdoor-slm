"""Synthesise question -> JSON training pairs for the SLM's parsing job.

Labels are known by construction, so no LLM is involved. Messy variants (lower case, typos,
Hinglish fillers, missing places, out-of-scope requests) are deliberately over-sampled, because
the rule-based parser already handles clean questions and the SLM is only consulted for messy ones.

    python finetune/make_parse_dataset.py --n 3000 --out finetune/data
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os  # noqa: E402

os.environ.setdefault("SLM_BACKEND", "none")
from app.slm import PARSE_SHOTS, PARSE_SYSTEM  # noqa: E402

ACT = {
    "running": ["go for a run", "jog", "go jogging", "do my 5k", "running", "a quick run", "dauдne".replace("д", "d")],
    "cycling": ["cycle", "ride my bike", "go cycling", "bike to office", "cycling", "a cycle ride"],
    "golf": ["play golf", "a round of golf", "golf", "hit the driving range"],
    "cricket": ["play cricket", "a cricket match", "box cricket", "nets practice", "gully cricket"],
    "football": ["play football", "five-a-side", "a football game", "futsal", "soccer"],
    "tennis": ["play tennis", "a tennis match", "tennis practice", "hit some tennis balls"],
    "hiking": ["go on a hike", "trek", "a nature walk", "go trekking", "hike up a fort"],
    "walking": ["go for a walk", "a morning walk", "an evening stroll", "walk", "take a walk", "sair"],
    "picnic": ["have a picnic", "a family outing", "a park outing", "a day out with family"],
    "event": ["host a birthday party", "an outdoor wedding", "a get-together", "a haldi function", "a small fair"],
    "photography": ["take sunset photos", "click some pictures", "a photo walk", "shoot golden hour", "photography"],
    "yoga": ["do yoga", "outdoor yoga", "meditate outside", "surya namaskar in the park"],
    "birdwatching": ["go birdwatching", "watch birds", "birding"],
    "kite": ["fly kites", "patang udana", "kite flying"],
    "stargazing": ["go stargazing", "see the stars", "watch the meteor shower"],
}
PLACES = ["Powai", "Andheri West", "Bandra", "Juhu Beach", "Colaba", "Thane", "Vashi", "Borivali", "Malad",
          "Koregaon Park", "Baner", "Aundh", "Hinjewadi", "Kothrud", "Viman Nagar", "Lonavala", "Khandala",
          "Whitefield", "Indiranagar", "Koramangala", "HSR Layout", "Jayanagar", "Cubbon Park", "Nandi Hills",
          "Gachibowli", "Banjara Hills", "Hussain Sagar", "Salt Lake", "New Town", "Connaught Place", "Lodhi Garden",
          "Gurugram", "Noida Sector 18", "Dwarka", "Anna Nagar", "Besant Nagar", "Marina Beach", "Adyar",
          "Navrangpura", "Sabarmati Riverfront", "Kochi", "Fort Kochi", "Panaji", "Mysuru", "Chandigarh",
          "Sukhna Lake", "Bhopal", "Indore", "Jaipur", "Malviya Nagar", "Shivaji Park", "Marine Drive", "Aarey Colony"]
PROFILE = {
    "senior": ["my parents", "my 72 year old dad", "my grandmother", "a senior citizen", "my elderly mom"],
    "child": ["my kids", "two children", "my 6 year old", "school kids", "my toddler"],
    "asthma": ["I have asthma", "I'm asthmatic", "with my breathing problem", "I have mild asthma"],
    "pregnancy": ["I'm pregnant", "my pregnant wife", "being 6 months pregnant"],
    "heart": ["I have a heart condition", "my dad is a heart patient", "with high blood pressure"],
}
TIMES = ["tomorrow morning", "this evening", "on Sunday", "this weekend", "after work", "tmrw", "kal subah",
         "at 6 am", "in the next two days", "on Saturday afternoon", "tonight", "", "", "around sunset"]
OOS = ["book a train ticket to Delhi", "what's the capital of France", "write me a poem", "best gym near Powai",
       "recipe for paneer tikka", "is Nifty going up tomorrow", "movie timings in Andheri", "fix my python code",
       "which mall in Pune has a sale", "translate hello to Marathi"]
FILLERS = ["", "", "hey ", "bro ", "pls tell ", "yaar ", "quick q: ", "hi, "]


def typo(s, rng):
    if len(s) < 6 or rng.random() > 0.3:
        return s
    i = rng.randrange(1, len(s) - 2)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def make(rng):
    if rng.random() < 0.08:
        q = rng.choice(OOS)
        return q, {"activity": "out_of_scope", "location": "", "radius_km": 0, "profiles": []}
    act = rng.choice(list(ACT))
    phrase = rng.choice(ACT[act])
    place = rng.choice(PLACES) if rng.random() > 0.12 else ""
    prof = rng.choice(list(PROFILE)) if rng.random() < 0.35 else None
    radius = rng.choice([0, 0, 0, 5, 10, 15, 20, 40])
    when = rng.choice(TIMES)
    parts = []
    shape = rng.randrange(6)
    loc_txt = ""
    if place:
        loc_txt = (f"within {radius} km of @@" if radius else rng.choice(["near @@", "around @@",
                   "in @@", "@@", "close to @@", "@@ side"]))
    elif radius:
        radius = 0
    who = rng.choice(PROFILE[prof]) if prof else ""
    templates = [
        f"where can I {phrase} {when} {loc_txt}",
        f"{loc_txt} {phrase} {when}?",
        f"{who + ', ' if who else ''}is it ok to {phrase} {when} {loc_txt}",
        f"best time to {phrase} {loc_txt} {when} {('with ' + who) if who and not who.startswith('I') else ''}",
        f"{phrase} {loc_txt} {when} - good idea?",
        f"planning to {phrase} {when} {loc_txt}. {who + '.' if who else ''} any tips",
    ]
    q = rng.choice(FILLERS) + templates[shape]
    if who and who not in q:
        q = f"{who}, {q}" if who.startswith("I") else q + f" with {who}"
    q = " ".join(q.split())
    lower = rng.random() < 0.5
    if lower:
        q = q.lower()
    head, sep, tail = q.partition("@@")
    q = typo(head, rng) + sep + tail if len(head) > len(tail) else head + sep + typo(tail, rng)
    loc_label = ""
    if place:
        loc_label = place.lower() if lower else place
        q = q.replace("@@", loc_label)
    return q, {"activity": act, "location": loc_label, "radius_km": radius if loc_label else 0,
               "profiles": [prof] if prof else []}


def to_chat(q, label):
    msgs = [{"role": "system", "content": PARSE_SYSTEM}]
    for sq, sa in PARSE_SHOTS:
        msgs += [{"role": "user", "content": sq}, {"role": "assistant", "content": json.dumps(sa)}]
    return {"messages": msgs + [{"role": "user", "content": q},
                                {"role": "assistant", "content": json.dumps(label)}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="finetune/data")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    while len(rows) < a.n:
        q, lab = make(rng)
        if q in seen:
            continue
        seen.add(q)
        rows.append(to_chat(q, lab))
    n_eval = max(100, a.n // 10)
    for name, part in (("train.jsonl", rows[n_eval:]), ("eval.jsonl", rows[:n_eval])):
        with open(out / name, "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows) - n_eval} train and {n_eval} eval examples to {out}")


if __name__ == "__main__":
    main()
