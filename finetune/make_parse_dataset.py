"""Synthesise question -> JSON training pairs for the SLM's parsing job.

Labels are known by construction, so no LLM is involved. Messy variants (lower case, typos,
Hinglish fillers, missing places, out-of-scope requests) are deliberately over-sampled, because
the rule-based parser already handles clean questions and the SLM is only consulted for messy ones.

Version 2 (after error analysis of version 1 on real questions):
* out-of-scope requests come from templates (errands, shopping, travel bookings, general knowledge),
  often naming a place, so the model learns that a place in an off-topic request is not a location slot;
* places appear after many more connectors ("to X", "at X", "X ke paas", "I'm at X") and the place list
  covers forts, hills, lakes, beaches and neighbourhoods across India, typed in any case;
* ages ("I am 67", "my 8 yo") and more time and activity slang.
The hand-written test sets (finetune/real_eval.jsonl, finetune/test_unseen.jsonl) are never used here.

    python finetune/make_parse_dataset.py --n 6000 --out finetune/data
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
    "running": ["go for a run", "jog", "go jogging", "do my 5k", "running", "a quick run", "daudne", "a 10k run",
                "a morning run", "an evening jog", "run"],
    "cycling": ["cycle", "ride my bike", "go cycling", "bike to office", "cycling", "a cycle ride", "a bike ride",
                "go for a ride", "cycle to work", "a cycling trip"],
    "golf": ["play golf", "a round of golf", "golf", "hit the driving range", "tee off", "nine holes of golf"],
    "cricket": ["play cricket", "a cricket match", "box cricket", "nets practice", "gully cricket", "a cricket game",
                "cricket with friends", "a tennis-ball cricket match"],
    "football": ["play football", "five-a-side", "a football game", "futsal", "soccer", "a 5-a-side game",
                 "kick a ball around", "a football match"],
    "tennis": ["play tennis", "a tennis match", "tennis practice", "hit some tennis balls", "a game of tennis",
               "doubles tennis"],
    "hiking": ["go on a hike", "trek", "a nature walk", "go trekking", "hike up a fort", "a fort trek", "hiking",
               "a hike", "climb", "trek up"],
    "walking": ["go for a walk", "a morning walk", "an evening stroll", "walk", "take a walk", "sair", "walk the dog",
                "a brisk walk", "a stroll"],
    "picnic": ["have a picnic", "a family outing", "a park outing", "a day out with family", "picnic",
               "a family picnic", "an outing"],
    "event": ["host a birthday party", "an outdoor wedding", "a get-together", "a haldi function", "a small fair",
              "a wedding", "an engagement ceremony", "a mela", "a college fest", "a sangeet", "a reunion party"],
    "photography": ["take sunset photos", "click some pictures", "a photo walk", "shoot golden hour", "photography",
                    "a photoshoot", "sunrise photos", "a pre-wedding shoot"],
    "yoga": ["do yoga", "outdoor yoga", "meditate outside", "surya namaskar in the park", "yoga", "meditation"],
    "birdwatching": ["go birdwatching", "watch birds", "birding", "spot flamingos"],
    "kite": ["fly kites", "patang udana", "kite flying", "fly a kite"],
    "stargazing": ["go stargazing", "see the stars", "watch the meteor shower", "see the milky way"],
}
PLACES = [
    # metros: neighbourhoods and landmarks
    "Powai", "Andheri West", "Bandra", "Juhu Beach", "Colaba", "Thane", "Vashi", "Borivali", "Malad", "Chembur",
    "Vikhroli", "Kurla", "Dadar", "Matunga", "Sion", "Goregaon", "Dahisar", "Mira Road", "Nerul", "Belapur", "Panvel",
    "Airoli", "Shivaji Park", "Marine Drive", "Aarey Colony", "Koregaon Park", "Baner", "Aundh", "Hinjewadi",
    "Kothrud", "Viman Nagar", "Wakad", "Pimple Saudagar", "Kharadi", "Hadapsar", "Whitefield", "Indiranagar",
    "Koramangala", "HSR Layout", "Jayanagar", "Cubbon Park", "Malleshwaram", "Hebbal", "Yelahanka", "Bannerghatta",
    "Gachibowli", "Banjara Hills", "Jubilee Hills", "Kondapur", "Madhapur", "Secunderabad", "Hussain Sagar",
    "Salt Lake", "New Town", "Park Street", "Gariahat", "Behala", "Connaught Place", "Lodhi Garden", "Saket",
    "Vasant Kunj", "Rohini", "Janakpuri", "Karol Bagh", "Gurugram", "Sector 29 Gurugram", "Golf Course Road",
    "Noida Sector 18", "Dwarka", "Anna Nagar", "Besant Nagar", "Marina Beach", "Adyar", "Velachery", "T Nagar",
    "Mylapore", "OMR", "Navrangpura", "Sabarmati Riverfront", "Vastrapur", "Bopal", "Gomti Nagar", "Hazratganj",
    "Kakkanad", "Edappally", "Fort Kochi", "C Scheme", "Vaishali Nagar", "Malviya Nagar",
    # cities and towns
    "Kochi", "Panaji", "Mysuru", "Chandigarh", "Bhopal", "Indore", "Jaipur", "Udaipur", "Nagpur", "Surat", "Vadodara",
    "Lucknow", "Dehradun", "Shimla", "Manali", "Ooty", "Kodaikanal", "Munnar", "Coorg", "Alibaug", "Karjat",
    "Igatpuri", "Mahabaleshwar", "Panchgani", "Matheran", "Lonavala", "Khandala", "Yercaud",
    # forts, hills, lakes, beaches, parks
    "Rajmachi", "Harishchandragad", "Kalsubai", "Lohagad", "Visapur fort", "Torna fort", "Raigad fort",
    "Nandi Hills", "Skandagiri", "Savandurga", "Chamundi Hills", "Sukhna Lake", "Mulshi lake", "Pawna lake",
    "Khadakwasla", "Bhandardara", "Upper Lake", "Fateh Sagar", "Kashid beach", "Tarkarli", "Kovalam", "Calangute",
    "Baga beach", "Bandstand", "Carter Road", "Sion fort", "Rock Garden", "Parvati hill", "Chaturshringi hill",
    "Bandra promenade", "Rankala lake", "Banganga tank", "Juhu Chowpatty", "Girgaon Chowpatty", "Ramgarh lake",
    "Nainital lake", "Chilika lake", "Lake Pichola", "Assi ghat", "Kufri hills", "Dal lake",
]
PROFILE = {
    "senior": ["my parents", "my 72 year old dad", "my grandmother", "a senior citizen", "my elderly mom",
               "I am 67", "I'm 70", "my 65 yr old mom", "my grandpa", "my retired father", "aged 75"],
    "child": ["my kids", "two children", "my 6 year old", "school kids", "my toddler", "my 8 yo",
              "my 5 year old son", "the children", "a group of kids"],
    "asthma": ["I have asthma", "I'm asthmatic", "with my breathing problem", "I have mild asthma", "my son has asthma",
               "I get wheezy"],
    "pregnancy": ["I'm pregnant", "my pregnant wife", "being 6 months pregnant", "my wife is expecting"],
    "heart": ["I have a heart condition", "my dad is a heart patient", "with high blood pressure",
              "I had a cardiac issue"],
}
TIMES = ["tomorrow morning", "this evening", "on Sunday", "this weekend", "after work", "tmrw", "kal subah",
         "at 6 am", "in the next two days", "on Saturday afternoon", "tonight", "", "", "", "around sunset",
         "tmrw nite", "sat or sun", "day after tomorrow", "parso", "next weekend", "this sat", "sunday morning",
         "before office", "at 5 pm", "early morning", "aaj shaam"]
# a connector + "@@" marks where the place goes; the label is the place itself
LOC_FORMS = ["near @@", "around @@", "in @@", "@@", "close to @@", "@@ side", "at @@", "to @@", "towards @@",
             "@@ ke paas", "@@ mein", "somewhere near @@", "@@ area", "along @@", "on @@"]
FILLERS = ["", "", "", "hey ", "bro ", "pls tell ", "yaar ", "quick q: ", "hi, ", "ok so ", "need help: "]

# Off-topic requests, often naming a place: the place must not become a location slot
OOS_WITH_PLACE = ["book a cab to @@", "book a train ticket to @@", "flight tickets to @@", "order pizza to @@",
                  "best restaurant in @@", "good cafe near @@", "cheap hotels in @@", "pg accommodation in @@",
                  "movie timings in @@", "gym membership near @@", "hospital near @@", "salon near @@",
                  "coaching classes in @@", "flats for rent in @@", "which mall in @@ has a sale", "atm near @@",
                  "petrol pump near @@", "property prices in @@", "jobs in @@", "bus from @@ to Pune",
                  "best biryani in @@", "doctor near @@", "school admission in @@", "traffic police number @@"]
OOS_NO_PLACE = ["what's the capital of France", "write me a poem", "recipe for paneer tikka",
                "is Nifty going up tomorrow", "fix my python code", "translate hello to Marathi",
                "price of gold today", "who won the cricket match yesterday", "explain quantum physics",
                "tell me a joke", "convert 5 km to miles", "what is my horoscope", "latest news headlines",
                "how to lose weight fast", "best phone under 20000", "ipl schedule", "help me write an email",
                "what is 25 times 4", "who is the prime minister", "suggest a netflix series"]
# Weather questions with no activity: the place is a slot, the activity is unknown
UNKNOWN = ["will it rain @@ {when}", "how's the weather @@ {when}", "is the air clean @@ {when}",
           "what's the aqi @@ {when}"]


def typo(s, rng):
    if len(s) < 6 or rng.random() > 0.3:
        return s
    i = rng.randrange(1, len(s) - 2)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def finish(q, place, rng):
    """Normalise spacing, maybe lower-case and add a typo away from the place, then insert the place."""
    q = " ".join(q.split())
    lower = rng.random() < 0.5
    if lower:
        q = q.lower()
    head, sep, tail = q.partition("@@")
    q = typo(head, rng) + sep + tail if len(head) > len(tail) else head + sep + typo(tail, rng)
    label = ""
    if place:
        label = place.lower() if lower or rng.random() < 0.15 else place
        q = q.replace("@@", label)
    return q, label


def make(rng):
    r = rng.random()
    if r < 0.09:
        q, _ = finish(rng.choice(FILLERS) + rng.choice(OOS_WITH_PLACE), rng.choice(PLACES), rng)
        return q, {"activity": "out_of_scope", "location": "", "radius_km": 0, "profiles": []}
    if r < 0.13:
        q, _ = finish(rng.choice(FILLERS) + rng.choice(OOS_NO_PLACE), "", rng)
        return q, {"activity": "out_of_scope", "location": "", "radius_km": 0, "profiles": []}
    if r < 0.16:
        form = rng.choice(["in @@", "near @@", "at @@", "around @@"])
        q, label = finish(rng.choice(UNKNOWN).replace("@@", form).format(when=rng.choice(TIMES)), rng.choice(PLACES), rng)
        return q, {"activity": "unknown", "location": label, "radius_km": 0, "profiles": []}

    act = rng.choice(list(ACT))
    phrase = rng.choice(ACT[act])
    place = rng.choice(PLACES) if rng.random() > 0.12 else ""
    prof = rng.choice(list(PROFILE)) if rng.random() < 0.35 else None
    radius = rng.choice([0, 0, 0, 0, 5, 10, 15, 20, 40])
    when = rng.choice(TIMES)
    loc = ""
    if place:
        loc = (rng.choice([f"within {radius} km of @@", f"{radius} km from @@", f"@@, max {radius} km"]) if radius
               else rng.choice(LOC_FORMS))
    else:
        radius = 0
    who = rng.choice(PROFILE[prof]) if prof else ""
    templates = [
        f"where can I {phrase} {when} {loc}",
        f"{loc} {phrase} {when}?",
        f"{who + ', ' if who else ''}is it ok to {phrase} {when} {loc}",
        f"best time to {phrase} {loc} {when} {('with ' + who) if who and not who.startswith('I') else ''}",
        f"{phrase} {loc} {when} - good idea?",
        f"planning to {phrase} {when} {loc}. {who + '.' if who else ''} any tips",
        f"can we {phrase} {loc} {when}",
        f"{when} {phrase} {loc}, which spot?",
        f"is {when or 'today'} good for {phrase} {loc}",
        f"I'm at @@, can I {phrase} {when}" if place and not radius else f"want to {phrase} {loc} {when}",
    ]
    q = rng.choice(FILLERS) + rng.choice(templates)
    if who and who not in q:
        q = f"{who}, {q}" if who.startswith("I") else q + f" with {who}"
    q, label = finish(q, place, rng)
    return q, {"activity": act, "location": label, "radius_km": radius if label else 0,
               "profiles": [prof] if prof else []}


def to_chat(q, label):
    msgs = [{"role": "system", "content": PARSE_SYSTEM}]
    for sq, sa in PARSE_SHOTS:
        msgs += [{"role": "user", "content": sq}, {"role": "assistant", "content": json.dumps(sa)}]
    return {"messages": msgs + [{"role": "user", "content": q},
                                {"role": "assistant", "content": json.dumps(label)}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", default="finetune/data")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    held_out = set()
    for f in ("real_eval.jsonl", "test_unseen.jsonl"):
        p = Path(__file__).parent / f
        if p.exists():
            held_out |= {json.loads(line)["q"].lower() for line in p.read_text(encoding="utf-8").splitlines() if line}
    seen, rows = set(), []
    while len(rows) < a.n:
        q, lab = make(rng)
        if q in seen or q.lower() in held_out:
            continue
        seen.add(q)
        rows.append(to_chat(q, lab))
    n_eval = max(100, a.n // 20)
    for name, part in (("train.jsonl", rows[n_eval:]), ("eval.jsonl", rows[:n_eval])):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows) - n_eval} train and {n_eval} eval examples to {out}")


if __name__ == "__main__":
    main()
