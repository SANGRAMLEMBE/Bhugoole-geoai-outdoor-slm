"""Score a GGUF model on the parsing job exactly as the live API runs it (llama.cpp + JSON-schema decoding).

    python finetune/eval_parse.py <model.gguf | hub> finetune/real_eval.jsonl finetune/test_unseen.jsonl

"hub" means the base model the API would download (Qwen2.5-1.5B-Instruct Q4_K_M). Prints exact-match and per-slot
accuracy for each test file, and every mistake.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    model, files = sys.argv[1], sys.argv[2:]
    if model == "hub":
        os.environ["SLM_GGUF_REPO"] = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
        os.environ["SLM_GGUF_FILE"] = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    else:
        os.environ["SLM_GGUF_PATH"] = model
    from app.slm import SLM_MODEL

    for f in files:
        rows = [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]
        hits = {"exact": 0, "activity": 0, "location": 0, "profiles": 0}
        for r in rows:
            p = SLM_MODEL.parse(r["q"])
            ok = {"activity": p.get("activity") == r["activity"],
                  "location": (p.get("location") or "").lower() == r["location"].lower(),
                  "profiles": sorted(p.get("profiles") or []) == sorted(r["profiles"])}
            ok["exact"] = all(ok.values())
            for k in hits:
                hits[k] += ok[k]
            if not ok["exact"]:
                print(f"  miss: {r['q'][:70]!r} -> {p.get('activity')!r}, {p.get('location')!r}, {p.get('profiles')}")
        print(f"{Path(f).name} ({len(rows)}): " + ", ".join(f"{k} {v / len(rows):.1%}" for k, v in hits.items()), flush=True)


if __name__ == "__main__":
    main()
