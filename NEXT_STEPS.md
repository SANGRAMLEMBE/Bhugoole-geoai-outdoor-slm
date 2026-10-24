# Submission runbook (deadline: Sat 24 Oct 2026, 11:59 PM IST)

## Live links

* API: `POST https://sangramlembe9696--bhugoole-geoai-web.us-east.modal.run/ask` (demo UI at `/`, `/health`, `/docs`)
* Model: https://huggingface.co/WIZARDIAN/Bhugoole-geoai-outdoor-slm (QLoRA fine-tune, merged weights + adapter + GGUF)
* Code: https://github.com/SANGRAMLEMBE/Bhugoole-geoai-outdoor-slm

## Status

* Endpoint on Modal (4 CPU, scales to zero; cold start ~4 s + model load ~4 s). Redeploy: `modal deploy modal_app.py`.
* Offline OSM layer (`osm_fallback/osm_india.sqlite`, Geofabrik 2026-10-21) answers when Overpass is down; it is not in
  git (202 MB) but is shipped with the Modal image. Rebuild: `scripts/build_osm_fallback.py`.
* Fine-tuned model beats base on 30 real questions: exact slot match 76.7% -> 93.3%; 8/8 vs 7/8 answers verified.
* 50 offline tests pass; `scripts/smoke_live.py --url <endpoint>` runs 18 live questions.

## Remaining

1. Slides: cover (team name, members, links), slide 4 worked example from a live run, slide 6 latency; export PDF/PPTX.
2. Submit the Google Form: endpoint URL, HF model URL, GitHub URL, README, slides.
3. Demo questions: asthma walk near Powai, Sat-vs-Sun hike near Lonavala, "five a side football tmrw nite near Andheri West".
