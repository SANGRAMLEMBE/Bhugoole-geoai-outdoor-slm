# GeoAI Outdoor Assistant

**Where and when to head outdoors**, for GeoAI Hackathon 2026 (CSRE, IIT Bombay, PS-01).

Ask a free-form question such as *"I have mild asthma, where's good for a morning walk near Powai in the next two days?"*.
The assistant fetches live open data for that question, scores **every nearby place for every upcoming hour**, and a
small open language model (Qwen2.5-1.5B-Instruct, ≤ 3B) explains the best option, what to carry, what to watch for and
when to avoid going. Every number, time and place in the answer is checked against the data that was actually fetched.

| | |
|---|---|
| **Live API endpoint** | `POST https://sangramlembe9696--bhugoole-geoai-web.us-east.modal.run/ask` |
| **Demo UI** | <https://sangramlembe9696--bhugoole-geoai-web.us-east.modal.run/> (health: `/health`, OpenAPI: `/docs`) |
| **Code** | <https://github.com/SANGRAMLEMBE/Bhugoole-geoai-outdoor-slm> |
| **Model on Hugging Face** | [`WIZARDIAN/Bhugoole-geoai-outdoor-slm`](https://huggingface.co/WIZARDIAN/Bhugoole-geoai-outdoor-slm) (QLoRA fine-tune, merged weights + adapter + Q4_K_M GGUF) |
| **Base model** | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), 1.54B params, Apache-2.0, run as 4-bit GGUF with llama.cpp |

## How it works

```
question ─► 1. understand ─► 2. find places ─► 3. fetch live context ─► 4. score place × hour ─► 5. write + verify
            rules, then SLM     Nominatim          Overpass: roads, industry,   explicit, named          SLM writes from a
            JSON (schema-       + Overpass         green cover, water           penalties/bonuses per     fact sheet; verifier
            constrained) for    venues for the     Open-Meteo: hourly weather   cell; hard limits for     checks numbers, times
            gaps only           activity           + air quality (CAMS)         storms, darkness,         and names; template
                                                   OSRM: travel time            Severe AQI, heat          fallback if it fails
```

1. **Understand the question.** A rule-based parser extracts the activity (15 supported, 10 required), the place, radius,
   time window ("tomorrow evening", "this weekend", "after work", "Saturday or Sunday", "at 8 AM", "next two days",
   "sunset") resolved to exact hours in IST, and personal profiles (asthma, senior, child, pregnancy, heart condition).
   Only when the rules leave a gap is the SLM asked to fill it, with grammar-constrained JSON. Any place the SLM
   returns must appear verbatim in the question, so the model cannot invent a location. Missing place, unknown place,
   out-of-scope request or a time beyond the forecast horizon → a clear clarifying answer instead of a guess.
2. **Find candidate places.** Geocode with Nominatim (Photon fallback), then query OpenStreetMap through Overpass for
   venues that suit the activity (cricket grounds and maidans, golf courses, tennis courts, forts and peaks for hikes,
   viewpoints and piers for photos, events venues for gatherings …). Duplicates are merged and a static prior picks the
   ~10 most promising places to score in depth.
3. **Fetch live context, in parallel, per question.**
   * OSM within each venue: major roads (with geometry), industrial zones, green cover within 500 m, water and coastline,
     power lines (for kites).
   * Open-Meteo hourly forecast for every venue in a single request (feels-like temperature, rain probability and amount,
     thunderstorm codes, wind speed, gusts and **direction**, UV, cloud, visibility, sunrise/sunset).
   * Open-Meteo air-quality forecast (CAMS): PM2.5, PM10, NO₂, O₃ → India's **CPCB National AQI** computed per hour.
   * OSRM travel time from the search centre, scaled by a disclosed rush-hour factor.
4. **Score place × hour.** Each cell starts at 100 and loses or gains points for named reasons. The key geospatial
   reasoning is hour-specific: if a major road or industrial area lies **upwind** of a venue at that hour, the venue is
   penalised, and the penalty disappears when the wind turns. This is the problem statement's own example: a ground
   downwind of an expressway at 8 AM can be pleasant by evening once the sea breeze sets in. Calm air near roads,
   rush-hour traffic, green cover, sea/lake breeze, travel time and venue suitability are also scored. Activities
   last for their typical duration (golf 4 h, cricket 3 h, run 1 h), so a session is scored over all its hours.
5. **Write and verify.** A fact sheet (best option, alternatives, best-per-day, avoid windows, what to carry and watch for,
   all from fetched data) is given to the SLM, which writes an 80–150 word answer. The **grounding verifier** checks
   every number, clock time and proper noun against the fact sheet; a failed draft is regenerated once with feedback,
   then falls back to a deterministic template. The response says which writer was used and what was checked.

### Activities and profiles

Required ten: running/jogging, cycling (leisure or commute), golf, cricket, football/five-a-side, tennis, hiking/nature
walk, picnic/park outing, outdoor event/gathering, photography. Extra: walking, outdoor yoga, birdwatching, kite flying,
stargazing. Each activity has its own comfort range, heat limit, rain and wind tolerance, daylight rule (required,
preferred, night), duration, typical hours, travel tolerance and OSM venue tags (`app/activities.py`). Personal profiles
tighten the limits (`app/profiles.py`): asthma lowers AQI thresholds by 40% and doubles the weight of traffic exposure;
seniors and pregnancy lower the heat limit by 4–5 °C and penalise long travel, and so on.

### Scoring factors (per place, per hour)

| Factor | Source | Effect |
|---|---|---|
| Feels-like temperature | Open-Meteo | Penalty grows quadratically above the activity's comfort limit; blocker above its safe limit |
| Thunderstorm, heavy rain | Open-Meteo (WMO codes, mm/h) | Blocker |
| Rain probability | Open-Meteo | Penalty above the activity's tolerance |
| Wind and gusts | Open-Meteo | Penalty above limits (tennis/golf strict); kites need ≥ 9 km/h |
| UV index | Open-Meteo | Penalty above 5, stronger for children and seniors |
| Daylight, golden hour, night sky | Computed sun position, moon phase | Darkness blocks trails and golf; golden hour bonus for photos; open water toward the sunset bonus |
| Air quality | Open-Meteo / CAMS → CPCB AQI | Penalty above Satisfactory (stricter for exertion and sensitive groups); blocker at Poor/Very Poor |
| Upwind roads and industry | OSM geometry × wind direction | Penalty when a motorway/trunk/primary road or industrial area is upwind; calm-air pooling; rush-hour multiplier |
| Green cover, water | OSM polygons | Small bonus (shade, cooler, cleaner air; breeze in heat) |
| Travel time | OSRM × rush-hour factor | Penalty beyond the activity's tolerance; heavier for seniors and children |
| Venue suitability | OSM tags | Dedicated venue preferred over a generic park |

## API

```bash
curl -s https://<endpoint>/ask -H 'Content-Type: application/json' \
  -d '{"question": "Where can I go running tomorrow evening near Koregaon Park?"}'
```

Limits: up to 60 questions per minute per client address (HTTP 429 with `Retry-After` beyond that), questions up to
500 characters, request bodies up to 16 KB; `lat`/`lon` are range-checked. Errors come back as JSON, never a stack
trace. The live endpoint scales out to 4 containers; in our check, 6 simultaneous full answers took 14 s median
and 18 s at most, and fast-mode answers about 3 s (`scripts/load_check.py`).

Request fields: `question` (also accepted as `query`, `q`, `prompt`, `message`, or a chat-style `messages` list),
optional `lat`/`lon` or `location` for questions without a place ("Is it okay to cycle to work at 8 AM?"), `fast: true`
to skip the SLM, `format: "text"` for a plain-text reply, `now` (ISO time) for reproducible tests.
`GET /ask?q=...` works too. Other routes: `GET /health`, `GET /activities`, `GET /docs` (OpenAPI), `GET /` (demo UI).

Response (abridged):

```json
{
  "status": "ok",                       // ok | clarify | out_of_scope | no_data
  "answer": "**Best bet: ...** ...",    // the user-facing answer (markdown)
  "verdict": "go_with_care",            // for yes/no questions about a specific time
  "recommendation": {"place": "...", "lat": 0, "lon": 0, "label": "Sat 24 Oct, 6:00 AM–7:00 AM", "score": 81,
                     "conditions": {...}, "why": [...]},
  "alternatives": [...], "by_day": [...], "avoid": [...], "prepare": [...], "watch_for": [...],
  "geospatial_context": {"roads_within_1km": [...], "green_cover_500m_pct": 34, ...},
  "heatmap": {"places": [...], "hours": [...], "scores": [[...]]},
  "fact_sheet": "...", "grounding": {"verified": true, "writer": "slm", ...},
  "sources": [{"source": "Open-Meteo Forecast API", "cached": false, "ok": true}, ...],
  "timings": {...}, "model": {...}
}
```

## Run it

```bash
pip install -r requirements.txt
pip install "llama-cpp-python>=0.3.2" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
uvicorn app.main:app --port 7860          # downloads the 1 GB GGUF on first start; UI at http://localhost:7860
SLM_BACKEND=none uvicorn app.main:app     # template writer only, no model needed
python scripts/build_osm_fallback.py osm_build/raw/india-latest.osm.pbf   # offline OSM layer (see osm_fallback/)
python scripts/smoke_live.py              # 18 live questions against the real APIs
pip install -r requirements-dev.txt && pytest   # 44 offline tests with a mocked world (no network needed)
```

Configuration is by environment variable (`app/config.py`): `SLM_BACKEND` (`llamacpp`, `transformers`, `none`),
`SLM_GGUF_REPO`/`SLM_GGUF_FILE`, `SLM_HF_MODEL` and `SLM_ADAPTER` for the transformers backend, `OVERPASS_URLS`, etc.

**Deploy** (the live endpoint runs on [Modal](https://modal.com): serverless, scales to zero, 4 CPU cores):
`pip install modal && modal token new && modal deploy modal_app.py`. Alternatives: any Docker host
(`docker build -t bhugoole . && docker run -p 7860:7860 bhugoole`), Google Cloud Run
(`scripts/deploy_cloudrun.py`), or Hugging Face Docker Spaces (`scripts/deploy_space.py`, now a paid plan).
**Model repo**: `python scripts/push_model.py --repo <user>/<name> --adapter finetune/out/adapter --merged finetune/out/merged --gguf <q4_k_m.gguf>`.

## Model

**Bhugoole-geoai-outdoor-slm**: a QLoRA fine-tune of `Qwen/Qwen2.5-1.5B-Instruct` (1.54B parameters, Apache-2.0),
quantised to Q4_K_M GGUF and run with llama.cpp on CPU. It is used for two narrow, checkable jobs: slot extraction
(JSON-schema-constrained decoding, temperature 0) and answer writing from the fact sheet (temperature 0.3, verified).
The model never sees the internet and is never asked to recall weather or places. Fine-tuning (`finetune/`): 2,700
synthetic question → JSON pairs, r=16, 1 epoch. On 30 hand-written real questions (never trained on), through llama.cpp,
exact slot match rose from 76.7% to 93.3%, and 8/8 answers passed the grounding verifier (base 7/8); see
`finetune/TRAINING_REPORT.md`.

## Data sources (all free and open, fetched live per question)

| Source | Used for | Licence / terms | Handling |
|---|---|---|---|
| OpenStreetMap via [Overpass API](https://overpass-api.de) (+ maps.mail.ru, kumi.systems, z.overpass-api.de and private.coffee mirrors) | Venues, roads, industry, green cover, water, power lines | ODbL, © OpenStreetMap contributors | One venue query and one batched context query per question; 24 h cache; hedged mirror failover with an 8 s budget (then skipped for 10 min), then the offline layer below |
| [Geofabrik India extract](https://download.geofabrik.de/asia/india.html) (OSM) | Offline copy of the same venue and context features, used **only when Overpass fails** | ODbL, © OpenStreetMap contributors | Built by `scripts/build_osm_fallback.py` into `osm_fallback/osm_india.sqlite` (R-tree index); answers that use it say so and give its data date |
| [Nominatim](https://nominatim.org) (OSM) | Geocoding place names | ODbL; 1 request/s policy | Throttled to < 1/s, identifying User-Agent, 7-day cache |
| [Photon](https://photon.komoot.io) (OSM) | Geocoding fallback | ODbL | Only if Nominatim fails |
| [Open-Meteo Forecast API](https://open-meteo.com) | Hourly weather, sunrise/sunset, elevation | CC BY 4.0 | All venues in one request; 15 min cache |
| [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) (Copernicus CAMS) | PM2.5, PM10, NO₂, O₃ | CC BY 4.0 / Copernicus | All venues in one request; 15 min cache |
| [OSRM demo server](https://project-osrm.org) | Driving time to each venue | BSD (software), ODbL (data) | One table request per question; distance estimate if unavailable |
| CPCB National AQI breakpoints | Converting concentrations to India's AQI | Public (CPCB) | Applied to hourly values, labelled *indicative* |

No paid or private APIs, no API keys, no scraping. Map data © OpenStreetMap contributors.

## Limitations (honest notes)

* **Air quality is modelled, not measured.** CAMS has ~40 km resolution, so nearby venues share regional AQI; the
  local difference comes from our wind-aware road/industry exposure model, which is a heuristic, not a dispersion model.
  CPCB defines AQI on 24 h averages; we apply its breakpoints hourly and say so. CAMS surface ozone is
  strongly over-forecast over Indian cities (we saw 460 µg/m³ forecast for Mumbai, which would make every afternoon
  "Very Poor"), so by default the index uses PM2.5, PM10 and NO₂ only (CPCB's minimum of three pollutants);
  set `AQI_USE_MODELLED_O3=1` to include ozone (8-hour mean).
* **No live traffic.** There is no free, open live-traffic feed for India; OSRM gives free-flow times and we apply a
  disclosed rush-hour multiplier.
* **Public Overpass servers are often overloaded.** Mirrors are tried in parallel within an 8 s budget; if all fail,
  places and surroundings come from the offline Geofabrik layer (a day or so older than live OSM), and the answer
  says so. Weather and air quality are always fetched live.
* **OSM coverage varies.** Unnamed or unmapped grounds can be missed; we widen the search, include the place the user
  named, and say when we fall back to area-level timing advice.
* **Weather grids are coarse** (several km), so venues close together often share a forecast; the ranking between them
  is driven by geospatial context and travel.
* **Latency.** On CPU a full answer with the SLM takes roughly 5–40 s (most of it SLM generation; slowest when the first
  draft fails verification and is rewritten). After an idle spell the first request also waits for an instance to
  start. `fast: true` returns the verified template answer in a few seconds.
* **Forecast horizon.** Air quality is forecast ~5 days ahead, weather 7 days; later times get a clear "can't answer
  yet" rather than a guess.
* Health guidance is general precaution, not medical advice.

## Disclosure

* Language model in the running assistant: Bhugoole-geoai-outdoor-slm only, a QLoRA fine-tune of
  Qwen2.5-1.5B-Instruct (open weights, ≤ 3B). No hosted LLM is called.
* Development tools: AI coding assistants for writing code (disclosed, as the rules ask).
* Data sources: listed above.

## Repository layout

```
app/
  main.py          FastAPI endpoints          pipeline.py   orchestration
  parser.py        question understanding     timeparse.py  time expressions -> IST hours
  activities.py    activity knowledge base    profiles.py   personal profiles
  scoring.py       place × hour scoring       compose.py    fact sheet, template, answer assembly
  slm.py           small language model       verifier.py   grounding verifier
  aqi.py, astro.py, geo.py                    CPCB AQI, sun/moon, geodesy
  data/            geocode, overpass, weather, routing, cached HTTP client
  static/index.html  demo UI (map + place × hour heatmap)
finetune/          synthetic data + QLoRA training
hf/MODEL_CARD.md   model card template        scripts/      deploy, push model, live smoke test
tests/             44 offline tests (mocked APIs)
```
