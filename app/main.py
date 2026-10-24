"""HTTP API.

POST /ask   {"question": "...", "lat": 19.12, "lon": 72.91, "location": "Powai", "fast": false}
GET  /ask?q=...                       (same, for quick testing in a browser)
GET  /health, GET /activities, GET / (demo UI), GET /docs (OpenAPI)
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from . import config
from .activities import ACTIVITIES, REQUIRED_TEN
from .data.http import HTTP
from .data.osm_local import LOCAL_OSM
from .pipeline import answer
from .profiles import PROFILES
from .slm import SLM_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the SLM in the background so the server answers health checks immediately.
    threading.Thread(target=SLM_MODEL._ensure, daemon=True).start()
    yield
    await HTTP.aclose()


app = FastAPI(title="GeoAI Outdoor Assistant", version="1.0", lifespan=lifespan,
              description="Where and when to head outdoors: grounded recommendations from live OSM, weather and "
                          "air-quality data, written by a small open language model.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

QUESTION_KEYS = ("question", "query", "q", "prompt", "message", "text", "input")


def _pick_question(d: dict) -> str | None:
    for k in QUESTION_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    msgs = d.get("messages")  # chat-style payloads: [{"role": ..., "content": ...}]
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict):
        return msgs[-1].get("content")
    return None


def _num(v):
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


async def _run(d: dict):
    question = _pick_question(d)
    if not question:
        raise HTTPException(400, "Send a JSON body like {\"question\": \"Where can I go running tomorrow evening "
                                 "near Koregaon Park?\"}")
    now = None
    if d.get("now"):
        try:
            now = datetime.fromisoformat(str(d["now"]))
            if now.tzinfo is None:
                now = now.replace(tzinfo=config.TZ)
        except ValueError:
            raise HTTPException(400, "'now' must be an ISO-8601 timestamp") from None
    fast = str(d.get("fast", "")).lower() in ("1", "true", "yes")
    res = await answer(question, lat=_num(d.get("lat")), lon=_num(d.get("lon") or d.get("lng")),
                       location=d.get("location"), now=now, use_slm=not fast)
    if str(d.get("format", "")).lower() == "text":
        return PlainTextResponse(res["answer"])
    return JSONResponse(res)


@app.post("/ask")
async def ask_post(request: Request):
    try:
        d = await request.json()
    except Exception:  # noqa: BLE001 - also accept form posts and raw text
        body = (await request.body()).decode("utf-8", "ignore").strip()
        form = {}
        if "=" in body and "&" in body or body.startswith(("question=", "q=")):
            from urllib.parse import parse_qs
            form = {k: v[0] for k, v in parse_qs(body).items()}
        d = form or {"question": body}
    if not isinstance(d, dict):
        d = {"question": str(d)}
    return await _run(d)


@app.get("/ask")
async def ask_get(request: Request):
    return await _run(dict(request.query_params))


@app.get("/health")
async def health():
    osm = {"available": LOCAL_OSM.available, "data_date": LOCAL_OSM.meta.get("data_date"),
           "used": "only when the live Overpass API fails" if config.OSM_SOURCE_ORDER == "live_first"
           else config.OSM_SOURCE_ORDER}
    return {"status": "ok", "model": SLM_MODEL.info, "offline_osm_layer": osm,
            "time": datetime.now(config.TZ).isoformat()}


@app.get("/activities")
async def activities():
    return {"required": REQUIRED_TEN,
            "supported": [{"id": a.id, "name": a.name, "default_radius_km": a.default_radius_km,
                           "typical_duration_h": a.duration_h} for a in ACTIVITIES.values()],
            "profiles": [{"id": p.id, "label": p.label} for p in PROFILES.values()]}


@app.get("/")
async def ui():
    return FileResponse(STATIC / "index.html")
