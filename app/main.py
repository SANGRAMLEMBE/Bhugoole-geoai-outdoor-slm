"""HTTP API.

POST /ask   {"question": "...", "lat": 19.12, "lon": 72.91, "location": "Powai", "fast": false}
GET  /ask?q=...                       (same, for quick testing in a browser)
GET  /health, GET /activities, GET / (demo UI), GET /docs (OpenAPI)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs

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
log = logging.getLogger("api")
STATIC = Path(__file__).parent / "static"


class RateLimiter:
    """Sliding one-minute window per client address, kept in memory (each container counts on its own)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float:
        """0 if the request may go ahead, else the seconds to wait."""
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > 10_000:  # forget idle clients so memory stays bounded
                self._hits = {k: q for k, q in self._hits.items() if q and now - q[-1] < 60}
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return 60 - (now - q[0])
            q.append(now)
            return 0.0


LIMITER = RateLimiter(config.RATE_LIMIT_PER_MIN)


def _client(request: Request) -> str:
    # Behind a proxy the nearest hop appends the real address last; a client can only forge earlier entries
    fwd = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    return fwd[-1] if fwd else (request.client.host if request.client else "unknown")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the SLM in the background so the server answers health checks immediately.
    threading.Thread(target=SLM_MODEL._ensure, daemon=True).start()
    yield
    await HTTP.aclose()


app = FastAPI(title="GeoAI Outdoor Assistant", version="1.0", lifespan=lifespan,
              description="Where and when to head outdoors: grounded recommendations from live OSM, weather and "
                          "air-quality data, written by a small open language model.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.url.path == "/ask":
        wait = LIMITER.check(_client(request))
        if wait:
            msg = f"Too many questions from this address; please try again in {wait:.0f} s."
            return JSONResponse({"status": "rate_limited", "answer": msg, "detail": msg}, status_code=429,
                                headers={"Retry-After": str(int(wait) + 1)})
        if int(request.headers.get("content-length") or 0) > config.MAX_BODY_BYTES:
            return JSONResponse({"status": "error", "detail": "Request body is too large."}, status_code=413)
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


QUESTION_KEYS = ("question", "query", "q", "prompt", "message", "text", "input")


def _pick_question(d: dict) -> str | None:
    for k in QUESTION_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    msgs = d.get("messages")  # chat-style payloads: [{"role": ..., "content": ...}]
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict) and isinstance(msgs[-1].get("content"), str):
        return msgs[-1]["content"]
    return None


def _num(v, lo: float, hi: float, name: str):
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise HTTPException(400, f"'{name}' must be a number") from None
    if not lo <= x <= hi:
        raise HTTPException(400, f"'{name}' must be between {lo:g} and {hi:g}")
    return x


async def _run(d: dict):
    question = _pick_question(d)
    if not question:
        raise HTTPException(400, "Send a JSON body like {\"question\": \"Where can I go running tomorrow evening "
                                 "near Koregaon Park?\"}")
    if len(question) > config.MAX_QUESTION_CHARS:
        raise HTTPException(400, f"Please keep the question under {config.MAX_QUESTION_CHARS} characters.")
    location = d.get("location")
    if location is not None and (not isinstance(location, str) or len(location) > 120):
        raise HTTPException(400, "'location' must be a place name of at most 120 characters")
    lat = _num(d.get("lat"), -90, 90, "lat")
    lon = _num(d.get("lon") or d.get("lng"), -180, 180, "lon")
    now = None
    if d.get("now"):
        try:
            now = datetime.fromisoformat(str(d["now"]))
            if now.tzinfo is None:
                now = now.replace(tzinfo=config.TZ)
        except ValueError:
            raise HTTPException(400, "'now' must be an ISO-8601 timestamp") from None
    fast = str(d.get("fast", "")).lower() in ("1", "true", "yes")
    try:
        res = await answer(question, lat=lat, lon=lon, location=location, now=now, use_slm=not fast)
    except Exception:  # noqa: BLE001 - never hand a traceback to the caller
        log.exception("answer failed for %r", question[:200])
        return JSONResponse({"status": "error", "answer": "Something went wrong while answering. Please try again "
                             "in a minute.", "detail": "internal error"}, status_code=500)
    if str(d.get("format", "")).lower() == "text":
        return PlainTextResponse(res["answer"])
    return JSONResponse(res)


@app.post("/ask")
async def ask_post(request: Request):
    raw = await request.body()
    if len(raw) > config.MAX_BODY_BYTES:  # chunked uploads carry no content-length
        raise HTTPException(413, "Request body is too large.")
    try:
        d = json.loads(raw)
    except ValueError:  # also accept form posts and raw text
        body = raw.decode("utf-8", "ignore").strip()
        form = {}
        if "=" in body and "&" in body or body.startswith(("question=", "q=")):
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
