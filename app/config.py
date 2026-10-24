"""Runtime configuration. Everything is overridable with environment variables."""
from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


TZ = ZoneInfo(_env("APP_TZ", "Asia/Kolkata"))

# Identify ourselves to OSM services, as their usage policies require.
USER_AGENT = _env(
    "APP_USER_AGENT",
    "Bhugoole-GeoAI-Outdoor-Assistant/1.0 (GeoAI Hackathon 2026; +https://github.com/SANGRAMLEMBE/Bhugoole-geoai-outdoor-slm)",
)

# ---- Data sources (all free and open; see README "Data sources") ----
NOMINATIM_URL = _env("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
PHOTON_URL = _env("PHOTON_URL", "https://photon.komoot.io/api/")
OVERPASS_URLS = [
    u.strip()
    for u in _env(
        "OVERPASS_URLS",
        "https://lz4.overpass-api.de/api/interpreter,"
        "https://overpass-api.de/api/interpreter,"
        "https://z.overpass-api.de/api/interpreter,"
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter,"
        "https://overpass.kumi.systems/api/interpreter,"
        "https://overpass.private.coffee/api/interpreter",
    ).split(",")
    if u.strip()
]
OPEN_METEO_FORECAST_URL = _env("OPEN_METEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast")
OPEN_METEO_AQ_URL = _env("OPEN_METEO_AQ_URL", "https://air-quality-api.open-meteo.com/v1/air-quality")
OPEN_METEO_ELEVATION_URL = _env("OPEN_METEO_ELEVATION_URL", "https://api.open-meteo.com/v1/elevation")
OSRM_URL = _env("OSRM_URL", "https://router.project-osrm.org")

HTTP_TIMEOUT = float(_env("HTTP_TIMEOUT", "25"))

# ---- API limits (protect the endpoint and the free services behind it) ----
RATE_LIMIT_PER_MIN = int(_env("RATE_LIMIT_PER_MIN", "60"))   # questions per client address per minute
MAX_QUESTION_CHARS = int(_env("MAX_QUESTION_CHARS", "500"))
MAX_BODY_BYTES = int(_env("MAX_BODY_BYTES", "16384"))
OVERPASS_HEDGE_S = float(_env("OVERPASS_HEDGE_S", "3"))     # start the next mirror after this long
OVERPASS_BUDGET_S = float(_env("OVERPASS_BUDGET_S", "8"))   # total wait for one Overpass query
OVERPASS_COOLDOWN_S = float(_env("OVERPASS_COOLDOWN_S", "600"))  # skip Overpass this long after a total failure
# Offline OSM layer built from a Geofabrik extract (scripts/build_osm_fallback.py).
# OSM_SOURCE_ORDER: "live_first" (Overpass, then the offline layer), "local_first", or "live_only".
OSM_LOCAL_DB = _env("OSM_LOCAL_DB", str(Path(__file__).resolve().parents[1] / "osm_fallback" / "osm_india.sqlite"))
OSM_SOURCE_ORDER = _env("OSM_SOURCE_ORDER", "live_first")

# ---- Cache TTLs (seconds) ----
TTL_GEOCODE = 7 * 24 * 3600
TTL_OSM = 24 * 3600
TTL_WEATHER = 15 * 60
TTL_ROUTE = 6 * 3600

# ---- Search limits ----
MAX_CANDIDATES = int(_env("MAX_CANDIDATES", "10"))      # places scored hour by hour
MAX_RADIUS_KM = float(_env("MAX_RADIUS_KM", "60"))
FORECAST_DAYS = int(_env("FORECAST_DAYS", "7"))
AQ_FORECAST_DAYS = int(_env("AQ_FORECAST_DAYS", "5"))
# CAMS surface ozone is strongly over-forecast over India; off by default (see app/aqi.py)
AQI_USE_MODELLED_O3 = _env("AQI_USE_MODELLED_O3", "0") == "1"

# ---- Small language model ----
# backend: "llamacpp" (GGUF, fast on CPU), "transformers" (from_pretrained), or "none" (template only)
SLM_BACKEND = _env("SLM_BACKEND", "llamacpp")
SLM_GGUF_REPO = _env("SLM_GGUF_REPO", "WIZARDIAN/Bhugoole-geoai-outdoor-slm")
SLM_GGUF_FILE = _env("SLM_GGUF_FILE", "bhugoole-geoai-outdoor-slm-q4_k_m.gguf")
SLM_GGUF_PATH = _env("SLM_GGUF_PATH", "")  # a local .gguf file instead of the Hub repo above
SLM_HF_MODEL = _env("SLM_HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
SLM_ADAPTER = _env("SLM_ADAPTER", "")  # optional LoRA adapter repo for the transformers backend
SLM_THREADS = int(_env("SLM_THREADS", str(os.cpu_count() or 2)))
SLM_CTX = int(_env("SLM_CTX", "4096"))
SLM_MAX_TOKENS = int(_env("SLM_MAX_TOKENS", "320"))
SLM_DISPLAY_NAME = _env("SLM_DISPLAY_NAME",
                        "Bhugoole-geoai-outdoor-slm (QLoRA of Qwen2.5-1.5B-Instruct, Q4_K_M GGUF via llama.cpp)")
HF_MODEL_REPO = _env("HF_MODEL_REPO", "WIZARDIAN/Bhugoole-geoai-outdoor-slm")
