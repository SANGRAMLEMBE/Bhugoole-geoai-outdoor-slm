"""Shared async HTTP client with a TTL cache, per-host rate limiting and retries.

Every outbound call is recorded in a per-request "source log" so the API can
show exactly which live sources were used to answer a question.
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any
from urllib.parse import urlparse

import httpx

from .. import config

log = logging.getLogger("http")

# Minimum spacing between requests to the same host (seconds). Nominatim's
# policy is an absolute maximum of 1 request per second.
_HOST_MIN_INTERVAL = {
    "nominatim.openstreetmap.org": 1.1,
    "photon.komoot.io": 0.5,
    "overpass-api.de": 1.0,
    "overpass.kumi.systems": 1.0,
    "overpass.private.coffee": 1.0,
    "maps.mail.ru": 1.0,
    "z.overpass-api.de": 1.0,
    "router.project-osrm.org": 0.25,
}


class TTLCache:
    def __init__(self, max_items: int = 2000):
        self._d: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.max_items = max_items

    def get(self, key: str):
        item = self._d.get(key)
        if not item:
            return None
        expires, value = item
        if expires < time.time():
            self._d.pop(key, None)
            return None
        self._d.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: float):
        self._d[key] = (time.time() + ttl, value)
        self._d.move_to_end(key)
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)

    def clear(self):
        self._d.clear()


CACHE = TTLCache()

# Per-request log of sources actually used: list of dicts.
SOURCE_LOG: contextvars.ContextVar[list | None] = contextvars.ContextVar("SOURCE_LOG", default=None)


def record_source(name: str, url: str, cached: bool, ok: bool = True, note: str = "") -> None:
    entries = SOURCE_LOG.get()
    if entries is None:
        return
    entries.append({"source": name, "url": url, "cached": cached, "ok": ok, "note": note})


class Http:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._host_last: dict[str, float] = {}

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=config.HTTP_TIMEOUT,
                headers={"User-Agent": config.USER_AGENT, "Accept": "application/json"},
                follow_redirects=True,
            )
        return self._client

    async def _throttle(self, host: str):
        interval = _HOST_MIN_INTERVAL.get(host, 0.0)
        if not interval:
            return
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._host_last.get(host, 0.0) + interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._host_last[host] = time.monotonic()

    async def get_json(self, url: str, params: dict | None = None, *, ttl: float, source: str,
                       retries: int = 2) -> Any:
        return await self._request("GET", url, params=params, data=None, ttl=ttl, source=source, retries=retries)

    async def post_json(self, url: str, data: dict, *, ttl: float, source: str, retries: int = 1) -> Any:
        return await self._request("POST", url, params=None, data=data, ttl=ttl, source=source, retries=retries)

    async def _request(self, method, url, params, data, ttl, source, retries):
        key = hashlib.sha1(json.dumps([method, url, params, data], sort_keys=True, default=str).encode()).hexdigest()
        hit = CACHE.get(key)
        if hit is not None:
            record_source(source, url, cached=True)
            return hit
        host = urlparse(url).hostname or ""
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                await self._throttle(host)
                if method == "GET":
                    r = await self.client.get(url, params=params)
                else:
                    r = await self.client.post(url, data=data)
                if r.status_code in (429, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                payload = r.json()
                CACHE.set(key, payload, ttl)
                record_source(source, url, cached=False)
                return payload
            except Exception as exc:  # noqa: BLE001 - we retry on anything network-ish
                last_exc = exc
                log.warning("%s %s failed (attempt %d): %s", method, url, attempt + 1, exc)
                if attempt < retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
        record_source(source, url, cached=False, ok=False, note=str(last_exc)[:120])
        raise last_exc  # type: ignore[misc]

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()


HTTP = Http()
