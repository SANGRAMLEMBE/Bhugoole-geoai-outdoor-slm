"""Small concurrency check of a deployed endpoint (a few dozen requests, not a stress test).

    python scripts/load_check.py --url https://<your-endpoint>

It reuses a handful of questions so that most map, weather and geocoding data comes from the endpoint's cache;
please keep the numbers modest, because every uncached question also calls free public services
(Nominatim allows one request per second) that the assistant depends on.
"""
import argparse
import asyncio
import statistics
import time

import httpx

QUESTIONS = [
    "Is Saturday or Sunday better for a hike near Lonavala?",
    "When is it safe for my parents to walk outdoors tomorrow in Bengaluru?",
    "Where can I take sunset photos this evening around Bandra?",
    "I have mild asthma, where's good for a morning walk near Powai in the next two days?",
]


async def burst(client, url, n, concurrency, fast):
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def one(i):
        async with sem:
            t = time.perf_counter()
            try:
                r = await client.post(url + "/ask", json={"question": QUESTIONS[i % len(QUESTIONS)], "fast": fast})
                status = r.status_code if r.status_code != 200 else r.json().get("status")
            except httpx.HTTPError as exc:
                status = type(exc).__name__
            results.append((time.perf_counter() - t, status))

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(n)))
    wall = time.perf_counter() - t0
    times = sorted(t for t, _ in results)
    codes = {}
    for _, s in results:
        codes[s] = codes.get(s, 0) + 1
    p95 = times[max(0, int(len(times) * 0.95) - 1)]
    print(f"{'fast' if fast else 'full'}: {n} requests, {concurrency} at a time, {wall:.1f} s wall | "
          f"median {statistics.median(times):.1f} s, p95 {p95:.1f} s, max {times[-1]:.1f} s | {codes}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--fast-n", type=int, default=24)
    ap.add_argument("--full-n", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    url = a.url.rstrip("/")
    async with httpx.AsyncClient(timeout=300) as client:
        print("health:", (await client.get(url + "/health")).json()["status"])
        for q in QUESTIONS:  # warm the caches one question at a time
            await client.post(url + "/ask", json={"question": q, "fast": True})
        await burst(client, url, a.fast_n, a.concurrency, fast=True)
        await burst(client, url, a.full_n, a.full_n, fast=False)


if __name__ == "__main__":
    asyncio.run(main())
