from fastapi.testclient import TestClient

from app.main import app


def test_api_accepts_many_payload_shapes(world, now):
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert len(c.get("/activities").json()["supported"]) >= 10
    for body in ({"question": "golf near Andheri on Saturday", "now": now.isoformat()},
                 {"query": "golf near Andheri on Saturday", "now": now.isoformat()},
                 {"messages": [{"role": "user", "content": "golf near Andheri on Saturday"}], "now": now.isoformat()}):
        r = c.post("/ask", json={**body, "fast": True})
        assert r.status_code == 200 and r.json()["status"] == "ok"
    r = c.get("/ask", params={"q": "golf near Andheri on Saturday", "fast": "1", "format": "text", "now": now.isoformat()})
    assert r.status_code == 200 and "golf" in r.text.lower() or "Greens" in r.text
    assert c.post("/ask", json={}).status_code == 400
    assert c.get("/").status_code == 200


def test_api_rejects_bad_input_and_floods(world, now, monkeypatch):
    from app import main
    c = TestClient(app)
    assert c.post("/ask", json={"question": "run " * 300}).status_code == 400                # too long
    assert c.post("/ask", json={"question": "run near Powai", "lat": 999, "lon": 72}).status_code == 400
    assert c.post("/ask", json={"question": "run", "location": "x" * 500}).status_code == 400
    assert c.post("/ask", content=b"q=" + b"a" * 20000).status_code == 413                    # oversized body
    assert c.post("/ask", json={"messages": [{"role": "user", "content": ["not", "text"]}]}).status_code == 400
    monkeypatch.setattr(main, "LIMITER", main.RateLimiter(2))
    codes = [c.post("/ask", json={"question": "book a flight", "now": now.isoformat()}).status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and codes[2:] == [429, 429]
    r = c.post("/ask", json={"question": "book a flight"})
    assert r.headers["Retry-After"] and r.json()["status"] == "rate_limited"
    assert c.get("/health").status_code == 200                                                # health is never limited
