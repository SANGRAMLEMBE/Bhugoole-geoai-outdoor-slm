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
