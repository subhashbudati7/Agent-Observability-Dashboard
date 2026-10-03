from fastapi.testclient import TestClient

from agent_obs.app import create_app
from agent_obs.db import Store, percentile


def make_client():
    return TestClient(create_app(Store()))


def sample(**kw):
    d = {
        "trace_id": "t1",
        "span_id": "s1",
        "kind": "llm",
        "name": "chat",
        "model": "gpt-4o-mini",
        "input_tokens": 1000,
        "output_tokens": 500,
        "latency_ms": 120,
        "status": "ok",
    }
    d.update(kw)
    return d


def test_ingest_returns_201():
    c = make_client()
    r = c.post("/api/events", json=sample())
    assert r.status_code == 201
    body = r.json()
    assert body["trace_id"] == "t1" and body["span_id"] == "s1"


def test_ingest_validation_rejects_bad_payload():
    c = make_client()
    assert c.post("/api/events", json={"trace_id": "t"}).status_code == 422
    bad = sample(kind="email")
    assert c.post("/api/events", json=bad).status_code == 422
    bad = sample(input_tokens=-1)
    assert c.post("/api/events", json=bad).status_code == 422


def test_trace_listing_and_detail():
    c = make_client()
    c.post("/api/events", json=sample())
    c.post("/api/events", json=sample(span_id="s2", kind="tool", name="search", model=None))
    traces = c.get("/api/traces").json()["traces"]
    assert len(traces) == 1
    assert traces[0]["spans"] == 2
    assert traces[0]["tokens"] == 3000
    detail = c.get("/api/traces/t1").json()
    assert len(detail["spans"]) == 2
    assert {s["kind"] for s in detail["spans"]} == {"llm", "tool"}


def test_trace_404():
    assert make_client().get("/api/traces/nope").status_code == 404


def test_stats_aggregation():
    c = make_client()
    c.post("/api/events", json=sample(latency_ms=100))
    c.post("/api/events", json=sample(span_id="s2", latency_ms=300, status="error"))
    s = c.get("/api/stats").json()
    assert s["spans"] == 2
    assert s["traces"] == 1
    assert s["errors"] == 1
    assert s["error_rate"] == 0.5
    assert s["input_tokens"] == 2000
    assert s["output_tokens"] == 1000
    assert s["latency_p50_ms"] == 200.0
    assert s["latency_p95_ms"] == 290.0
    assert s["est_cost_usd"] > 0
    assert s["by_model"][0]["model"] == "gpt-4o-mini"


def test_stats_empty_store():
    s = make_client().get("/api/stats").json()
    assert s["spans"] == 0 and s["error_rate"] == 0.0 and s["est_cost_usd"] == 0


def test_dashboard_index_serves_html():
    r = make_client().get("/")
    assert r.status_code == 200
    assert "Agent Observability Dashboard" in r.text


def test_percentile():
    assert percentile([], 50) == 0.0
    assert percentile([10], 95) == 10
    assert percentile([1, 2, 3, 4], 50) == 2.5
