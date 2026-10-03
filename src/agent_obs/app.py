"""FastAPI service: ingest agent spans, query traces, stats, dashboard UI."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .db import Store, percentile
from .prices import estimate_cost


class SpanIn(BaseModel):
    trace_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    parent_span_id: str | None = None
    kind: Literal["llm", "tool"]
    name: str = Field(min_length=1)
    model: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    status: Literal["ok", "error"] = "ok"
    meta: dict = Field(default_factory=dict)


def create_app(store: Store | None = None) -> FastAPI:
    store = store or Store()
    app = FastAPI(title="Agent Observability Dashboard")

    @app.post("/api/events", status_code=201)
    def ingest(span: SpanIn):
        sid = store.add_span(span.model_dump())
        return {"id": sid, "trace_id": span.trace_id, "span_id": span.span_id}

    @app.get("/api/traces")
    def list_traces():
        return {"traces": store.traces()}

    @app.get("/api/traces/{trace_id}")
    def get_trace(trace_id: str):
        spans = store.spans_for_trace(trace_id)
        if not spans:
            raise HTTPException(status_code=404, detail="trace not found")
        return {"trace_id": trace_id, "spans": spans}

    @app.get("/api/stats")
    def get_stats():
        t = store.totals()
        cost = sum(
            estimate_cost(m["model"], m["input_tokens"], m["output_tokens"])
            for m in t["by_model"]
        )
        lat = t["latencies_ms"]
        spans = t["spans"]
        return {
            "spans": spans,
            "traces": len(store.traces()),
            "input_tokens": t["input_tokens"],
            "output_tokens": t["output_tokens"],
            "errors": t["errors"],
            "error_rate": round(t["errors"] / spans, 4) if spans else 0.0,
            "latency_p50_ms": round(percentile(lat, 50), 2),
            "latency_p95_ms": round(percentile(lat, 95), 2),
            "est_cost_usd": round(cost, 6),
            "by_model": t["by_model"],
        }

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    return app


app = create_app()
