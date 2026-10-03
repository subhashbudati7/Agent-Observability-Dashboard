"""FastAPI service: ingest agent spans."""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .db import Store


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

    return app


app = create_app()
