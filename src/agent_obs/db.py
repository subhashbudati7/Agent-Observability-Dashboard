"""SQLite storage for agent observability spans."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    model TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ok',
    meta TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    UNIQUE(trace_id, span_id)
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
"""

_lock = threading.Lock()


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolation percentile of a sorted-able sample."""
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


class Store:
    """Thread-safe SQLite span store."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with _lock:
            self._conn.executescript(SCHEMA)

    def add_span(self, span: dict) -> int:
        with _lock:
            cur = self._conn.execute(
                """INSERT OR REPLACE INTO spans
                   (trace_id, span_id, parent_span_id, kind, name, model,
                    input_tokens, output_tokens, latency_ms, status, meta, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    span["trace_id"],
                    span["span_id"],
                    span.get("parent_span_id"),
                    span["kind"],
                    span["name"],
                    span.get("model"),
                    int(span.get("input_tokens", 0)),
                    int(span.get("output_tokens", 0)),
                    float(span.get("latency_ms", 0)),
                    span.get("status", "ok"),
                    json.dumps(span.get("meta", {})),
                    time.time(),
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def traces(self) -> list[dict]:
        with _lock:
            rows = self._conn.execute(
                """SELECT trace_id, COUNT(*) AS spans,
                          COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens,
                          COALESCE(SUM(latency_ms), 0) AS latency_ms,
                          SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
                          MIN(created_at) AS started_at
                   FROM spans GROUP BY trace_id ORDER BY started_at DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def spans_for_trace(self, trace_id: str) -> list[dict]:
        with _lock:
            rows = self._conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY created_at",
                (trace_id,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["meta"] = json.loads(d["meta"])
            out.append(d)
        return out

    def totals(self) -> dict:
        with _lock:
            t = self._conn.execute(
                """SELECT COUNT(*) AS spans,
                          COALESCE(SUM(input_tokens), 0) AS input_tokens,
                          COALESCE(SUM(output_tokens), 0) AS output_tokens,
                          SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors
                   FROM spans"""
            ).fetchone()
            lat = [
                r[0]
                for r in self._conn.execute(
                    "SELECT latency_ms FROM spans ORDER BY latency_ms"
                ).fetchall()
            ]
            by_model = self._conn.execute(
                """SELECT model,
                          COALESCE(SUM(input_tokens), 0) AS input_tokens,
                          COALESCE(SUM(output_tokens), 0) AS output_tokens,
                          COUNT(*) AS spans
                   FROM spans WHERE kind = 'llm' AND model IS NOT NULL
                   GROUP BY model"""
            ).fetchall()
        return {
            "spans": t["spans"],
            "input_tokens": t["input_tokens"],
            "output_tokens": t["output_tokens"],
            "errors": t["errors"] or 0,
            "latencies_ms": lat,
            "by_model": [dict(r) for r in by_model],
        }
